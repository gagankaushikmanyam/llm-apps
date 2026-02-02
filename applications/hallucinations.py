"""
Hallucinations Lab (Prompting Techniques)

This Streamlit app demonstrates practical ways to reduce hallucinations in LLM outputs
using prompt-level strategies. It is intentionally model-agnostic and runs CPU-friendly.

Techniques compared on the same question:

1) Baseline free-form answer (more likely to hallucinate)
2) JSON-only response (schema forces structured output and reduces rambling)
3) JSON + explicit refusal policy ("UNKNOWN" if not sure) + confidence
4) Context-only answering (answer must be supported by provided context)
5) Self-consistency voting (sample multiple times, choose most frequent parsed answer)

Important:
- Prompting techniques can reduce hallucinations, but cannot guarantee correctness
  without grounding (retrieval, citations, tools, etc.).
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from utils.seed import set_seed

APP_NAME = "Hallucinations Lab: Prompting Techniques"
APP_DESCRIPTION = (
    "Compare baseline vs JSON-structured prompting, refusal policies, context-only answering, "
    "and self-consistency voting to reduce hallucinations."
)


# -----------------------------
# Dependency checks
# -----------------------------
def _require_transformers() -> None:
    """Fail gracefully with helpful UI if required ML libraries are missing."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception as exc:
        st.error("Missing required ML dependencies.")
        st.write("Install requirements with:")
        st.code("python -m pip install -r requirements.txt")
        st.exception(exc)
        st.stop()


def _get_device_label() -> str:
    """Return a user-friendly device label."""
    import torch

    if torch.cuda.is_available():
        return f"GPU (cuda:{torch.cuda.current_device()})"
    return "CPU"


# -----------------------------
# Model / tokenizer loading
# -----------------------------
@st.cache_resource(show_spinner=False)
def _load_tokenizer_and_model(model_name: str):
    """Load and cache tokenizer + model for repeated experiments."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model


# -----------------------------
# Generation helpers
# -----------------------------
def _generate(
    *,
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    seed: int,
    decoding_mode: str,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> str:
    """Generate text with basic repetition controls and selectable decoding."""
    import torch

    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen_common = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )

    with torch.no_grad():
        if decoding_mode == "beam":
            out = model.generate(
                **inputs,
                **gen_common,
                do_sample=False,
                num_beams=4,
                early_stopping=True,
            )
        elif decoding_mode == "sample":
            out = model.generate(
                **inputs,
                **gen_common,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
            )
        else:
            out = model.generate(
                **inputs,
                **gen_common,
                do_sample=False,
            )

    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


def _try_extract_json(text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Attempt to extract a JSON object from a model output.

    Strategy:
    - Find first '{' and last '}' and parse substring
    - Return (obj, error_message)
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None, "No JSON object detected in the output."

    candidate = text[start : end + 1]
    try:
        obj = json.loads(candidate)
        if not isinstance(obj, dict):
            return None, "Parsed JSON is not an object/dict."
        return obj, ""
    except Exception as exc:
        return None, f"JSON parsing failed: {exc}"


# -----------------------------
# Prompt templates
# -----------------------------
def _prompt_baseline(question: str) -> str:
    return "You are a helpful assistant.\n\n" f"Question: {question}\n" "Answer:"


def _prompt_json_only(question: str) -> str:
    """
    JSON output format can reduce hallucination-style rambling by forcing structure.
    It does NOT guarantee correctness, but often improves controllability.
    """
    schema = {
        "answer": "string",
        "confidence": "number from 0 to 1",
        "notes": "string (brief)",
    }
    return (
        "You are a careful assistant.\n"
        "Return ONLY valid JSON. No markdown. No extra text.\n"
        "Use this schema:\n"
        f"{json.dumps(schema, indent=2)}\n\n"
        f"Question: {question}\n"
        "JSON:"
    )


def _prompt_json_with_refusal(question: str) -> str:
    """
    A refusal policy ("UNKNOWN") reduces hallucinations by allowing uncertainty explicitly.
    """
    schema = {
        "answer": "string ('UNKNOWN' if you are not sure)",
        "confidence": "number from 0 to 1",
        "reason": "string (one sentence explaining why)",
    }
    return (
        "You are a factual assistant. If you are not confident, you MUST say UNKNOWN.\n"
        "Return ONLY valid JSON. No markdown. No extra text.\n"
        "Use this schema:\n"
        f"{json.dumps(schema, indent=2)}\n\n"
        f"Question: {question}\n"
        "JSON:"
    )


def _prompt_context_only(question: str, context: str) -> str:
    """
    Context-only answering is a simple RAG-like control without retrieval:
    'Answer ONLY using the provided context; otherwise say UNKNOWN.'
    """
    schema = {
        "answer": "string ('UNKNOWN' if not supported by context)",
        "supported_by_context": "boolean",
        "evidence": "string (quote or excerpt from context, or empty if UNKNOWN)",
    }
    return (
        "You are a grounded assistant.\n"
        "Answer ONLY using the context below. If the answer is not clearly supported, say UNKNOWN.\n"
        "Return ONLY valid JSON. No markdown. No extra text.\n"
        "Use this schema:\n"
        f"{json.dumps(schema, indent=2)}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "JSON:"
    )


# -----------------------------
# Self-consistency voting
# -----------------------------
def _self_consistency_vote(trials: List[Dict[str, Any]]) -> Tuple[str, float]:
    """
    Choose the most frequent 'answer' among successfully parsed JSON trials.

    trials entries:
      { "raw_text": str, "parsed": Optional[dict], "parse_error": str }

    Returns:
      (winning_answer, vote_ratio_over_valid_parses)
    """
    answers: List[str] = []
    for t in trials:
        parsed = t.get("parsed")
        if parsed and isinstance(parsed.get("answer"), str):
            answers.append(parsed["answer"].strip())

    if not answers:
        return "UNKNOWN", 0.0

    counts: Dict[str, int] = {}
    for a in answers:
        counts[a] = counts.get(a, 0) + 1

    winner = max(counts.items(), key=lambda x: x[1])[0]
    ratio = counts[winner] / max(1, len(answers))
    return winner, float(ratio)


# -----------------------------
# Streamlit UI
# -----------------------------
def run() -> None:
    _require_transformers()

    st.markdown("### Hallucinations Lab")
    st.write(
        "This app compares **prompt-level strategies** that can reduce hallucinations by improving "
        "output controllability and encouraging uncertainty. These techniques **do not guarantee correctness** "
        "unless you add grounding (retrieval, citations, tools)."
    )

    col_a, col_b, col_c = st.columns([1.2, 1.0, 1.0])
    with col_a:
        model_name = st.selectbox(
            "Model (CPU-friendly)",
            options=["sshleifer/tiny-gpt2", "distilgpt2"],
            index=0,
        )
    with col_b:
        st.text_input("Device", value=_get_device_label(), disabled=True)
    with col_c:
        seed = int(
            st.number_input("Seed", min_value=0, max_value=10_000_000, value=42, step=1)
        )

    st.markdown("---")
    st.markdown("### Question")
    question = st.text_area(
        "Ask a question (try something factual to observe hallucination behavior):",
        value="What is the capital of Australia?",
        height=80,
    )

    st.markdown("### Generation settings")
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        max_new_tokens = int(st.slider("Max new tokens", 16, 128, 64, 1))
    with col2:
        decoding_mode = st.selectbox("Decoding", ["greedy", "beam", "sample"], index=0)
    with col3:
        temperature = float(st.slider("Temperature (sample only)", 0.1, 1.5, 0.8, 0.1))
    with col4:
        top_p = float(st.slider("Top-p (sample only)", 0.1, 1.0, 0.9, 0.05))

    st.markdown("### Anti-repetition controls")
    col5, col6 = st.columns([1, 1])
    with col5:
        repetition_penalty = float(st.slider("Repetition penalty", 1.0, 2.0, 1.2, 0.05))
    with col6:
        no_repeat_ngram_size = int(st.slider("No-repeat n-gram size", 0, 6, 3, 1))

    st.markdown("---")
    st.markdown("### Technique")
    technique = st.radio(
        "Choose a mitigation strategy",
        options=[
            "Baseline (free-form)",
            "JSON-only format",
            "JSON + refusal policy (UNKNOWN)",
            "Context-only answering (grounded)",
            "Self-consistency voting (JSON + multiple trials)",
        ],
        index=1,
    )

    context = ""
    n_trials = 7

    if technique == "Context-only answering (grounded)":
        context = st.text_area(
            "Context (answer must be supported by this text, otherwise UNKNOWN)",
            value="Australia's capital city is Canberra. Sydney is the largest city.",
            height=100,
        )

    if technique == "Self-consistency voting (JSON + multiple trials)":
        n_trials = int(st.slider("Number of trials", 3, 15, 7, 1))
        st.caption(
            "We generate multiple JSON answers and pick the most frequent `answer` among valid parses."
        )

    run_btn = st.button("Run", type="primary", use_container_width=True)
    if not run_btn:
        return

    tokenizer, model = _load_tokenizer_and_model(model_name)

    # Build prompt
    if technique == "Baseline (free-form)":
        prompt = _prompt_baseline(question)
    elif technique == "JSON-only format":
        prompt = _prompt_json_only(question)
    elif technique == "JSON + refusal policy (UNKNOWN)":
        prompt = _prompt_json_with_refusal(question)
    elif technique == "Context-only answering (grounded)":
        prompt = _prompt_context_only(question, context)
    else:
        prompt = _prompt_json_with_refusal(question)

    st.markdown("---")
    st.markdown("### Output")
    t0 = time.perf_counter()

    if technique != "Self-consistency voting (JSON + multiple trials)":
        text = _generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            seed=seed,
            decoding_mode=decoding_mode,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        elapsed = time.perf_counter() - t0

        st.markdown("**Raw model output**")
        st.code(text, language="text")

        if technique == "Baseline (free-form)":
            st.info(
                "Baseline outputs are unconstrained; the model may sound confident even if incorrect."
            )
        else:
            parsed, err = _try_extract_json(text)
            st.markdown("**Parsed JSON**")
            if parsed is None:
                st.error(err)
                st.write(
                    "Tip: try `decoding=beam` or increase `max_new_tokens` slightly."
                )
            else:
                st.json(parsed)

        st.caption(f"Runtime: {elapsed:.2f}s")

    else:
        trials: List[Dict[str, Any]] = []

        # For self-consistency we vary the seed per trial to diversify generations
        for i in range(n_trials):
            text_i = _generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                seed=seed + i,
                decoding_mode="sample" if decoding_mode != "beam" else "beam",
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
            parsed_i, err_i = _try_extract_json(text_i)
            trials.append(
                {"raw_text": text_i, "parsed": parsed_i, "parse_error": err_i}
            )

        winner, ratio = _self_consistency_vote(trials)
        elapsed = time.perf_counter() - t0

        st.markdown("**Vote result**")
        st.write(f"**Selected answer:** `{winner}`")
        st.write(f"**Vote ratio (among valid parses):** {ratio:.2f}")

        st.markdown("**Trial details**")
        for idx, tr in enumerate(trials, start=1):
            with st.expander(f"Trial {idx}"):
                st.code(tr["raw_text"], language="text")
                if tr["parsed"] is None:
                    st.error(tr["parse_error"])
                else:
                    st.json(tr["parsed"])

        st.caption(f"Runtime: {elapsed:.2f}s")

    st.markdown("---")
    st.markdown("### Notes (What to learn from this)")
    st.write(
        "- **JSON format** improves controllability: outputs are easier to parse and validate.\n"
        "- **Refusal policies** reduce hallucinations by allowing the model to say `UNKNOWN`.\n"
        "- **Context-only answering** is a simple grounding rule (RAG-like behavior without retrieval).\n"
        "- **Self-consistency** helps when single generations are unstable (vote over multiple samples).\n"
        "\nFor production-grade factuality, add **retrieval**, **citations**, and/or **tool calls**."
    )
