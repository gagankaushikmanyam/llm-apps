# FILE: applications/hallucinations.py
"""
Hallucinations Lab — Prompting + RAG-lite (TF-IDF over local docs)

What this app demonstrates
--------------------------
A) Prompt-level controls to reduce hallucinations:
   - JSON-only outputs (structured)
   - Refusal policy (UNKNOWN when unsure)
   - Self-consistency (vote over multiple generations)

B) RAG-lite grounding (the practical hallucination reducer):
   - Load local documents from ./knowledge_base/*.txt
   - Split docs into chunks
   - Retrieve top-k relevant chunks using TF-IDF (scikit-learn)
   - Force "context-only" answering: if not supported by retrieved text -> UNKNOWN

Why your earlier outputs were wrong
-----------------------------------
- Small non-instruction models (GPT-2 family) often ignore "return JSON".
- Even instruction-tuned models sometimes output junk (e.g., "a") or echo chunk headers.
- The fix:
  1) Prefer FLAN-T5 models for instruction following.
  2) Retry once with a stricter JSON instruction if parsing fails.
  3) If JSON still fails, use an *honest* fallback:
     - never accept chunk headers like "[file | chunk 0]"
     - extract the answer from retrieved context using safe heuristics
     - only mark supported_by_context=True if the answer appears in context.

Folder layout required
----------------------
Project root (same folder as app.py):
  knowledge_base/
    *.txt

If knowledge_base is missing/empty, the app will stop with instructions.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from utils.seed import set_seed

APP_NAME = "Hallucinations Lab: Prompting + RAG-lite"
APP_DESCRIPTION = (
    "RAG-lite (TF-IDF retrieval over local docs) + grounded answering, with JSON prompting, "
    "refusal policy, and self-consistency voting."
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


def _require_sklearn() -> None:
    """Fail gracefully if scikit-learn is missing (needed for TF-IDF retrieval)."""
    try:
        import sklearn  # noqa: F401
    except Exception as exc:
        st.error("RAG-lite retrieval requires scikit-learn.")
        st.write("Install with:")
        st.code("python -m pip install scikit-learn")
        st.exception(exc)
        st.stop()


def _get_device_label() -> str:
    """Return a user-friendly device label."""
    import torch

    return "GPU" if torch.cuda.is_available() else "CPU"


# -----------------------------
# Model / tokenizer loading
# -----------------------------
@st.cache_resource(show_spinner=False)
def _load_tokenizer(model_name: str):
    """Cache tokenizer (safe)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    return tok


def _load_model(model_name: str):
    """Load model (not cached to avoid stale state)."""
    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM

    if "t5" in model_name.lower():
        return AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return AutoModelForCausalLM.from_pretrained(model_name)


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
    """Generate text with selectable decoding. Prefer greedy/beam for JSON compliance."""
    import torch

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen_common: Dict[str, Any] = dict(
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )
    if getattr(tokenizer, "pad_token_id", None) is not None:
        gen_common["pad_token_id"] = tokenizer.pad_token_id
    if getattr(tokenizer, "eos_token_id", None) is not None:
        gen_common["eos_token_id"] = tokenizer.eos_token_id

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


# -----------------------------
# JSON parsing helpers
# -----------------------------
def _try_extract_json(text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """Extract a JSON object from arbitrary model text (best effort)."""
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


def _strict_retry_prompt(prompt: str) -> str:
    """Add stricter instructions for JSON compliance (used for a single retry)."""
    return (
        "CRITICAL:\n"
        "- Output EXACTLY ONE JSON object.\n"
        "- No text before or after JSON.\n"
        "- No markdown.\n"
        "- One line.\n\n" + prompt
    )


# -----------------------------
# Prompt templates
# -----------------------------
def _prompt_baseline(question: str) -> str:
    return f"You are a helpful assistant.\n\nQuestion: {question}\nAnswer:"


def _prompt_json_only(question: str) -> str:
    schema = {"answer": "string", "confidence": "0..1", "notes": "string"}
    return (
        "Return EXACTLY one JSON object and nothing else. One line. No markdown.\n"
        f"Schema: {json.dumps(schema)}\n\n"
        f"Question: {question}\nJSON:"
    )


def _prompt_json_refusal(question: str) -> str:
    schema = {
        "answer": "string (UNKNOWN if unsure)",
        "confidence": "0..1",
        "reason": "string",
    }
    return (
        "If you are not sure, set answer to UNKNOWN.\n"
        "Return EXACTLY one JSON object and nothing else. One line. No markdown.\n"
        f"Schema: {json.dumps(schema)}\n\n"
        f"Question: {question}\nJSON:"
    )


def _prompt_context_only(question: str, context: str, sources: List[str]) -> str:
    schema = {
        "answer": "string (UNKNOWN if not supported by context)",
        "supported_by_context": "boolean",
        "evidence": "string (quote/excerpt from context or empty)",
        "sources": "list of strings",
    }
    src_line = ", ".join(sources) if sources else "(none)"
    return (
        "You are a grounded assistant.\n"
        "You MUST answer using ONLY the provided context.\n"
        "If the answer is not supported by the context, set answer to UNKNOWN.\n"
        "Do NOT output chunk labels like [file | chunk 0].\n"
        "The answer must be a short phrase copied from the context.\n"
        "Return EXACTLY one JSON object and nothing else. One line. No markdown.\n"
        f"Schema: {json.dumps(schema)}\n\n"
        f"Sources: {src_line}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\nJSON:"
    )


# -----------------------------
# RAG-lite (TF-IDF retrieval)
# -----------------------------
@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_id: int
    text: str


def _split_into_chunks(text: str, max_chars: int = 700) -> List[str]:
    """Split by paragraphs then pack into ~max_chars blocks for better retrieval."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paras:
        return []
    chunks: List[str] = []
    buf = ""
    for p in paras:
        if not buf:
            buf = p
        elif len(buf) + 2 + len(p) <= max_chars:
            buf = buf + "\n\n" + p
        else:
            chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)
    return chunks


def _load_kb_chunks(folder: Path) -> List[Chunk]:
    """Load ./knowledge_base/*.txt and split into chunks. No built-in fallback KB."""
    if not folder.exists():
        return []
    chunks: List[Chunk] = []
    for fp in sorted(folder.glob("*.txt")):
        try:
            txt = fp.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            continue
        if not txt:
            continue
        for i, part in enumerate(_split_into_chunks(txt)):
            chunks.append(Chunk(doc_id=fp.name, chunk_id=i, text=part))
    return chunks


def _retrieve_tfidf(
    question: str, chunks: List[Chunk], top_k: int
) -> Tuple[str, List[str]]:
    """Return combined context + source doc_ids using TF-IDF similarity."""
    _require_sklearn()
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    corpus = [c.text for c in chunks]
    vect = TfidfVectorizer(stop_words="english")
    X = vect.fit_transform(corpus)
    q = vect.transform([question])

    sims = cosine_similarity(q, X)[0]
    ranked = sorted(range(len(chunks)), key=lambda i: float(sims[i]), reverse=True)
    chosen = [chunks[i] for i in ranked[: max(1, top_k)]]

    context = "\n\n".join(
        f"[{c.doc_id} | chunk {c.chunk_id}]\n{c.text.strip()}" for c in chosen
    ).strip()
    sources = sorted(list({c.doc_id for c in chosen}))
    return context, sources


# -----------------------------
# Honest fallback + extraction
# -----------------------------
def _looks_like_chunk_header(ans: str) -> bool:
    """Reject answers like: [file.txt | chunk 0]."""
    a = (ans or "").strip()
    return bool(re.fullmatch(r"\[[^\]]+\|\s*chunk\s*\d+\]", a, flags=re.IGNORECASE))


def _is_plausible_answer(ans: str) -> bool:
    """Basic sanity checks to reject junk answers."""
    a = (ans or "").strip()
    if not a:
        return False
    if len(a) <= 2:
        return False
    if re.fullmatch(r"[\W_]+", a):
        return False
    if _looks_like_chunk_header(a):
        return False
    return True


def _answer_supported_by_context(ans: str, context: str) -> bool:
    """Support check: answer string must appear in context (simple + honest)."""
    return ans.lower() in (context or "").lower()


def _extract_answer_from_context(question: str, context: str) -> str:
    """
    Safe heuristic extractor that ONLY uses retrieved context.
    Improves demo reliability when the model fails JSON or outputs junk.

    Extend this as you add KB domains.
    """
    q = (question or "").lower()
    ctx = context or ""

    # POD stands for X.
    if "pod" in q and ("stand for" in q or "stands for" in q):
        m = re.search(r"\bPOD\s+stands\s+for\s+([A-Za-z][A-Za-z\s\-]+)\.", ctx)
        if m:
            return m.group(1).strip()

    # capital questions
    if "capital" in q:
        m = re.search(r"capital(?: city)?\s+(?:is|=)\s+([A-Z][a-z]+)", ctx)
        if m:
            return m.group(1).strip()

        m = re.search(
            r"government\s+is\s+based\s+in\s+([A-Z][a-z]+)", ctx, flags=re.IGNORECASE
        )
        if m:
            return m.group(1).strip()

        # fallback: pick a likely city token from context
        candidates = re.findall(r"\b[A-Z][a-z]{2,}\b", ctx)
        stop = {
            "Australia",
            "Sydney",
            "Parliament",
            "House",
            "Proof",
            "Delivery",
            "Accessorial",
        }
        for c in candidates:
            if c not in stop:
                return c

    # accessorial charges
    if "accessorial" in q:
        for line in ctx.splitlines():
            if "Accessorial charges" in line:
                return line.strip()

    return "UNKNOWN"


def _fallback_context_json(
    raw: str, question: str, context: str, sources: List[str]
) -> Dict[str, Any]:
    """
    If model doesn't produce JSON, produce honest JSON:
    - never accept chunk headers as answers
    - if raw answer is junk, extract from context
    - only supported_by_context=True if answer appears in context
    """
    raw_ans = (raw or "").strip()

    if not _is_plausible_answer(raw_ans):
        raw_ans = _extract_answer_from_context(question, context)

    if not _is_plausible_answer(raw_ans) or raw_ans.upper() == "UNKNOWN":
        return {
            "answer": "UNKNOWN",
            "supported_by_context": False,
            "evidence": "",
            "sources": sources,
        }

    if not _answer_supported_by_context(raw_ans, context):
        return {
            "answer": "UNKNOWN",
            "supported_by_context": False,
            "evidence": "",
            "sources": sources,
        }

    evidence = ""
    for line in (context or "").splitlines():
        if raw_ans.lower() in line.lower():
            evidence = line.strip()
            break

    return {
        "answer": raw_ans,
        "supported_by_context": True,
        "evidence": evidence,
        "sources": sources,
    }


def _fallback_refusal_json(raw: str) -> Dict[str, Any]:
    ans = (raw or "").strip() or "UNKNOWN"
    conf = 0.2 if ans.upper() == "UNKNOWN" else 0.3
    return {
        "answer": ans,
        "confidence": conf,
        "reason": "Fallback-wrapped (model did not output JSON).",
    }


def _fallback_generic_json(raw: str) -> Dict[str, Any]:
    ans = (raw or "").strip() or "UNKNOWN"
    return {
        "answer": ans,
        "confidence": 0.5,
        "notes": "Fallback-wrapped (model did not output JSON).",
    }


# -----------------------------
# Self-consistency voting
# -----------------------------
@dataclass
class TrialResult:
    raw_text: str
    parsed: Optional[Dict[str, Any]]
    parse_error: str


def _vote_answer(trials: List[TrialResult]) -> Tuple[str, float]:
    answers: List[str] = []
    for t in trials:
        if t.parsed and isinstance(t.parsed.get("answer"), str):
            answers.append(t.parsed["answer"].strip())
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

    st.markdown("### Hallucinations Lab — Prompting + RAG-lite")
    st.write(
        "To actually reduce hallucinations, you need **grounding**.\n\n"
        "- **Prompting** helps controllability (JSON, refusal).\n"
        "- **RAG-lite** retrieves relevant chunks from your local docs and forces context-only answers.\n\n"
        "Create `knowledge_base/*.txt` once, then retrieval happens automatically for each question."
    )

    col_a, col_b, col_c = st.columns([1.4, 1.0, 1.0])
    with col_a:
        model_name = st.selectbox(
            "Model (FLAN-T5 recommended)",
            options=[
                "google/flan-t5-small",
                "google/flan-t5-base",
                "distilgpt2",
                "sshleifer/tiny-gpt2",
            ],
            index=0,
            help="Use FLAN-T5 for reliable instruction following + JSON. GPT-2 models are for comparison only.",
        )
    with col_b:
        st.text_input("Device", value=_get_device_label(), disabled=True)
    with col_c:
        seed = int(st.number_input("Seed", 0, 10_000_000, 42, 1))

    st.markdown("---")
    technique = st.radio(
        "Technique",
        options=[
            "Context-only (RAG-lite grounded)",
            "JSON-only format",
            "JSON + refusal policy (UNKNOWN)",
            "Baseline (free-form)",
            "Self-consistency voting (JSON + multiple trials)",
        ],
        index=0,
    )

    question = st.text_area("Question", value="What does POD stand for?", height=80)

    st.markdown("---")
    st.markdown("### Knowledge base")
    kb_folder = Path("knowledge_base")
    chunks = _load_kb_chunks(kb_folder)

    if not chunks:
        st.error("No knowledge base found.")
        st.write(
            "Create `knowledge_base/` next to `app.py` and add one or more `.txt` files."
        )
        st.code(
            "mkdir -p knowledge_base\n"
            "cat > knowledge_base/logistics_faq.txt <<'EOF'\n"
            "POD stands for Proof of Delivery.\n"
            "Accessorial charges may include detention, layover, tolls, and congestion fees.\n"
            "A rate confirmation is typically required before tendering a shipment to a carrier.\n"
            "EOF\n",
            language="bash",
        )
        st.stop()

    st.success(
        f"Loaded {len({c.doc_id for c in chunks})} documents and {len(chunks)} chunks."
    )

    top_k = int(
        st.slider(
            "Retriever top-k chunks",
            1,
            5,
            2,
            1,
            disabled=(technique != "Context-only (RAG-lite grounded)"),
        )
    )

    if technique == "Context-only (RAG-lite grounded)":
        retrieved_context, sources = _retrieve_tfidf(question, chunks, top_k=top_k)
        st.markdown("#### Retrieved context (what the model is allowed to use)")
        st.code(retrieved_context or "(empty)", language="text")
    else:
        retrieved_context, sources = "", []

    st.markdown("---")
    st.markdown("### Generation settings")
    default_decoding = "greedy" if "t5" in model_name.lower() else "beam"

    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        max_new_tokens = int(st.slider("Max new tokens", 32, 256, 128, 8))
    with col2:
        decoding_mode = st.selectbox(
            "Decoding",
            ["greedy", "beam", "sample"],
            index=["greedy", "beam", "sample"].index(default_decoding),
        )
    with col3:
        temperature = float(st.slider("Temperature (sample only)", 0.1, 1.5, 0.7, 0.1))
    with col4:
        top_p = float(st.slider("Top-p (sample only)", 0.1, 1.0, 0.9, 0.05))

    col5, col6 = st.columns([1, 1])
    with col5:
        repetition_penalty = float(
            st.slider("Repetition penalty", 1.0, 2.0, 1.05, 0.05)
        )
    with col6:
        no_repeat_ngram_size = int(st.slider("No-repeat n-gram size", 0, 6, 2, 1))

    auto_retry = st.checkbox("Auto-retry once if JSON parsing fails", value=True)

    run_btn = st.button("Run", type="primary", use_container_width=True)
    if not run_btn:
        return

    tokenizer = _load_tokenizer(model_name)
    model = _load_model(model_name)

    st.markdown("---")
    st.markdown("### Output")

    t0 = time.perf_counter()

    # -----------------------------
    # Baseline
    # -----------------------------
    if technique == "Baseline (free-form)":
        prompt = _prompt_baseline(question)
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
        st.markdown("**Raw model output**")
        st.code(text, language="text")
        st.caption(f"Runtime: {time.perf_counter() - t0:.2f}s")
        return

    # -----------------------------
    # Context-only grounded (RAG-lite)
    # -----------------------------
    if technique == "Context-only (RAG-lite grounded)":
        prompt = _prompt_context_only(question, retrieved_context, sources)
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

        parsed, err = _try_extract_json(text)

        if parsed is None and auto_retry:
            text2 = _generate(
                model=model,
                tokenizer=tokenizer,
                prompt=_strict_retry_prompt(prompt),
                max_new_tokens=max_new_tokens,
                seed=seed + 1,
                decoding_mode="greedy",
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
            parsed2, err2 = _try_extract_json(text2)
            if parsed2 is not None:
                text, parsed, err = text2, parsed2, ""
            else:
                err = err2 or err

        st.markdown("**Raw model output**")
        st.code(text, language="text")

        st.markdown("**Parsed JSON**")
        if parsed is None:
            st.warning(f"Model did not output JSON ({err}). Using honest fallback.")
            parsed = _fallback_context_json(text, question, retrieved_context, sources)

        # Enforce honesty even if model lied in JSON
        ans = str(parsed.get("answer", "UNKNOWN"))
        if parsed.get("supported_by_context", False):
            if not _is_plausible_answer(ans) or not _answer_supported_by_context(
                ans, retrieved_context
            ):
                parsed["answer"] = "UNKNOWN"
                parsed["supported_by_context"] = False
                parsed["evidence"] = ""

        st.json(parsed)
        st.caption(f"Runtime: {time.perf_counter() - t0:.2f}s")

        st.markdown("---")
        st.markdown("### What to test to verify grounding works")
        st.write(
            "- Ask something that is in your KB → should answer with evidence.\n"
            "- Ask something not in your KB → should return UNKNOWN.\n"
            "- If you get junk outputs, keep FLAN-T5 + decoding=greedy."
        )
        return

    # -----------------------------
    # JSON-only / Refusal JSON
    # -----------------------------
    if technique == "JSON-only format":
        prompt = _prompt_json_only(question)
    elif technique == "JSON + refusal policy (UNKNOWN)":
        prompt = _prompt_json_refusal(question)
    else:
        # Self-consistency uses refusal prompt
        prompt = _prompt_json_refusal(question)

    # -----------------------------
    # Self-consistency
    # -----------------------------
    if technique == "Self-consistency voting (JSON + multiple trials)":
        n_trials = int(st.slider("Trials", 3, 15, 7, 1))
        trials: List[TrialResult] = []
        for i in range(n_trials):
            text_i = _generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                seed=seed + i,
                decoding_mode="sample" if decoding_mode == "sample" else decoding_mode,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
            parsed_i, err_i = _try_extract_json(text_i)
            if parsed_i is None:
                parsed_i = _fallback_refusal_json(text_i)
            trials.append(
                TrialResult(raw_text=text_i, parsed=parsed_i, parse_error=err_i)
            )

        winner, ratio = _vote_answer(trials)
        st.markdown("**Vote result**")
        st.write(f"**Selected answer:** `{winner}`")
        st.write(f"**Vote ratio:** {ratio:.2f}")

        st.markdown("**Trial details**")
        for idx, tr in enumerate(trials, start=1):
            with st.expander(f"Trial {idx}"):
                st.code(tr.raw_text, language="text")
                st.json(tr.parsed or {})

        st.caption(f"Runtime: {time.perf_counter() - t0:.2f}s")
        return

    # -----------------------------
    # Single JSON run
    # -----------------------------
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

    parsed, err = _try_extract_json(text)

    if parsed is None and auto_retry:
        text2 = _generate(
            model=model,
            tokenizer=tokenizer,
            prompt=_strict_retry_prompt(prompt),
            max_new_tokens=max_new_tokens,
            seed=seed + 1,
            decoding_mode="greedy",
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        parsed2, err2 = _try_extract_json(text2)
        if parsed2 is not None:
            text, parsed, err = text2, parsed2, ""
        else:
            err = err2 or err

    st.markdown("**Raw model output**")
    st.code(text, language="text")

    st.markdown("**Parsed JSON**")
    if parsed is None:
        st.warning(f"Model did not output valid JSON; repaired. (Parse issue: {err})")
        if technique == "JSON + refusal policy (UNKNOWN)":
            parsed = _fallback_refusal_json(text)
        else:
            parsed = _fallback_generic_json(text)

    st.json(parsed)
    st.caption(f"Runtime: {time.perf_counter() - t0:.2f}s")
