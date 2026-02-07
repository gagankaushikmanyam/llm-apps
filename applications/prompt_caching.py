"""
Prompt Caching Lab (KV-cache reuse) — latency before vs after

What "Prompt Caching" means here
-------------------------------
When you repeatedly query an LLM with a large shared prefix (e.g., system prompt,
policies, long document context), you can precompute the model's internal
attention state for that prefix once (the "KV cache"), then reuse it for each
new query. This avoids recomputing the prefix tokens over and over.

This app measures:
- Baseline latency: run the model on (prefix + question) every time
- Cached latency: run prefix once, then reuse its KV cache for each question

Note:
- This is a *local* cache demo (in-process). Real systems do similar caching
  at the server layer across requests.
- Speedup is larger when prefix is long and you have many queries.

Model:
- Default: sshleifer/tiny-gpt2 (fast, CPU-friendly)
- Optional: distilgpt2 (better text, slower)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import streamlit as st

APP_NAME = "Prompt Caching"
APP_DESCRIPTION = (
    "Measure latency speedups from KV-cache prompt caching: baseline (prefix+question each time) "
    "vs cached (prefix prefill once, reuse KV cache for multiple questions)."
)


# -----------------------------
# Dependency checks
# -----------------------------
def _require_transformers() -> None:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception as exc:
        st.error("Missing required ML dependencies.")
        st.code("python -m pip install -r requirements.txt")
        st.exception(exc)
        st.stop()


def _get_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_device_label() -> str:
    import torch

    return "GPU" if torch.cuda.is_available() else "CPU"


# -----------------------------
# Model/tokenizer loading
# -----------------------------
@st.cache_resource(show_spinner=False)
def _load_tokenizer_and_model(model_name: str):
    """
    Cache inference-only model + tokenizer.
    (Safe here because we never train/mutate weights.)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    return tok, model


# -----------------------------
# KV-cache generation helpers
# -----------------------------
def _prefill_kv_cache(model, prefix_ids, device):
    """
    Run the model on the prefix once and return past_key_values.
    """
    import torch

    with torch.no_grad():
        out = model(input_ids=prefix_ids.to(device), use_cache=True)
    return out.past_key_values


def _greedy_generate_with_past(
    model,
    input_ids,
    past_key_values,
    max_new_tokens: int,
    device,
) -> Tuple[List[int], object]:
    """
    Greedy decode max_new_tokens given initial input_ids and optional past_key_values.
    Returns generated token ids (only new tokens) and final past_key_values.
    """
    import torch

    generated: List[int] = []

    with torch.no_grad():
        # 1) consume the provided input_ids (e.g., question ids), updating cache
        out = model(
            input_ids=input_ids.to(device),
            use_cache=True,
            past_key_values=past_key_values,
        )
        past = out.past_key_values
        next_token = int(torch.argmax(out.logits[:, -1, :], dim=-1).item())
        generated.append(next_token)

        # 2) decode further tokens one-by-one
        cur = torch.tensor([[next_token]], dtype=torch.long, device=device)
        for _ in range(max_new_tokens - 1):
            out = model(input_ids=cur, use_cache=True, past_key_values=past)
            past = out.past_key_values
            next_token = int(torch.argmax(out.logits[:, -1, :], dim=-1).item())
            generated.append(next_token)
            cur = torch.tensor([[next_token]], dtype=torch.long, device=device)

    return generated, past


def _run_baseline(
    *,
    model,
    tokenizer,
    prefix: str,
    question: str,
    max_new_tokens: int,
    device,
) -> str:
    """
    Baseline: compute (prefix + question) from scratch every time.
    """
    full_prompt = f"{prefix}\n\nQuestion: {question}\nAnswer:"
    ids = tokenizer(full_prompt, return_tensors="pt").input_ids
    # No cache to start
    gen_ids, _ = _greedy_generate_with_past(
        model=model,
        input_ids=ids,
        past_key_values=None,
        max_new_tokens=max_new_tokens,
        device=device,
    )
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def _run_cached(
    *,
    model,
    tokenizer,
    prefix_past,
    question: str,
    max_new_tokens: int,
    device,
) -> str:
    """
    Cached: reuse prefix KV cache, only run question+Answer: through model.
    """
    q_prompt = f"\n\nQuestion: {question}\nAnswer:"
    q_ids = tokenizer(q_prompt, return_tensors="pt").input_ids
    gen_ids, _ = _greedy_generate_with_past(
        model=model,
        input_ids=q_ids,
        past_key_values=prefix_past,
        max_new_tokens=max_new_tokens,
        device=device,
    )
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


# -----------------------------
# UI / measurement
# -----------------------------
@dataclass
class TimingRow:
    question: str
    baseline_ms: float
    cached_ms: float
    speedup_x: float


def _measure(
    *,
    model,
    tokenizer,
    prefix: str,
    questions: List[str],
    max_new_tokens: int,
    repeats: int,
    device,
) -> Tuple[List[TimingRow], Dict[str, float], Dict[str, str]]:
    """
    Returns:
      rows: per-question timing summary
      totals: overall baseline/cached avg timings and speedup
      samples: one sample output per mode for the first question
    """
    # Build prefix ids once for cached mode
    prefix_ids = tokenizer(prefix, return_tensors="pt").input_ids
    prefix_past = _prefill_kv_cache(model, prefix_ids, device=device)

    rows: List[TimingRow] = []

    # Example outputs (first question only)
    samples: Dict[str, str] = {}

    baseline_all: List[float] = []
    cached_all: List[float] = []

    for qi, q in enumerate(questions):
        # Baseline timings
        b_times: List[float] = []
        c_times: List[float] = []

        for r in range(repeats):
            t0 = time.perf_counter()
            out_b = _run_baseline(
                model=model,
                tokenizer=tokenizer,
                prefix=prefix,
                question=q,
                max_new_tokens=max_new_tokens,
                device=device,
            )
            t1 = time.perf_counter()
            b_times.append((t1 - t0) * 1000.0)

            t0 = time.perf_counter()
            out_c = _run_cached(
                model=model,
                tokenizer=tokenizer,
                prefix_past=prefix_past,
                question=q,
                max_new_tokens=max_new_tokens,
                device=device,
            )
            t1 = time.perf_counter()
            c_times.append((t1 - t0) * 1000.0)

            # Capture first sample outputs
            if qi == 0 and r == 0:
                samples["baseline_sample"] = out_b
                samples["cached_sample"] = out_c

        b_avg = sum(b_times) / max(1, len(b_times))
        c_avg = sum(c_times) / max(1, len(c_times))
        speed = (b_avg / c_avg) if c_avg > 0 else float("inf")

        baseline_all.append(b_avg)
        cached_all.append(c_avg)

        rows.append(
            TimingRow(
                question=q,
                baseline_ms=b_avg,
                cached_ms=c_avg,
                speedup_x=speed,
            )
        )

    baseline_mean = sum(baseline_all) / max(1, len(baseline_all))
    cached_mean = sum(cached_all) / max(1, len(cached_all))
    speedup_mean = (baseline_mean / cached_mean) if cached_mean > 0 else float("inf")

    totals = {
        "baseline_mean_ms": float(baseline_mean),
        "cached_mean_ms": float(cached_mean),
        "speedup_mean_x": float(speedup_mean),
    }
    return rows, totals, samples


def run() -> None:
    _require_transformers()

    st.markdown("### Prompt Caching Lab (Latency)")
    st.write(
        "This app measures **latency before vs after prompt caching** using a local KV-cache reuse demo.\n\n"
        "**When it helps:** you have a **long shared prefix** (policies / document context / system prompt) "
        "and you run multiple queries that reuse it.\n\n"
        "**What you should expect:** cached mode becomes faster as the prefix gets longer."
    )

    col_a, col_b = st.columns([1.3, 1.0])
    with col_a:
        model_name = st.selectbox(
            "Model",
            options=["sshleifer/tiny-gpt2", "distilgpt2"],
            index=0,
            help="tiny-gpt2 is fastest. distilgpt2 is higher quality but slower.",
        )
    with col_b:
        st.text_input("Device", value=_get_device_label(), disabled=True)

    tokenizer, model = _load_tokenizer_and_model(model_name)
    device = _get_device()
    model = model.to(device)

    st.markdown("---")
    st.markdown("### Shared Prefix (what gets cached)")
    prefix = st.text_area(
        "Prefix / context (make this long to see bigger speedups)",
        value=(
            "You are a logistics operations assistant.\n"
            "Follow these policies:\n"
            "1) Be concise.\n"
            "2) Use clear subject lines.\n"
            "3) Prefer actionable language.\n\n"
            "Reference notes:\n"
            "- POD = Proof of Delivery\n"
            "- Accessorial charges include detention, layover, tolls, congestion.\n"
            "- Rate confirmation is required before tendering.\n"
            "- If weather causes delays, communicate updated ETA.\n"
            "- If a pickup is missed, escalate and propose new windows.\n"
        ),
        height=180,
    )

    st.markdown("### Questions (same prefix reused)")
    questions_text = st.text_area(
        "One question per line",
        value=(
            "Write an email subject for a shipment delayed due to weather.\n"
            "Write an email subject requesting a POD for delivery completed yesterday.\n"
            "Write an email subject asking for a tracking update on load 47219.\n"
            "Write an email subject escalating a missed pickup and asking to reschedule.\n"
        ),
        height=120,
    )
    questions = [q.strip() for q in questions_text.splitlines() if q.strip()]

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        max_new_tokens = int(st.slider("Max new tokens", 8, 80, 24, 1))
    with col2:
        repeats = int(st.slider("Repeats (avg timing)", 1, 10, 3, 1))
    with col3:
        st.caption("Tip: Increase prefix length for a clearer caching speedup.")

    run_btn = st.button(
        "Run latency benchmark", type="primary", use_container_width=True
    )
    if not run_btn:
        return

    if not questions:
        st.error("Add at least one question (one per line).")
        return

    with st.spinner("Running benchmark..."):
        rows, totals, samples = _measure(
            model=model,
            tokenizer=tokenizer,
            prefix=prefix,
            questions=questions,
            max_new_tokens=max_new_tokens,
            repeats=repeats,
            device=device,
        )

    st.markdown("---")
    st.markdown("### ✅ Results (Latency)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline avg (ms)", f"{totals['baseline_mean_ms']:.1f}")
    c2.metric("Cached avg (ms)", f"{totals['cached_mean_ms']:.1f}")
    c3.metric("Speedup (x)", f"{totals['speedup_mean_x']:.2f}")

    st.markdown("#### Per-question breakdown")
    table = [
        {
            "Question": r.question,
            "Baseline (ms)": f"{r.baseline_ms:.1f}",
            "Cached (ms)": f"{r.cached_ms:.1f}",
            "Speedup (x)": f"{r.speedup_x:.2f}",
        }
        for r in rows
    ]
    st.dataframe(table, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔎 What got faster, exactly?")
    st.write(
        "- **Baseline** recomputes the model’s attention state for the entire **prefix + question** every time.\n"
        "- **Cached** computes the prefix once (the **prefill**) and reuses its **KV cache** for each new question.\n"
        "- The bigger your prefix (more tokens), the more compute you avoid."
    )

    st.markdown("### 🧪 Sample outputs (first question)")
    colx, coly = st.columns(2)
    with colx:
        st.markdown("**Baseline sample**")
        st.code(samples.get("baseline_sample", ""), language="text")
    with coly:
        st.markdown("**Cached sample**")
        st.code(samples.get("cached_sample", ""), language="text")

    st.info(
        "Note: Output text can differ slightly between baseline and cached due to small numerical differences and "
        "the greedy step boundary. The goal here is to compare **latency**, not quality."
    )
