# FILE: applications/langchain_orchestration.py
"""
LangChain Orchestration Demo (Taxes) — Robust Structured Outputs (KV parsing)

This app demonstrates "orchestration" using a multi-step workflow:
1) Classification        -> category + risk + rationale
2) Clarifying Questions  -> missing info to reduce wrong assumptions
3) Checklist + Docs      -> actionable checklist + documents list
4) Email Draft           -> drafts an email using the artifacts above

Why KV (key=value) instead of JSON?
----------------------------------
Small instruction models (even FLAN-T5) often fail strict JSON formatting:
- broken quotes
- missing braces
- empty outputs due to prompt truncation

Instead, we ask the model to return a small set of "key=value" lines.
We parse them deterministically and build structured objects ourselves.

This makes the app reliable and the orchestration visible.

Disclaimer
----------
This is a demo and NOT tax advice.
"""

from __future__ import annotations

import re
import textwrap
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import streamlit as st

APP_NAME = "LangChain Orchestration: Tax Workflow Assistant"
APP_DESCRIPTION = (
    "Multi-step tax workflow using HF Transformers (FLAN-T5) with robust structured outputs. "
    "Classification → Questions → Checklist → Email. Uses key=value parsing (no fragile JSON)."
)

DISCLAIMER = (
    "⚠️ **Disclaimer:** This is a workflow demo and is **not tax advice**. "
    "Consult a qualified tax professional for real decisions."
)


# -----------------------------
# Dependency checks
# -----------------------------
def _require_deps() -> None:
    """Fail gracefully if required deps are missing."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception as exc:
        st.error("Missing dependencies for this app.")
        st.write("Install requirements with:")
        st.code("python -m pip install -r requirements.txt")
        st.exception(exc)
        st.stop()


def _device_label() -> str:
    import torch

    return "GPU (cuda)" if torch.cuda.is_available() else "CPU"


# -----------------------------
# HF pipeline creation + invoke
# -----------------------------
@st.cache_resource(show_spinner=False)
def _build_hf_pipe(model_name: str):
    """
    Build and cache a HF text2text-generation pipeline.
    device_map='auto' uses GPU if available; works on CPU too.
    """
    from transformers import pipeline

    return pipeline(
        task="text2text-generation",
        model=model_name,
        device_map="auto",
    )


def _hf_invoke(pipe, prompt: str, *, max_new_tokens: int, num_beams: int) -> str:
    """
    Reliable HF call returning plain text.
    Retries once if output is empty.
    """
    outputs = pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        do_sample=False,
        early_stopping=True,
    )

    text = ""
    if isinstance(outputs, list) and outputs and isinstance(outputs[0], dict):
        text = outputs[0].get("generated_text", "") or outputs[0].get("text", "") or ""
    else:
        text = str(outputs)

    text = (text or "").strip()

    # Retry once if empty (common with truncation or too short generation)
    if not text:
        outputs = pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            num_beams=max(1, num_beams),
            do_sample=False,
            early_stopping=True,
            min_new_tokens=24,
        )
        if isinstance(outputs, list) and outputs and isinstance(outputs[0], dict):
            text = (
                outputs[0].get("generated_text", "") or outputs[0].get("text", "") or ""
            )
        text = (text or "").strip()

    return text


# -----------------------------
# Intake model
# -----------------------------
@dataclass(frozen=True)
class TaxIntake:
    country: str
    year: str
    employment: str
    residency_status: str
    income_sources: str
    deductions: str
    special_events: str
    goal: str


def _format_intake(i: TaxIntake) -> str:
    """
    Compact intake to avoid exceeding FLAN-T5 input length.
    """
    return textwrap.dedent(
        f"""
        Country: {i.country}
        Year: {i.year}
        Employment: {i.employment}
        Residency: {i.residency_status}
        Income: {i.income_sources}
        Deductions: {i.deductions}
        Events: {i.special_events}
        Goal: {i.goal}
        """.strip()
    )


# -----------------------------
# KV parsing helpers
# -----------------------------
_KV_LINE_RE = re.compile(r"^\s*([A-Za-z0-9_]+)\s*=\s*(.*?)\s*$")


def _parse_kv_lines(raw: str) -> Dict[str, str]:
    """
    Parse key=value lines from model output.

    Lines not matching key=value are ignored.
    Keys are normalized to lowercase.
    """
    out: Dict[str, str] = {}
    for line in (raw or "").splitlines():
        m = _KV_LINE_RE.match(line.strip())
        if not m:
            continue
        key = m.group(1).strip().lower()
        val = m.group(2).strip()
        out[key] = val
    return out


def _run_step_kv(
    pipe,
    *,
    step_name: str,
    prompt: str,
    required_keys: List[str],
    max_new_tokens: int,
    num_beams: int,
) -> Tuple[Dict[str, Any], str, str]:
    """
    Run a step where the model returns key=value lines.
    Returns (result_dict, note, raw_output).
    """
    raw = _hf_invoke(pipe, prompt, max_new_tokens=max_new_tokens, num_beams=num_beams)
    parsed = _parse_kv_lines(raw)

    missing = [k for k in required_keys if k.lower() not in parsed]
    if missing:
        note = (
            f"{step_name}: Missing keys {missing}. "
            "The model did not follow the required key=value format."
        )
        return {"raw_output": (raw or "").strip()}, note, raw

    result = {k: parsed[k.lower()] for k in required_keys}
    return result, "", raw


# -----------------------------
# Prompt builders (KV format)
# -----------------------------
def _prompt_classify_kv(intake_text: str) -> str:
    return f"""
You are a tax workflow triage assistant.

Return EXACTLY 3 lines in this format (key=value):
category=employment_only|mixed_income|self_employed|crypto|foreign_income|real_estate|student|other
risk=low|medium|high
rationale=one short sentence

Rules:
- Do NOT copy the Goal text.
- Choose the closest category.
- Be concise.

Scenario:
{intake_text}

Output:
""".strip()


def _prompt_questions_kv(intake_text: str, classification: Dict[str, Any]) -> str:
    # Keep it short; only include minimal classification fields
    cat = str(classification.get("category", "other"))
    risk = str(classification.get("risk", "medium"))
    return f"""
You are a careful assistant that prepares tax workflows.

Return EXACTLY 5 lines (key=value):
q1=...
q2=...
q3=...
q4=...
q5=...

Rules:
- Questions must reduce uncertainty and prevent wrong assumptions.
- Do NOT copy the Goal text.
- Keep each question short.

Scenario:
{intake_text}

Classification: category={cat}, risk={risk}

Output:
""".strip()


def _prompt_checklist_kv(
    intake_text: str, classification: Dict[str, Any], questions: Dict[str, Any]
) -> str:
    cat = str(classification.get("category", "other"))
    # Provide just the questions list (short)
    qs = [questions.get(f"q{i}", "") for i in range(1, 6)]
    qs_text = "; ".join([q for q in qs if isinstance(q, str) and q.strip()])

    return f"""
You are a tax workflow assistant.

Return EXACTLY these lines (key=value):
step1=...
step2=...
step3=...
doc1=...
doc2=...
doc3=...
next_action=one sentence

Rules:
- Steps must be actionable.
- Docs must be specific (forms/receipts/statements).
- Do NOT copy the Goal text.
- Use the scenario and clarifying questions to tailor the checklist.

Scenario:
{intake_text}

Classification: category={cat}
Key questions: {qs_text}

Output:
""".strip()


def _prompt_email(
    intake_text: str, questions: Dict[str, Any], checklist: Dict[str, Any]
) -> str:
    docs = [
        checklist.get("doc1", ""),
        checklist.get("doc2", ""),
        checklist.get("doc3", ""),
    ]
    steps = [
        checklist.get("step1", ""),
        checklist.get("step2", ""),
        checklist.get("step3", ""),
    ]
    qs = [questions.get("q1", ""), questions.get("q2", ""), questions.get("q3", "")]

    docs_bullets = "\n".join(
        [f"- {d}" for d in docs if isinstance(d, str) and d.strip()]
    )
    steps_bullets = "\n".join(
        [f"- {s}" for s in steps if isinstance(s, str) and s.strip()]
    )
    qs_bullets = "\n".join([f"- {q}" for q in qs if isinstance(q, str) and q.strip()])

    return f"""
Write a concise professional email to a tax advisor.

Must include these sections (in this order):
1) Subject line
2) Short summary (2-3 lines)
3) Documents I can provide (bullets)
4) Clarifying questions I want to ask (bullets)
5) Requested next step (meeting / review)

Scenario:
{intake_text}

Checklist steps:
{steps_bullets}

Documents:
{docs_bullets}

Clarifying questions:
{qs_bullets}

Email:
""".strip()


# -----------------------------
# UI helpers
# -----------------------------
def _kv_table(title: str, data: Dict[str, Any], keys: List[str]) -> None:
    st.markdown(f"**{title}**")
    rows = []
    for k in keys:
        rows.append({"Field": k, "Value": str(data.get(k, ""))})
    st.table(rows)


def run() -> None:
    _require_deps()

    st.markdown("## 🧪 LangChain-style Orchestration — Tax Workflow Assistant")
    st.info(DISCLAIMER)
    st.caption(f"Device: **{_device_label()}**")

    st.markdown("---")
    c1, c2, c3 = st.columns([1.4, 1.0, 1.0])
    with c1:
        model_name = st.selectbox(
            "Model (recommended: flan-t5-base)",
            options=["google/flan-t5-base", "google/flan-t5-small"],
            index=0,
        )
    with c2:
        max_new_tokens = int(st.slider("Max new tokens", 128, 512, 256, 32))
    with c3:
        num_beams = int(st.slider("Beams", 1, 8, 4, 1))

    pipe = _build_hf_pipe(model_name)

    st.markdown("---")
    st.markdown("### 1) Intake")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        country = st.text_input("Country", value="Germany")
        year = st.text_input("Tax year", value="2025")
        employment = st.selectbox(
            "Employment",
            options=["Full-time employee", "Student", "Self-employed", "Mixed/Other"],
            index=0,
        )
        residency = st.selectbox(
            "Residency status",
            options=["Resident", "Non-resident", "Unsure"],
            index=0,
        )
    with col_b:
        income_sources = st.text_area(
            "Income sources",
            value="Salary income, small freelance side income, bank interest.",
            height=90,
        )
        deductions = st.text_area(
            "Deductions / credits",
            value="Commute costs, work equipment, training expenses.",
            height=90,
        )

    special_events = st.text_area(
        "Special events",
        value="Moved cities mid-year; worked remotely for some months.",
        height=80,
    )
    goal = st.text_area(
        "Goal",
        value="Need a filing prep checklist and what to clarify.",
        height=70,
        help="Keep this short. Long goals often get copied by smaller models.",
    )

    intake = TaxIntake(
        country=country.strip(),
        year=year.strip(),
        employment=employment,
        residency_status=residency,
        income_sources=income_sources.strip(),
        deductions=deductions.strip(),
        special_events=special_events.strip(),
        goal=goal.strip(),
    )
    intake_text = _format_intake(intake)

    with st.expander("Formatted intake (what the model sees)"):
        st.code(intake_text, language="text")

    st.markdown("---")
    st.markdown("### 2) Run Orchestration")
    st.write(
        "**Orchestration** means we run multiple smaller steps and store artifacts:\n"
        "1) Classification → 2) Questions → 3) Checklist+Docs → 4) Email\n"
        "Each step’s output is shown so you can debug and improve."
    )

    run_btn = st.button("Run", type="primary", use_container_width=True)
    if not run_btn:
        return

    tabs = st.tabs(
        [
            "Classification",
            "Clarifying Questions",
            "Checklist + Docs",
            "Email Draft",
            "Summary",
        ]
    )

    # STEP 1
    with tabs[0]:
        st.markdown("### Step 1 — Classification")
        prompt = _prompt_classify_kv(intake_text)
        with st.expander("Prompt sent to the model"):
            st.code(prompt, language="text")

        t0 = time.perf_counter()
        classification, note, raw = _run_step_kv(
            pipe,
            step_name="Classification",
            prompt=prompt,
            required_keys=["category", "risk", "rationale"],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
        t1 = time.perf_counter()

        if note:
            st.warning(note)

        _kv_table("Parsed output", classification, ["category", "risk", "rationale"])
        with st.expander("Raw model output"):
            st.code(raw, language="text")
        st.caption(f"Runtime: {t1 - t0:.2f}s")

    # STEP 2
    with tabs[1]:
        st.markdown("### Step 2 — Clarifying Questions")
        prompt = _prompt_questions_kv(intake_text, classification)
        with st.expander("Prompt sent to the model"):
            st.code(prompt, language="text")

        t0 = time.perf_counter()
        questions, note, raw = _run_step_kv(
            pipe,
            step_name="Clarifying Questions",
            prompt=prompt,
            required_keys=["q1", "q2", "q3", "q4", "q5"],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
        t1 = time.perf_counter()

        if note:
            st.warning(note)

        _kv_table("Parsed output", questions, ["q1", "q2", "q3", "q4", "q5"])
        with st.expander("Raw model output"):
            st.code(raw, language="text")
        st.caption(f"Runtime: {t1 - t0:.2f}s")

    # STEP 3
    with tabs[2]:
        st.markdown("### Step 3 — Checklist + Docs")
        prompt = _prompt_checklist_kv(intake_text, classification, questions)
        with st.expander("Prompt sent to the model"):
            st.code(prompt, language="text")

        t0 = time.perf_counter()
        checklist, note, raw = _run_step_kv(
            pipe,
            step_name="Checklist",
            prompt=prompt,
            required_keys=[
                "step1",
                "step2",
                "step3",
                "doc1",
                "doc2",
                "doc3",
                "next_action",
            ],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
        t1 = time.perf_counter()

        if note:
            st.warning(note)

        _kv_table("Checklist steps", checklist, ["step1", "step2", "step3"])
        _kv_table("Documents to collect", checklist, ["doc1", "doc2", "doc3"])
        _kv_table("Next action", checklist, ["next_action"])
        with st.expander("Raw model output"):
            st.code(raw, language="text")
        st.caption(f"Runtime: {t1 - t0:.2f}s")

    # STEP 4
    with tabs[3]:
        st.markdown("### Step 4 — Email Draft")
        prompt = _prompt_email(intake_text, questions, checklist)
        with st.expander("Prompt sent to the model"):
            st.code(prompt, language="text")

        t0 = time.perf_counter()
        email = _hf_invoke(
            pipe, prompt, max_new_tokens=max_new_tokens, num_beams=num_beams
        )
        t1 = time.perf_counter()

        if not email.strip():
            st.warning(
                "Email came back empty. Use flan-t5-base and increase max_new_tokens."
            )
        st.text_area("Email draft", value=email.strip(), height=280)
        st.caption(f"Runtime: {t1 - t0:.2f}s")

    # SUMMARY
    with tabs[4]:
        st.markdown("## ✅ Pipeline Summary")
        st.write("This shows exactly what each step produced.")
        st.markdown("### Classification")
        st.json(classification)
        st.markdown("### Clarifying Questions")
        st.json(questions)
        st.markdown("### Checklist + Docs")
        st.json(checklist)

        st.markdown("---")
        st.markdown("### If you still see weak outputs")
        st.write(
            "- Use **google/flan-t5-base** (small models often copy the Goal).\n"
            "- Keep the intake short.\n"
            "- Increase **max_new_tokens** to 384–512.\n"
            "- Beam search (**beams=4**) helps structured outputs."
        )
