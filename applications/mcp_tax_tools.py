# FILE: applications/mcp_tax_tools.py
"""
MCP Tools Lab (FastMCP + Hugging Face) — Tax Workflow (Germany example)

What this app demonstrates
--------------------------
A "tools-first" workflow that behaves like real MCP/agent systems:

- Tools are deterministic and return typed/structured outputs (auditable).
- LLM is OPTIONAL and used only to "polish" natural language (email).
- Streamlit UI clearly shows:
  - logging
  - progress
  - intermediate tool calls (inputs/outputs + timing)
  - stable final results

Why your previous version produced weird output
-----------------------------------------------
1) Streamlit reruns the script, so local variables reset after the button click.
   Fix: persist results in st.session_state.

2) Asking a small model to follow strict "key=value" formatting is fragile.
   Fix: tools create the structure deterministically; LLM can polish wording.

Tools (3)
---------
1) classify_tax_case: rule-based classification + risk
2) build_prep_checklist: deterministic checklist + documents by category
3) draft_tax_email: template email using outputs (LLM polish optional)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import streamlit as st

APP_NAME = "MCP: Model Context Protocol"
APP_DESCRIPTION = (
    "3-tool FastMCP-style workflow (Taxes) with clear logs, progress, tool calls, "
    "and deterministic structured outputs (LLM used only for optional polishing)."
)


# -----------------------------
# Dependency checks
# -----------------------------
def _require_deps() -> None:
    """Fail gracefully with helpful UI if dependencies are missing."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception as exc:
        st.error("Missing ML dependencies (torch / transformers).")
        st.write("Install requirements with:")
        st.code("python -m pip install -r requirements.txt")
        st.exception(exc)
        st.stop()


def _try_init_fastmcp() -> Tuple[bool, Optional[Any], str]:
    """
    Try to initialize FastMCP if installed.
    Optional: the UI still works without it.
    """
    try:
        from mcp.server.fastmcp import FastMCP  # type: ignore

        mcp = FastMCP("tax-tools")
        return True, mcp, "FastMCP detected and initialized."
    except Exception as exc:
        return (
            False,
            None,
            f"FastMCP not available (optional). Running local tool registry only. Details: {exc}",
        )


def _device_label() -> str:
    import torch

    return "GPU (cuda)" if torch.cuda.is_available() else "CPU"


# -----------------------------
# Streamlit session state helpers (CRITICAL)
# -----------------------------
def _ss_init() -> None:
    st.session_state.setdefault("mcp_logs", [])
    st.session_state.setdefault("mcp_calls", [])
    st.session_state.setdefault("last_classification", {})
    st.session_state.setdefault("last_checklist", {})
    st.session_state.setdefault("last_email", {})
    st.session_state.setdefault("last_intake", "")


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    st.session_state["mcp_logs"].append(f"[{ts}] {msg}")


# -----------------------------
# HF pipeline (optional polish)
# -----------------------------
@st.cache_resource(show_spinner=False)
def _build_hf_pipe(model_name: str):
    """
    Use a deterministic seq2seq generator.
    Avoid device_map="auto" for beginner stability; explicitly select device.
    """
    from transformers import pipeline
    import torch

    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        task="text2text-generation",
        model=model_name,
        device=device,
    )


def _hf_invoke(pipe, prompt: str, *, max_new_tokens: int, num_beams: int) -> str:
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

    return (text or "").strip()


# -----------------------------
# Tool call tracing (UI)
# -----------------------------
@dataclass
class ToolCall:
    tool_name: str
    input: Dict[str, Any]
    output: Dict[str, Any]
    seconds: float


def _trace_tool_call(
    name: str, inp: Dict[str, Any], out: Dict[str, Any], seconds: float
) -> None:
    st.session_state["mcp_calls"].append(
        ToolCall(tool_name=name, input=inp, output=out, seconds=seconds)
    )


# -----------------------------
# Local tool registry (always works)
# -----------------------------
ToolFn = Callable[..., Dict[str, Any]]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, ToolFn] = {}

    def register(self, name: str, fn: ToolFn) -> None:
        self._tools[name] = fn

    def call(self, name: str, **kwargs: Any) -> Dict[str, Any]:
        if name not in self._tools:
            raise KeyError(f"Tool not found: {name}")
        return self._tools[name](**kwargs)


# -----------------------------
# Tool 1: deterministic classification (NO LLM)
# -----------------------------
CATEGORIES = [
    "employment_only",
    "mixed_income",
    "self_employed",
    "investments_interest",
    "relocation_remote_work",
    "deductions_focus",
    "other",
]


def _tool_classify_tax_case(*, intake: str) -> Dict[str, Any]:
    """
    Deterministic rule-based classification.
    This is what "tools" should look like: fast, stable, auditable.
    """
    t = intake.lower()

    has_salary = "salary" in t or "full-time" in t or "employer" in t
    has_freelance = "freelance" in t or "self-employed" in t or "invoice" in t
    has_interest = "interest" in t or "dividend" in t or "capital" in t or "invest" in t
    has_move = "moved" in t or "relocat" in t or "new city" in t
    has_remote = "remote" in t or "home office" in t
    has_deductions = (
        "deduction" in t or "commute" in t or "training" in t or "equipment" in t
    )

    # Category
    if has_salary and not has_freelance and not has_interest:
        category = "employment_only"
    elif has_salary and has_freelance:
        category = "mixed_income"
    elif has_freelance and not has_salary:
        category = "self_employed"
    elif has_interest and not has_freelance and has_salary:
        category = "investments_interest"
    elif has_move or has_remote:
        category = "relocation_remote_work"
    elif has_deductions:
        category = "deductions_focus"
    else:
        category = "other"

    # Risk heuristic
    risk = "low"
    if category in {"mixed_income", "self_employed"}:
        risk = "medium"
    if "foreign" in t or "crypto" in t or "rental" in t or "real estate" in t:
        risk = "high"

    rationale = {
        "employment_only": "Primarily wage income; standard filing with typical documents.",
        "mixed_income": "Salary plus freelance income needs separation of income/expenses (EUR).",
        "self_employed": "Self-employed income requires careful bookkeeping and potentially VAT considerations.",
        "investments_interest": "Investment income/interest statements required; check withholding/tax certificates.",
        "relocation_remote_work": "Move/remote work introduces allocation questions and additional documentation.",
        "deductions_focus": "Focus is deductions; strong receipt/document discipline improves outcomes.",
        "other": "Scenario unclear or mixed; requires clarifying questions to categorize properly.",
    }[category]

    return {"category": category, "risk": risk, "rationale": rationale}


# -----------------------------
# Tool 2: deterministic checklist/documents (NO LLM)
# -----------------------------
CHECKLIST_BY_CATEGORY: Dict[str, Dict[str, Any]] = {
    "employment_only": {
        "steps": [
            "Collect Lohnsteuerbescheinigung (wage tax certificate) from employer(s).",
            "Compile deduction receipts: commute, work equipment, training, home office (if applicable).",
            "Confirm health insurance, pension, and other payroll-related statements (if needed).",
        ],
        "documents": [
            "Lohnsteuerbescheinigung (2025)",
            "Commute records (tickets, distance, office days)",
            "Receipts for work equipment (laptop, chair, etc.)",
            "Training invoices/certificates",
        ],
        "next_action": "Verify which deductions are eligible and whether you need Anlage N only.",
    },
    "mixed_income": {
        "steps": [
            "Separate employment vs freelance income clearly (timeline + amounts).",
            "Prepare simple income/expense summary for freelance work (EUR).",
            "Collect deduction receipts (commute, equipment, training, home office).",
        ],
        "documents": [
            "Lohnsteuerbescheinigung (2025)",
            "Freelance invoices issued",
            "Freelance expense receipts (software, equipment, travel, etc.)",
            "Bank statements (to reconcile payments)",
            "Training invoices/certificates",
        ],
        "next_action": "Clarify whether freelance qualifies for Kleinunternehmer (VAT) and which forms apply.",
    },
    "investments_interest": {
        "steps": [
            "Collect annual tax certificates from banks/brokers (Jahressteuerbescheinigung).",
            "Check if any capital gains require additional reporting (especially foreign brokers).",
            "Confirm Freistellungsauftrag/withholding and whether to reclaim overwithheld taxes.",
        ],
        "documents": [
            "Jahressteuerbescheinigung (bank/broker)",
            "Broker statements (gains/losses)",
            "Foreign account statements (if any)",
        ],
        "next_action": "Confirm whether any investment income was foreign-sourced and needs special reporting.",
    },
    "relocation_remote_work": {
        "steps": [
            "Document move date(s) and addresses; collect registration confirmations if available.",
            "Track remote work vs office days (important for deductions).",
            "Gather receipts for moving-related costs if deductible in your situation.",
        ],
        "documents": [
            "Move confirmation / registration (Anmeldung) if available",
            "Lease contracts (old/new) or proof of residence change",
            "Remote work log (months/days)",
            "Commute records (tickets, distance)",
        ],
        "next_action": "Clarify deduction rules for home office and whether move expenses are deductible for you.",
    },
    "deductions_focus": {
        "steps": [
            "Organize all receipts by category (commute, equipment, training, home office).",
            "Write short notes for each expense: purpose + date + amount.",
            "Summarize totals per category to speed filing or advisor review.",
        ],
        "documents": [
            "Receipts for equipment/training",
            "Commute documentation",
            "Home office documentation (if applicable)",
        ],
        "next_action": "Confirm which deduction categories apply to your employment status and tax year.",
    },
    "self_employed": {
        "steps": [
            "Prepare complete income/expense summary (EUR) with supporting documentation.",
            "Check VAT status (Kleinunternehmer vs VAT filings).",
            "Collect all invoices, contracts, and proof of payment.",
        ],
        "documents": [
            "Issued invoices",
            "Expense receipts",
            "Bank statements",
            "Contracts with clients",
        ],
        "next_action": "Clarify VAT obligations and which annexes/forms apply for self-employment.",
    },
    "other": {
        "steps": [
            "List all income sources and amounts (salary, freelance, capital, rental, etc.).",
            "Identify any special events (move, foreign income, marriage, etc.).",
            "Collect all available documents first; then clarify missing items with an advisor.",
        ],
        "documents": [
            "Any wage tax certificate(s) if employed",
            "Bank/broker annual statements if applicable",
            "Freelance invoices/receipts if applicable",
        ],
        "next_action": "Answer the clarifying questions below to categorize the case correctly.",
    },
}


def _tool_build_prep_checklist(
    *, intake: str, classification: Dict[str, Any]
) -> Dict[str, Any]:
    cat = str(classification.get("category", "other"))
    if cat not in CHECKLIST_BY_CATEGORY:
        cat = "other"

    base = CHECKLIST_BY_CATEGORY[cat]
    steps = list(base["steps"])
    docs = list(base["documents"])
    next_action = str(base["next_action"])

    # Add scenario-specific items (cheap heuristics)
    t = intake.lower()
    if "training" in t and "training invoices" not in " ".join(docs).lower():
        docs.append("Training invoices/certificates (Fortbildungskosten)")
    if "equipment" in t and "equipment" not in " ".join(docs).lower():
        docs.append("Receipts for work equipment (Arbeitsmittel)")
    if "commute" in t and "commute" not in " ".join(docs).lower():
        docs.append("Commute documentation (Entfernungspauschale evidence)")
    if "interest" in t and "jahressteuerbescheinigung" not in " ".join(docs).lower():
        docs.append("Bank interest statement / Jahressteuerbescheinigung")

    return {
        "steps": steps,
        "documents": docs,
        "next_action": next_action,
    }


# -----------------------------
# Tool 3: email drafting (template-first; optional LLM polish)
# -----------------------------
def _tool_draft_tax_email(
    *,
    pipe,
    intake: str,
    checklist: Dict[str, Any],
    max_new_tokens: int,
    num_beams: int,
    polish_with_llm: bool,
) -> Dict[str, Any]:
    steps: List[str] = checklist.get("steps", []) or []
    docs: List[str] = checklist.get("documents", []) or []
    next_action: str = checklist.get("next_action", "") or ""

    clarifying_questions = [
        "Do I need to file an EUR or any additional annexes (e.g., Anlage S/G/KAP) given my situation?",
        "Which expenses in my list are most relevant/eligible for deductions for my employment status and tax year?",
        "Are there any special rules I should consider due to relocation/remote work during the year?",
    ]

    subject = "Request for tax filing checklist and clarification (Germany, 2025)"
    email_template = f"""Subject: {subject}

Hello [Gagan's Imaginary Tax Advisor],

I’m preparing my 2025 tax filing in Germany and would like to confirm what I should gather and clarify before filing.

Summary:
- {intake.replace(chr(10), chr(10)+'- ')}

Documents I can provide:
{chr(10).join([f"- {d}" for d in docs]) if docs else "- (none yet)"}

What I plan to do next:
{chr(10).join([f"- {s}" for s in steps]) if steps else "- (none yet)"}

3 clarifying questions:
{chr(10).join([f"{i+1}) {q}" for i, q in enumerate(clarifying_questions)])}

Next step request:
Could you confirm if anything is missing and advise which forms/annexes apply? If possible, I’d appreciate a short call or a written checklist.

Best regards,
Gagan Kaushik Manyam
"""

    if not polish_with_llm:
        return {"email": email_template, "raw": email_template}

    # Optional LLM polish (safe because structure already exists)
    prompt = f"""
Polish this email for clarity and professionalism.
Keep the same structure and sections.
Do not remove bullet lists.
Do not add new facts.
Return ONLY the improved email text.

EMAIL:
{email_template}
""".strip()

    refined = _hf_invoke(
        pipe, prompt, max_new_tokens=max_new_tokens, num_beams=num_beams
    ).strip()

    def _looks_like_full_email(s: str) -> bool:
        s_low = s.lower()
        required_markers = [
            "subject:",
            "summary:",
            "documents i can provide",
            "clarifying questions",
            "next step",
            "best regards",
        ]
        if len(s) < 250:  # too short to be a real email
            return False
        hits = sum(1 for m in required_markers if m in s_low)
        return hits >= 3  # allow some variation, but must include several sections

    if not refined or not _looks_like_full_email(refined):
        # Model collapsed (e.g., returned only "2025 tax filing checklist")
        # Use safe deterministic template so the system stays reliable.
        return {
            "email": email_template,
            "raw": refined,
            "note": "LLM polish collapsed; fell back to deterministic template.",
        }

    return {"email": refined, "raw": refined}


# -----------------------------
# Main Streamlit UI
# -----------------------------
def run() -> None:
    _require_deps()
    _ss_init()

    st.set_page_config(page_title="MCP Tools Lab", page_icon="🧰", layout="wide")
    st.markdown("## 🧰 MCP Tools Lab — FastMCP + Hugging Face (Taxes)")
    st.caption(
        "Goal: show tools, logging, progress, and intermediate tool calls clearly."
    )
    st.info(
        "Tools run locally (in-process). FastMCP is optional. "
        "Outputs are deterministic + auditable; LLM is used only to optionally polish the email."
    )

    fastmcp_ok, mcp_obj, mcp_note = _try_init_fastmcp()
    st.caption(mcp_note)

    # Sidebar controls
    st.sidebar.header("Runtime Controls")
    model_name = st.sidebar.selectbox(
        "HF model (email polish only)",
        options=["google/flan-t5-base", "google/flan-t5-small"],
        index=0,
        help="Used only to polish email wording. Tools 1 & 2 are deterministic.",
    )
    max_new_tokens = int(st.sidebar.slider("Max new tokens (polish)", 64, 512, 256, 32))
    num_beams = int(st.sidebar.slider("Beams (polish)", 1, 8, 4, 1))
    polish_with_llm = st.sidebar.checkbox(
        "Polish email with LLM",
        value=True,
        help="If off, email is pure template (fastest + most stable).",
    )
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Device: {_device_label()}")

    pipe = _build_hf_pipe(model_name) if polish_with_llm else None

    # Intake UI
    st.markdown("### 1) Scenario Intake")
    col1, col2 = st.columns([1, 1])
    with col1:
        country = st.text_input("Country", value="Germany")
        tax_year = st.text_input("Tax year", value="2025")
        employment = st.selectbox(
            "Employment type",
            ["Full-time employee", "Student", "Self-employed", "Mixed/Other"],
            index=0,
        )
        residency = st.selectbox(
            "Residency status", ["Resident", "Non-resident", "Unsure"], index=0
        )

    with col2:
        income = st.text_area(
            "Income sources",
            value="Salary + small freelance + bank interest.",
            height=90,
        )
        deductions = st.text_area(
            "Deductions/credits", value="Commute, work equipment, training.", height=90
        )

    events = st.text_area(
        "Special events", value="Moved cities mid-year; remote work months.", height=70
    )
    goal = st.text_area(
        "Goal",
        value="Need a checklist of what to gather and what to clarify before filing.",
        height=70,
    )

    intake = (
        f"Country: {country}\n"
        f"Year: {tax_year}\n"
        f"Employment: {employment}\n"
        f"Residency: {residency}\n"
        f"Income: {income}\n"
        f"Deductions: {deductions}\n"
        f"Events: {events}\n"
        f"Goal: {goal}"
    )

    st.session_state["last_intake"] = intake

    with st.expander("Formatted intake (what tools receive)"):
        st.code(intake, language="text")

    # Tool registry
    registry = ToolRegistry()
    registry.register("classify_tax_case", lambda **kw: _tool_classify_tax_case(**kw))
    registry.register(
        "build_prep_checklist", lambda **kw: _tool_build_prep_checklist(**kw)
    )
    registry.register(
        "draft_tax_email",
        lambda **kw: (
            _tool_draft_tax_email(pipe=pipe, **kw)
            if pipe is not None
            else _tool_draft_tax_email(pipe=None, **kw)
        ),
    )

    # Optional FastMCP registration (best effort)
    if fastmcp_ok and mcp_obj is not None:
        try:

            @mcp_obj.tool()
            def classify_tax_case(intake: str) -> Dict[str, Any]:
                return _tool_classify_tax_case(intake=intake)

            @mcp_obj.tool()
            def build_prep_checklist(
                intake: str, classification: Dict[str, Any]
            ) -> Dict[str, Any]:
                return _tool_build_prep_checklist(
                    intake=intake, classification=classification
                )

            @mcp_obj.tool()
            def draft_tax_email(
                intake: str, checklist: Dict[str, Any]
            ) -> Dict[str, Any]:
                return _tool_draft_tax_email(
                    pipe=pipe,
                    intake=intake,
                    checklist=checklist,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    polish_with_llm=polish_with_llm,
                )

        except Exception:
            pass

    st.markdown("---")
    st.markdown("### 2) Run Tool Workflow (with logs + progress)")
    col_run, col_reset = st.columns([1, 1])
    with col_run:
        run_btn = st.button(
            "🚀 Run 3-tool workflow", type="primary", use_container_width=True
        )
    with col_reset:
        if st.button("Reset logs + calls + results", use_container_width=True):
            st.session_state["mcp_logs"] = []
            st.session_state["mcp_calls"] = []
            st.session_state["last_classification"] = {}
            st.session_state["last_checklist"] = {}
            st.session_state["last_email"] = {}
            st.experimental_rerun()

    progress = st.progress(0)

    if run_btn:
        _log("Starting workflow…")
        progress.progress(5)

        # Tool 1
        _log("Calling tool: classify_tax_case")
        t0 = time.perf_counter()
        inp1 = {"intake": intake}
        out1 = registry.call("classify_tax_case", **inp1)
        dt = time.perf_counter() - t0
        _trace_tool_call("classify_tax_case", inp1, out1, dt)
        st.session_state["last_classification"] = out1
        _log(f"Tool finished: classify_tax_case ({dt:.2f}s)")
        progress.progress(35)

        # Tool 2
        _log("Calling tool: build_prep_checklist")
        t0 = time.perf_counter()
        inp2 = {"intake": intake, "classification": out1}
        out2 = registry.call("build_prep_checklist", **inp2)
        dt = time.perf_counter() - t0
        _trace_tool_call("build_prep_checklist", inp2, out2, dt)
        st.session_state["last_checklist"] = out2
        _log(f"Tool finished: build_prep_checklist ({dt:.2f}s)")
        progress.progress(70)

        # Tool 3
        _log("Calling tool: draft_tax_email")
        t0 = time.perf_counter()
        inp3 = {
            "intake": intake,
            "checklist": out2,
            "max_new_tokens": max_new_tokens,
            "num_beams": num_beams,
            "polish_with_llm": polish_with_llm,
        }
        out3 = _tool_draft_tax_email(
            pipe=pipe, **inp3
        )  # call directly (pipe may be None)
        dt = time.perf_counter() - t0
        _trace_tool_call("draft_tax_email", inp3, out3, dt)
        st.session_state["last_email"] = out3
        _log(f"Tool finished: draft_tax_email ({dt:.2f}s)")
        progress.progress(100)

        _log("Workflow completed ✅")

    # Display
    st.markdown("---")
    tabs = st.tabs(["Intermediate Tool Calls", "Logs", "Results"])

    with tabs[0]:
        st.markdown("### Intermediate Tool Calls (inputs → outputs)")
        calls: List[ToolCall] = st.session_state["mcp_calls"]
        if not calls:
            st.caption("Run the workflow to see intermediate tool calls here.")
        else:
            rows = []
            for c in calls:
                rows.append(
                    {
                        "tool": c.tool_name,
                        "seconds": round(c.seconds, 3),
                        "input_keys": ", ".join(sorted(c.input.keys())),
                        "output_keys": ", ".join(sorted(c.output.keys())),
                    }
                )
            st.dataframe(rows, use_container_width=True)

            for idx, c in enumerate(calls, start=1):
                with st.expander(f"Call {idx}: {c.tool_name} ({c.seconds:.2f}s)"):
                    st.markdown("**Input**")
                    st.json(c.input)
                    st.markdown("**Output**")
                    st.json(c.output)

    with tabs[1]:
        st.markdown("### Live Logs")
        logs: List[str] = st.session_state["mcp_logs"]
        if not logs:
            st.caption("No logs yet.")
        else:
            st.code("\n".join(logs), language="text")

    with tabs[2]:
        st.markdown("### Results (persist across reruns ✅)")
        out_classification = st.session_state["last_classification"]
        out_checklist = st.session_state["last_checklist"]
        out_email = st.session_state["last_email"]

        if out_classification:
            st.markdown("#### 1) Classification")
            st.json(out_classification)

        if out_checklist:
            st.markdown("#### 2) Checklist + Documents")
            st.json(out_checklist)

        if out_email:
            st.markdown("#### 3) Email Draft")
            email = (out_email.get("email", "") or "").strip()
            st.text_area("Email", value=email, height=320)
            if not email:
                st.warning(
                    "Email is empty. Turn off polish or increase max_new_tokens and use flan-t5-base."
                )
