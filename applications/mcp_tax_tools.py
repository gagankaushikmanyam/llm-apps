# FILE: applications/mcp_tax_tools.py
"""
MCP Tools Lab (FastMCP + Hugging Face) — Tax Workflow

What this app demonstrates
--------------------------
- A tiny "tool server" built using FastMCP (if installed)
- 3 tools (like an agent/toolbox would expose)
- A Streamlit UI that clearly shows:
  - logging
  - progress
  - intermediate tool calls (inputs/outputs + timing)

Important note
--------------
This app runs tools *locally* (in-process). You do NOT need the MCP CLI.
If you later want an external MCP server, the tool definitions here are already MCP-shaped.

Tools (3)
---------
1) classify_tax_case: classify scenario into a category + risk
2) build_prep_checklist: generate a prep checklist + documents list
3) draft_tax_email: draft an email to a tax advisor using checklist + questions

LLM usage
---------
We use an instruction-tuned HF model (FLAN-T5) because it follows instructions better than GPT-2.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import streamlit as st

APP_NAME = "MCP (Tools): Model Context Protocol"
APP_DESCRIPTION = "Demonstrates a 3-tool FastMCP-style workflow with clear logs, progress, and intermediate tool calls."


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

    The MCP Python package/API can vary by version, so:
    - We treat FastMCP as OPTIONAL.
    - We still run tools locally via a simple registry.
    """
    try:
        # Common pattern in MCP Python SDKs:
        # from mcp.server.fastmcp import FastMCP
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
# HF pipeline
# -----------------------------
@st.cache_resource(show_spinner=False)
def _build_hf_pipe(model_name: str):
    from transformers import pipeline

    # FLAN-T5 is text2text-generation; stable for instruction prompts.
    return pipeline(
        task="text2text-generation",
        model=model_name,
        device_map="auto",  # uses GPU if present
    )


def _hf_invoke(pipe, prompt: str, *, max_new_tokens: int, num_beams: int) -> str:
    """
    Deterministic invocation to reduce randomness and improve repeatability.
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


def _ss_logs() -> List[str]:
    if "mcp_logs" not in st.session_state:
        st.session_state["mcp_logs"] = []
    return st.session_state["mcp_logs"]


def _ss_calls() -> List[ToolCall]:
    if "mcp_calls" not in st.session_state:
        st.session_state["mcp_calls"] = []
    return st.session_state["mcp_calls"]


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    _ss_logs().append(f"[{ts}] {msg}")


def _trace_tool_call(
    name: str, inp: Dict[str, Any], out: Dict[str, Any], seconds: float
) -> None:
    _ss_calls().append(ToolCall(tool_name=name, input=inp, output=out, seconds=seconds))


# -----------------------------
# Local tool registry (always works)
# -----------------------------
ToolFn = Callable[..., Dict[str, Any]]


class ToolRegistry:
    """Minimal tool registry. We use this to execute tools deterministically in-process."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolFn] = {}

    def register(self, name: str, fn: ToolFn) -> None:
        self._tools[name] = fn

    def names(self) -> List[str]:
        return sorted(self._tools.keys())

    def call(self, name: str, **kwargs: Any) -> Dict[str, Any]:
        if name not in self._tools:
            raise KeyError(f"Tool not found: {name}")
        return self._tools[name](**kwargs)


# -----------------------------
# The 3 MCP-style tools (tax)
# -----------------------------
def _tool_classify_tax_case(
    *, pipe, intake: str, max_new_tokens: int, num_beams: int
) -> Dict[str, Any]:
    """
    Tool 1: classify a tax scenario into a category + risk level.
    Returns structured dict (not model JSON).
    """
    prompt = f"""
You are a tax workflow triage assistant.

Return EXACTLY 3 lines:
category=<employment_only|mixed_income|self_employed|crypto|foreign_income|real_estate|student|other>
risk=<low|medium|high>
rationale=<one short sentence>

Do NOT copy the input text.
Scenario:
{intake}

Output:
""".strip()

    raw = _hf_invoke(pipe, prompt, max_new_tokens=max_new_tokens, num_beams=num_beams)

    parsed: Dict[str, str] = {}
    for line in raw.splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        parsed[k.strip().lower()] = v.strip()

    return {
        "category": parsed.get("category", "other"),
        "risk": parsed.get("risk", "medium"),
        "rationale": parsed.get("rationale", raw.strip()[:200]),
        "raw": raw,
    }


def _tool_build_prep_checklist(
    *,
    pipe,
    intake: str,
    classification: Dict[str, Any],
    max_new_tokens: int,
    num_beams: int,
) -> Dict[str, Any]:
    """
    Tool 2: Build a prep checklist + documents list, based on the intake + classification.
    """
    cat = classification.get("category", "other")
    risk = classification.get("risk", "medium")

    prompt = f"""
You are a tax workflow assistant.

Return EXACTLY these lines:
step1=<short step>
step2=<short step>
step3=<short step>
doc1=<document name>
doc2=<document name>
doc3=<document name>
next_action=<one sentence>

Tailor to category={cat}, risk={risk}.
Do NOT copy the input.

Scenario:
{intake}

Output:
""".strip()

    raw = _hf_invoke(pipe, prompt, max_new_tokens=max_new_tokens, num_beams=num_beams)

    parsed: Dict[str, str] = {}
    for line in raw.splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        parsed[k.strip().lower()] = v.strip()

    steps = [parsed.get("step1", ""), parsed.get("step2", ""), parsed.get("step3", "")]
    docs = [parsed.get("doc1", ""), parsed.get("doc2", ""), parsed.get("doc3", "")]
    steps = [s for s in steps if s.strip()]
    docs = [d for d in docs if d.strip()]

    return {
        "steps": steps,
        "documents": docs,
        "next_action": parsed.get("next_action", ""),
        "raw": raw,
    }


def _tool_draft_tax_email(
    *,
    pipe,
    intake: str,
    checklist: Dict[str, Any],
    max_new_tokens: int,
    num_beams: int,
) -> Dict[str, Any]:
    """
    Tool 3: Draft a professional email to a tax advisor using the checklist artifacts.
    """
    steps = checklist.get("steps", [])
    docs = checklist.get("documents", [])
    next_action = checklist.get("next_action", "")

    steps_txt = "\n".join([f"- {s}" for s in steps]) if steps else "- (none)"
    docs_txt = "\n".join([f"- {d}" for d in docs]) if docs else "- (none)"

    prompt = f"""
Write a concise professional email to a tax advisor. No fluff.

Required sections (in order):
1) Subject line
2) Short summary (2-3 lines)
3) Documents I can provide (bullets)
4) 3 clarifying questions I should ask
5) Next step request

Scenario:
{intake}

Checklist steps:
{steps_txt}

Documents:
{docs_txt}

Suggested next action:
{next_action}

Email:
""".strip()

    raw = _hf_invoke(pipe, prompt, max_new_tokens=max_new_tokens, num_beams=num_beams)

    return {"email": raw, "raw": raw}


# -----------------------------
# Main Streamlit UI
# -----------------------------
def run() -> None:
    _require_deps()

    st.set_page_config(page_title="MCP Tools Lab", page_icon="🧰", layout="wide")
    st.markdown("## 🧰 MCP Tools Lab — FastMCP + Hugging Face (Taxes)")
    st.caption(
        "Goal: show tools, logging, progress, and intermediate tool calls clearly."
    )
    st.info(
        "This app executes tools locally (in-process). FastMCP is detected if installed, "
        "but you do not need the MCP CLI to run this UI."
    )

    # Init tool infra
    fastmcp_ok, mcp_obj, mcp_note = _try_init_fastmcp()
    st.caption(mcp_note)

    # Sidebar controls
    st.sidebar.header("Runtime Controls")
    model_name = st.sidebar.selectbox(
        "HF model",
        options=["google/flan-t5-base", "google/flan-t5-small"],
        index=0,
        help="Use flan-t5-base for stronger instruction following.",
    )
    max_new_tokens = int(st.sidebar.slider("Max new tokens", 128, 512, 256, 32))
    num_beams = int(st.sidebar.slider("Beams", 1, 8, 4, 1))

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Device: {_device_label()}")

    pipe = _build_hf_pipe(model_name)

    # Intake
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

    with st.expander("Formatted intake (what tools receive)"):
        st.code(intake, language="text")

    # Tool registry (always used for execution)
    registry = ToolRegistry()
    registry.register(
        "classify_tax_case", lambda **kw: _tool_classify_tax_case(pipe=pipe, **kw)
    )
    registry.register(
        "build_prep_checklist", lambda **kw: _tool_build_prep_checklist(pipe=pipe, **kw)
    )
    registry.register(
        "draft_tax_email", lambda **kw: _tool_draft_tax_email(pipe=pipe, **kw)
    )

    # If FastMCP is available, register tools there too (for future external server use)
    # We keep this best-effort + optional.
    if fastmcp_ok and mcp_obj is not None:
        try:
            # The FastMCP decorator API may vary by version; best-effort only.
            # If your MCP package supports it, these tools become server-exposable.
            @mcp_obj.tool()
            def classify_tax_case(intake: str) -> Dict[str, Any]:
                return _tool_classify_tax_case(
                    pipe=pipe,
                    intake=intake,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                )

            @mcp_obj.tool()
            def build_prep_checklist(
                intake: str, classification: Dict[str, Any]
            ) -> Dict[str, Any]:
                return _tool_build_prep_checklist(
                    pipe=pipe,
                    intake=intake,
                    classification=classification,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
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
                )

        except Exception:
            # Don't block UI if MCP registration differs in your version.
            pass

    st.markdown("---")
    st.markdown("### 2) Run Tool Workflow (with logs + progress)")
    col_run, col_reset = st.columns([1, 1])
    with col_run:
        run_btn = st.button(
            "🚀 Run 3-tool workflow", type="primary", use_container_width=True
        )
    with col_reset:
        if st.button("Reset logs + calls", use_container_width=True):
            st.session_state["mcp_logs"] = []
            st.session_state["mcp_calls"] = []
            st.experimental_rerun()

    progress = st.progress(0)

    # Output containers
    out_classification: Dict[str, Any] = {}
    out_checklist: Dict[str, Any] = {}
    out_email: Dict[str, Any] = {}

    if run_btn:
        _log("Starting workflow…")
        progress.progress(5)

        # Tool 1
        _log("Calling tool: classify_tax_case")
        t0 = time.perf_counter()
        inp1 = {
            "intake": intake,
            "max_new_tokens": max_new_tokens,
            "num_beams": num_beams,
        }
        out1 = registry.call("classify_tax_case", **inp1)
        dt = time.perf_counter() - t0
        _trace_tool_call("classify_tax_case", inp1, out1, dt)
        out_classification = out1
        _log(f"Tool finished: classify_tax_case ({dt:.2f}s)")
        progress.progress(35)

        # Tool 2
        _log("Calling tool: build_prep_checklist")
        t0 = time.perf_counter()
        inp2 = {
            "intake": intake,
            "classification": {
                "category": out1.get("category"),
                "risk": out1.get("risk"),
            },
            "max_new_tokens": max_new_tokens,
            "num_beams": num_beams,
        }
        out2 = registry.call("build_prep_checklist", **inp2)
        dt = time.perf_counter() - t0
        _trace_tool_call("build_prep_checklist", inp2, out2, dt)
        out_checklist = out2
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
        }
        out3 = registry.call("draft_tax_email", **inp3)
        dt = time.perf_counter() - t0
        _trace_tool_call("draft_tax_email", inp3, out3, dt)
        out_email = out3
        _log(f"Tool finished: draft_tax_email ({dt:.2f}s)")
        progress.progress(100)
        _log("Workflow completed ✅")

    # Display area
    st.markdown("---")
    tabs = st.tabs(["Intermediate Tool Calls", "Logs", "Results"])

    with tabs[0]:
        st.markdown("### Intermediate Tool Calls (inputs → outputs)")
        calls = _ss_calls()
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
        logs = _ss_logs()
        if not logs:
            st.caption("No logs yet.")
        else:
            st.code("\n".join(logs), language="text")

    with tabs[2]:
        st.markdown("### Results")
        if out_classification:
            st.markdown("#### 1) Classification")
            st.json(
                {
                    "category": out_classification.get("category"),
                    "risk": out_classification.get("risk"),
                    "rationale": out_classification.get("rationale"),
                }
            )
            with st.expander("Raw model output (classification)"):
                st.code(out_classification.get("raw", ""), language="text")

        if out_checklist:
            st.markdown("#### 2) Checklist + Documents")
            st.json(
                {
                    "steps": out_checklist.get("steps", []),
                    "documents": out_checklist.get("documents", []),
                    "next_action": out_checklist.get("next_action", ""),
                }
            )
            with st.expander("Raw model output (checklist)"):
                st.code(out_checklist.get("raw", ""), language="text")

        if out_email:
            st.markdown("#### 3) Email Draft")
            email = out_email.get("email", "") or ""
            st.text_area("Email", value=email.strip(), height=280)
            if not email.strip():
                st.warning(
                    "Email is empty. Increase max_new_tokens or use flan-t5-base."
                )
