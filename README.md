Author: Gagan Kaushik Manyam  
---

# 🧪 LLM Lab — A Systems-First Playground for Practical LLM Engineering

**LLM Lab** is a modular, Streamlit-based environment to **learn, test, and debug real LLM system behaviors**—from fine-tuning and hallucination mitigation to orchestration, tool usage, and full Retrieval-Augmented Generation (RAG).

This repo is **concept-first** and **system-first**:
- It shows **why** outputs fail (hallucinations, weak retrieval, overfitting),
- and **how** engineering patterns (grounding, orchestration, tools, re-ranking, strict outputs) make them reliable.

All demos are designed to be:
- **CPU-friendly by default** (GPU optional),
- **inspectable** (no hidden magic),
- and **reproducible** (seed controls, deterministic knobs).

> This is a learning + research lab, not a production framework.

---

## 👤 Who This Repository Is For

This repository is built for:
- **Aspiring AI / LLM Engineers** starting industry roles
- **Software / ML Engineers** transitioning into LLM systems
- **Researchers** who want clarity on *why systems fail/succeed*
- **Recruiters / hiring managers** evaluating practical systems skills

If you want to understand:
- why LLMs hallucinate,
- why “prompting harder” isn’t enough,
- how retrieval and citations enforce correctness,
- how orchestration and tools create reliable pipelines,

this repo is for you.

---

## ✨ Design Principles

- 🔍 **Inspectability over magic**
- 🧠 **Concept-first demos** (failure modes are part of the learning)
- 💻 **CPU-first**, GPU optional
- 🎯 **Reproducibility** (explicit seeds + deterministic modes)
- 🧩 **Plugin-style architecture** (drop-in apps)
- 🚫 No hidden datasets, no black boxes

---

## 🏗 Architecture Overview

The lab is driven by one Streamlit launcher:

- **`app.py`** — discovers and loads all apps under **`applications/`**

Every app must expose:

- `APP_NAME` (required)
- `APP_DESCRIPTION` (optional)
- `run() -> None` (required)

Add a new app by creating:

- `applications/<new_app>.py`

…and restarting Streamlit.

---

## 📌 Quick Summary of All Apps

| App | File | What it teaches | Primary “System Skill” |
|---|---|---|---|
| Fine-tuning | `applications/finetuning.py` | Adapting model weights to a task | Training + evaluation discipline |
| Hallucinations Lab | `applications/hallucinations.py` | Why hallucinations happen + how to block them | Grounding + refusal + verification |
| LangChain Orchestration | `applications/langchain_orchestration.py` | Multi-step pipelines with traceable steps | Orchestration + debuggability |
| MCP Tax Tools | `applications/mcp_tax_tools.py` | Tool-based execution with logging | Deterministic, auditable actions |
| Full RAG (Chroma) | `applications/full_rag_chroma.py` | Retrieval + re-ranking + strict citations | Retrieval quality + evidence-first answers |

---

## ▶️ Running the Lab

```bash
python -m venv llms-venv
source llms-venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py

Important: Prefer python -m streamlit to ensure Streamlit runs inside the active venv.

⸻

🧠 App 1 — Hugging Face Fine-Tuning (Supervised)

📄 File: applications/finetuning.py

What it is

Supervised fine-tuning continues training a pretrained model on a small task dataset to shift its behavior toward your domain.

Mathematically, you minimize cross-entropy over tokens:
[
\mathcal{L} = -\sum_{t} \log p_\theta(x_t \mid x_{<t})
]
You’re updating weights (\theta), not just changing the prompt.

Intended goal

Teach a model to generate logistics email subject lines from instructions.

Advantages
	•	✅ Improves task formatting and domain style (tone, structure, key terms)
	•	✅ Demonstrates training dynamics (overfitting vs generalization)
	•	✅ Shows the difference between “model behavior” vs “prompt tricks”

What this app shows
	•	TRUE Before vs After (fresh base model vs fine-tuned model)
	•	Train & validation loss curves
	•	Early stopping (prevents overfitting/repetition)
	•	Holdout benchmark (examples not seen during training)
	•	Simple metrics:
	•	Exact Match
	•	Token-level F1
	•	Artifacts saved under:
	•	artifacts/finetuning/<timestamp>/

Example

Instruction
	•	Write an email subject for a shipment delayed due to weather. Mention the new ETA is tomorrow.

Expected outcome after fine-tuning
	•	“Weather Delay: Updated ETA — Delivery Tomorrow”

Note: This app performs full fine-tuning, not LoRA/QLoRA.

⸻

🧠 App 2 — Hallucinations Lab (Prompting + Grounding)

📄 File: applications/hallucinations.py

What it is

Hallucinations happen because LLMs are probabilistic next-token predictors, not truth engines.
Without grounding, the model tries to produce a plausible continuation even when it lacks facts.

Intended goal

Show:
	1.	how hallucinations appear in baseline prompting
	2.	why formatting constraints help structure but not truth
	3.	why grounding (context-only) blocks hallucinations
	4.	how a “RAG-lite” pattern improves reliability

Advantages
	•	✅ Makes hallucination behavior visible and testable
	•	✅ Teaches refusal behavior (UNKNOWN) as a safety mechanism
	•	✅ Demonstrates grounding rules (“only answer if supported”)

Modes included (what to learn)
	•	Baseline (free-form): fluent + confident + wrong is common
	•	JSON-only: better structure, still wrong if model lacks knowledge
	•	Refusal policy: allows the model to say UNKNOWN
	•	Self-consistency: improves stability, not factuality
	•	Context-only: enforces truth by limiting allowed information

Example tests

Baseline hallucination test

Ask:
	•	“What year did Isaac Newton invent the smartphone?”

Expected:
	•	A confident fabricated answer (hallucination).

Context-only correctness test

Put into context:
	•	“Australia’s capital city is Canberra.”

Ask:
	•	“What is the capital of Australia?”

Expected:
	•	JSON answer supported by context + evidence.

⸻

🧠 App 3 — LangChain Orchestration (Multi-Step Pipeline)

📄 File: applications/langchain_orchestration.py

What it is

Orchestration decomposes a task into explicit steps. Each step has:
	•	a purpose,
	•	inputs,
	•	outputs,
	•	and can be debugged independently.

This mirrors how enterprise systems avoid “one huge prompt that does everything”.

Intended goal

Create a tax-prep assistant pipeline that breaks work into steps:
	1.	classify the case
	2.	generate clarifying questions
	3.	produce checklist + required documents
	4.	optionally draft a structured email

Advantages
	•	✅ Traceability: you can see what each step produced
	•	✅ Debuggability: identify which step caused failure
	•	✅ Control: enforce constraints per-step (JSON, refusal, etc.)

Example

Input:
	•	“I’m filing taxes in Germany; I need a checklist and what to clarify with a tax advisor.”

Expected:
	•	Classification (category)
	•	3–7 clarifying questions
	•	Checklist of documents
	•	Optional email draft to advisor

⸻

🧠 App 4 — MCP Tax Tools (Tool-Based Execution)

📄 File: applications/mcp_tax_tools.py

What it is

Tool-based systems turn LLM workflows into auditable function calls:
	•	each tool has typed inputs/outputs,
	•	deterministic logic,
	•	and logs of what happened.

This resembles the core idea behind tool protocols (MCP-style patterns).

Intended goal

Demonstrate how an assistant can call tools like:
	1.	classify_tax_case
	2.	build_prep_checklist
	3.	draft_tax_email

Advantages
	•	✅ Deterministic outputs for key steps
	•	✅ Auditable logs + intermediate states
	•	✅ Reduced hallucination by limiting “free-form invention”

Example

Input:
	•	“I need checklist + questions before filing.”

Expected:
	•	tool call logs shown in UI
	•	checklist + questions generated as structured outputs
	•	email draft assembled using tool outputs

⸻

🧠 App 5 — Full RAG (ChromaDB + HF) with Re-Rank + Strict Citations

📄 File: applications/full_rag_chroma.py

What it is

Retrieval-Augmented Generation (RAG) is a system design pattern:
	•	retrieve relevant text from a knowledge base,
	•	inject it into the prompt,
	•	and force the model to answer from evidence.

Core idea (mathematically)

We want:
[
p(y \mid x) \approx \sum_{d \in \mathcal{D}} p(y \mid x, d),p(d \mid x)
]
Where:
	•	(x) = question
	•	(d) = retrieved document chunk
	•	(p(d \mid x)) = retriever relevance score
	•	(p(y \mid x, d)) = generator conditioned on retrieved evidence

Intended goal

Build a local, inspectable full RAG pipeline:
	1.	Chunk documents
	2.	Embed chunks
	3.	Store/query in ChromaDB
	4.	Retrieve Top-N candidates
	5.	Re-rank candidates for better evidence
	6.	Generate an answer with strict JSON citations + quoted evidence

Why re-ranking is added

Vector retrieval is fast but approximate. It can return “kind of related” chunks.

Re-ranking uses a stronger model that scores:
	•	(question, chunk)

This improves retrieval quality, especially when multiple candidates look similar.

Strict answer + citations JSON (why it matters)

This app enforces a JSON contract:
	•	answer
	•	supported_by_context
	•	citations (source + chunk id)
	•	quoted_evidence (short direct quotes)

If evidence is insufficient, the model must answer:
	•	UNKNOWN

This turns RAG into an auditable system instead of “trust me bro”.

Advantages
	•	✅ Retrieval-based correctness (when KB is correct)
	•	✅ Evidence-first answers
	•	✅ Stronger retrieval quality via re-rank
	•	✅ Debuggable: you can inspect retrieved chunks and scores

Example

Knowledge base includes australia.txt:
	•	“Australia’s national government is based in Canberra…”

Ask:
	•	“What is the capital of Australia?”

Expected:
	•	JSON answer: Canberra
	•	citations show australia.txt chunk id
	•	quoted evidence contains the supporting line

⸻

🚀 Roadmap

Planned additions:
	•	LoRA / QLoRA fine-tuning
	•	Embedding-based RAG variants + vector DB comparisons
	•	LangGraph workflows
	•	MCP protocol integrations
	•	Multi-agent coordination

⸻

🧠 Portfolio Note (Recruiter-Friendly)

This repo demonstrates practical LLM systems skills:
	•	training discipline (eval, overfitting control)
	•	hallucination mitigation via grounding
	•	retrieval + re-ranking + citations
	•	orchestration (step-by-step pipelines)
	•	tool-based execution and logging

If you’re hiring for AI/LLM roles, this repo reflects how I design systems:
inspectable, auditable, and resilient.

⭐ If this repo helped you learn something — consider starring it.
💬 If you’re hiring — feel free to reach out.

