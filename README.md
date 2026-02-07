
⸻
Author: Gagan Kaushik Manyam  
---

# 🧪 LLM Lab — A Systems-First Playground for Modern LLM Engineering

**LLM Lab** is a modular, Streamlit-based experimentation environment for learning and demonstrating  
**how real-world LLM systems are designed, debugged, and extended** — beyond prompts and demos.

This repository focuses on **system behavior**, not model hype.

It covers:
- supervised fine-tuning
- hallucination mitigation
- RAG-lite grounding
- multi-step orchestration
- tool-based (MCP-style) execution

All examples are:
- **CPU-friendly**
- **fully inspectable**
- **explicit about failure modes**

This is a **learning + research lab**, not a production framework.

---

## 👤 Who This Repository Is For

This repo is designed for:

- **Aspiring AI / LLM Engineers** entering industry roles  
- **Software / ML Engineers** transitioning into LLM systems  
- **Researchers** who want to understand *why* LLMs fail or succeed  
- **Recruiters & hiring managers** evaluating practical system design skills  

If you want to understand:
- why hallucinations happen,
- why prompting is not enough,
- how orchestration actually works,
- how tools change LLM behavior,

this repo is for you.

---

## ✨ Design Principles

- 🔍 **Inspectability over magic**
- 🧠 **Concept-first demos** (why things work or fail)
- 💻 **CPU-first**, GPU optional
- 🎯 **Reproducibility** (explicit seeds)
- 🧩 **Plugin-style architecture**
- 🚫 No hidden datasets, no black boxes

---

## 🏗 Architecture Overview

The lab is structured around a **single Streamlit launcher**:

app.py

Applications are auto-discovered from:

applications/

### 🔌 Application Contract

Every app must expose:

```python
APP_NAME = "Human-readable name"
APP_DESCRIPTION = "Optional description"

def run() -> None:
    ...

➡️ Drop a new file into applications/
➡️ Restart Streamlit
➡️ No launcher changes required

This keeps the system scalable and clean.

⸻

🧠 App 1 — Hugging Face Fine-tuning (Supervised)

📄 File: applications/finetuning.py

Demonstrates end-to-end supervised fine-tuning of a causal language model using
Hugging Face Transformers.

⸻

🎯 Task

Logistics email subject line generation from short instructions.

Instruction: Write an email subject for a shipment delayed due to weather.
Subject: Weather Delay: Updated ETA for Shipment (Arrives Tomorrow)


⸻

📊 What This App Shows
	•	TRUE before vs after comparison
	•	Validation loss + early stopping
	•	Holdout benchmark (not seen during training)
	•	Simple metrics:
	•	Exact Match
	•	Token-level F1
	•	Saved artifacts:

artifacts/finetuning/<timestamp>/



⸻

🤖 Models
	•	sshleifer/tiny-gpt2 — ultra-fast, educational
	•	distilgpt2 — higher quality, still CPU-friendly

⚠️ This app performs full fine-tuning, not LoRA / QLoRA
(LoRA/QLoRA are planned extensions.)

⸻

🧠 Key Lesson

Fine-tuning:
	•	improves task alignment
	•	does NOT inject knowledge
	•	overfits easily with small data

This app shows what fine-tuning can and cannot do.

⸻

🧠 App 2 — Hallucinations Lab (Prompting + RAG-lite)

📄 File: applications/hallucinations.py

Demonstrates why hallucinations happen and why
grounding with context is the only reliable mitigation.

⸻

🔴 Baseline (Free-form)
	•	No structure
	•	No refusal
	•	No grounding

Ask:

What year did Isaac Newton invent the smartphone?

You’ll get:
	•	fluent
	•	confident
	•	fabricated answers

This is default LLM behavior.

⸻

⚠️ Prompting Alone Is Not Enough

JSON-only, refusal, self-consistency:
	•	improve format
	•	improve stability
	•	do NOT guarantee truth

Prompting reduces chaos — not hallucinations.

⸻

🟢 Context-Only Answering (RAG-lite)

The model:
	•	may ONLY use retrieved context
	•	must say UNKNOWN if unsupported

This is a minimal RAG system.

⸻

📚 Knowledge Base (Explicit & Local)

knowledge_base/
  australia.txt
  logistics_faq.txt

Example:

Australia's national government is based in Canberra.
Sydney is the largest city by population.

No hidden data. No magic.

⸻

🔎 Retrieval
	•	TF-IDF (scikit-learn)
	•	Chunking + similarity ranking
	•	Top-K chunks injected into prompt

Why scikit-learn?
	•	Transparent
	•	CPU-friendly
	•	No vector DB required

⸻

🧠 Key Lesson

Hallucinations are a system design problem, not a model bug.

⸻

🧠 App 3 — LangChain Orchestration (Multi-Step Reasoning)

📄 File: applications/langchain_orchestration.py

Demonstrates explicit multi-step orchestration using LangChain.

⸻

🔁 Pipeline
	1.	Classification
	2.	Clarifying questions
	3.	Checklist + required documents
	4.	Optional email draft

Each step:
	•	runs independently
	•	consumes prior output
	•	is visible in the UI

⸻

🧠 Key Lesson

Orchestration provides:
	•	control
	•	traceability
	•	debuggability

This mirrors real enterprise LLM workflows.

⸻

🧠 App 4 — MCP Tools Lab (Tool-Based Systems)

📄 File: applications/mcp_tax_tools.py

Demonstrates tool-based LLM systems inspired by
the Model Context Protocol (MCP).

⸻

🧰 Tools Implemented
	1.	classify_tax_case
	2.	build_prep_checklist
	3.	draft_tax_email

Each tool is:
	•	deterministic
	•	typed
	•	auditable

⸻

📊 UI Shows
	•	live logs
	•	progress bar
	•	each tool call (inputs + outputs)
	•	final composed result

⸻

🧠 Key Lesson

Tools turn LLMs from:

“text generators”
into
inspectable systems

⸻

🧩 How Everything Fits Together

App	What It Teaches
Fine-tuning	Weight adaptation
Hallucinations	Why grounding is required
Orchestration	Structured reasoning
MCP Tools	Controlled execution

Together, they demonstrate modern LLM system design.

⸻

▶️ Running the Lab

python -m venv llms-venv
source llms-venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py

⚠️ Always use python -m streamlit to ensure the correct environment.

⸻

🚀 Roadmap

Planned additions:
	•	LoRA / QLoRA fine-tuning
	•	Full RAG with embeddings
	•	LangGraph workflows
	•	MCP protocol integrations
	•	Multi-agent coordination
	•	ML & AI classics (trees, sparse regression, neural nets)

⸻

🧠 Final Takeaway

This repository is not about making LLMs sound smart.

It is about understanding:
	•	why they fail
	•	how systems constrain them
	•	how engineers make them reliable

That is the difference between demos and production.

⸻

⭐ If this repo helped you learn something — star it.
💬 If you’re hiring — this repo reflects how I think about AI systems.

---