Author: Gagan Kaushik Manyam  
---

# 🧪 LLM Lab — A Systems-First Playground for Practical LLM Engineering

**LLM Lab** is a modular, Streamlit-based experimentation environment for learning, testing, and **debugging real Large Language Model (LLM) systems**.

This repository is **not about prompt tricks or flashy demos**.  
It is about understanding **how LLM systems actually work in practice** — where they fail, why they hallucinate, and how engineers make them reliable.

The lab covers:
- supervised fine-tuning
- hallucination mitigation
- RAG-lite and full RAG grounding
- multi-step orchestration
- tool-based (MCP-style) execution
- prompt caching and latency behavior

All examples are:
- CPU-friendly by default
- fully inspectable
- explicit about failure modes
- reproducible (seeded)

This is a **learning + research lab**, not a production framework.

---

## 👤 Who This Repository Is For

This repository is designed for:

- Aspiring **AI / LLM Engineers** entering industry roles  
- **Software / ML Engineers** transitioning into LLM systems  
- **Researchers** who want to understand *why* LLMs fail or succeed  
- **Recruiters & hiring managers** evaluating real system-design skills  

If you want to understand:
- why hallucinations happen,
- why prompting alone is insufficient,
- how retrieval enforces correctness,
- how orchestration and tools create reliability,

this repository is for you.

---

## ✨ Design Principles

- Inspectability over magic  
- Concept-first demos (failure is part of learning)  
- CPU-first, GPU optional  
- Reproducibility via explicit seeds  
- Plugin-style architecture  
- No hidden datasets, no black boxes  

---

## 🏗 Architecture Overview

The lab is driven by a single Streamlit launcher:

- `app.py`

Applications are automatically discovered from:

- `applications/`

### Application Contract

Every application must expose:

- `APP_NAME`
- `APP_DESCRIPTION` (optional)
- `run() -> None`

To add a new app:
1. Drop a file into `applications/`
2. Restart Streamlit  
3. No launcher changes required

---

## 📌 Quick Summary of Applications

| App | File | What It Demonstrates |
|---|---|---|
| Fine-tuning | `finetuning.py` | Weight adaptation + evaluation |
| Hallucinations Lab | `hallucinations.py` | Why hallucinations happen and how to block them |
| LangChain Orchestration | `langchain_orchestration.py` | Explicit multi-step pipelines |
| MCP Tools Lab | `mcp_tax_tools.py` | Tool-based deterministic execution |
| Full RAG | `full_rag_chroma.py` | Retrieval + re-rank + citations |
| Prompt Caching | `prompt_caching.py` | Latency reduction via caching |

---

## 🧠 App 1 — Hugging Face Fine-Tuning (Supervised)

**File:** `applications/finetuning.py`

### What It Is
Supervised fine-tuning continues training a pretrained model on a task-specific dataset, updating model weights by minimizing cross-entropy loss.

### Intended Goal
Generate **logistics email subject lines** from short instructions.

### What This App Shows
- True BEFORE vs AFTER comparison  
- Validation loss and early stopping  
- Holdout benchmark (not trained on)  
- Metrics: Exact Match, Token-level F1  
- Saved artifacts under `artifacts/finetuning/<timestamp>/`

### Key Lesson
Fine-tuning:
- improves task alignment
- does NOT inject new factual knowledge
- overfits easily with small datasets

This app demonstrates what fine-tuning **can and cannot** do.

---

## 🧠 App 2 — Hallucinations Lab (Prompting + Grounding)

**File:** `applications/hallucinations.py`

### What It Is
LLMs are probabilistic next-token predictors, not truth engines.  
Without grounding, they hallucinate confidently.

### What This App Demonstrates
- Baseline hallucinations (free-form prompting)
- Why JSON and refusal help structure, not truth
- Why **context-only grounding** blocks hallucinations
- A transparent **RAG-lite** system using TF-IDF retrieval

### Key Lesson
Hallucinations are a **system design problem**, not a model bug.

---

## 🧠 App 3 — LangChain Orchestration (Multi-Step Pipelines)

**File:** `applications/langchain_orchestration.py`

### What It Is
Explicit orchestration breaks a task into **visible, debuggable steps**.

### Pipeline Demonstrated
1. Classification  
2. Clarifying questions  
3. Checklist and required documents  
4. Optional structured email draft  

Each step:
- runs independently
- consumes prior outputs
- is visible in the UI

### Key Lesson
Orchestration provides control, traceability, and debuggability — essential for real systems.

---

## 🧠 App 4 — MCP Tools Lab (Tool-Based Systems)

**File:** `applications/mcp_tax_tools.py`

### What It Is
Tool-based systems move LLMs from free-text generation to **deterministic execution**.

### Tools Demonstrated
- classify_tax_case  
- build_prep_checklist  
- draft_tax_email  

### UI Shows
- live logs
- progress indicators
- each tool call with inputs and outputs

### Key Lesson
Tools turn LLMs from text generators into **auditable systems**.

---

## 🧠 App 5 — Full RAG (ChromaDB + Re-Rank + Citations)

**File:** `applications/full_rag_chroma.py`

### What It Is
A full Retrieval-Augmented Generation pipeline:
1. Chunk documents
2. Embed and store in ChromaDB
3. Retrieve Top-K candidates
4. Re-rank for higher evidence quality
5. Generate answers with **strict JSON citations and quoted evidence**

### Why Re-Ranking
Vector retrieval is approximate.  
Re-ranking improves precision by scoring question–chunk relevance more accurately.

### Strict Output Format
Answers must include:
- answer
- supported_by_context flag
- citations
- quoted evidence

If evidence is insufficient, the model must return `UNKNOWN`.

### Key Lesson
Correctness comes from **retrieval + evidence enforcement**, not model size.

---

## 🧠 App 6 — Prompt Caching (Latency & Systems Optimization)

**File:** `applications/prompt_caching.py`

### What It Is
Prompt caching stores previous prompt-response pairs to avoid repeated model execution.

### What This App Demonstrates
- Latency before caching
- Latency after caching
- Cache hits vs misses
- Deterministic outputs reused instantly

### Key Lesson
Many LLM system gains come from **systems optimization**, not better models.

---

## ▶️ Running the Lab

```bash
python -m venv llms-venv
source llms-venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py


Always use python -m streamlit to ensure the correct environment.

⸻

🚀 Roadmap

Planned additions:
	•	LoRA / QLoRA fine-tuning
	•	Embedding comparisons
	•	LangGraph workflows
	•	MCP protocol integrations
	•	Multi-agent coordination
	•	Classical ML & AI systems

⸻

🧠 Final Takeaway

This repository is not about making LLMs sound smart.

It is about understanding:
	•	why they fail
	•	how systems constrain them
	•	how engineers make them reliable

That is the difference between demos and production.

⭐ If this repo helped you learn something — star it.
💬 If you’re hiring — this repo reflects how I think about AI systems.