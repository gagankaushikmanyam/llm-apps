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

The lab is driven by a **single Streamlit launcher**:

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

| App | File | Core Idea |
|---|---|---|
| Fine-tuning | `finetuning.py` | Weight adaptation + evaluation |
| Hallucinations Lab | `hallucinations.py` | Why hallucinations happen & how to block them |
| LangChain Orchestration | `langchain_orchestration.py` | Explicit multi-step pipelines |
| MCP Tools Lab | `mcp_tax_tools.py` | Deterministic tool-based execution |
| Full RAG | `full_rag_chroma.py` | Retrieval + re-rank + citations |
| Prompt Caching | `prompt_caching.py` | Latency optimization |

---

## 🧠 App 1 — Hugging Face Fine-Tuning (Supervised)

**File:** `applications/finetuning.py`

### What It Is
Supervised fine-tuning continues training a pretrained model on a task-specific dataset by minimizing **cross-entropy loss**.

Mathematically:
- The model updates weights θ to minimize  
  `L = − Σ log P(y | x; θ)`

### Intended Goal
Generate **logistics email subject lines** from short instructions.

### Key Lessons
- Fine-tuning improves *task alignment*
- It does **not** inject new knowledge
- Small datasets overfit quickly

---

## 🧠 App 2 — Hallucinations Lab (Prompting + Grounding)

**File:** `applications/hallucinations.py`

### What It Is
LLMs model **P(next_token | context)** — not truth.

Without grounding:
- They always answer
- They sound confident
- They hallucinate

### What This App Demonstrates
- Baseline hallucinations
- Why JSON / refusal help *format*, not *truth*
- Why **context-only answering** blocks hallucinations
- A transparent **RAG-lite** system using TF-IDF retrieval

### Key Lesson
Hallucinations are a **system design problem**, not a model bug.

---

## 🧠 App 3 — LangChain Orchestration (Multi-Step Pipelines)

**File:** `applications/langchain_orchestration.py`

### What It Is
Complex tasks are decomposed into **explicit, inspectable steps**.

### Pipeline
1. Classification  
2. Clarifying questions  
3. Checklist + required documents  
4. Structured email draft  

Each step:
- runs independently
- consumes prior output
- is visible in the UI

### Key Lesson
Orchestration gives **control, traceability, and debuggability**.

---

## 🧠 App 4 — MCP Tools Lab (Tool-Based Systems)

**File:** `applications/mcp_tax_tools.py`

### What It Is
This app demonstrates **Model Context Protocol (MCP)-style tools** executed locally.

### How MCP Tools Are Built Here

Each tool:
- has a stable name
- takes explicit inputs
- returns structured outputs
- performs one deterministic task
- has no hidden state

Example tools:
- `classify_tax_case`
- `build_prep_checklist`
- `draft_tax_email`

A local **ToolRegistry** acts as an MCP runtime:
- dispatches tools
- logs inputs/outputs
- records timing

> The tools are MCP-compliant **by contract**, even without a running MCP server.

### Key Lesson
Tools turn LLMs from text generators into **auditable systems**.

---

## 🧠 App 5 — Full RAG (ChromaDB + Re-Rank + Citations)

**File:** `applications/full_rag_chroma.py`

### What It Is
A complete Retrieval-Augmented Generation system:

1. Documents are chunked  
2. Chunks are embedded and stored in **ChromaDB**  
3. Query retrieves Top-N chunks via vector similarity  
4. A **cross-encoder re-ranker** scores (Question, Chunk) pairs  
5. The best evidence is passed to the generator  
6. Output is forced into **strict JSON with citations**

### Why ChromaDB
- Persistent vector storage
- Fast approximate nearest-neighbor search
- Decouples retrieval from generation

### Why Re-Ranking
Vector similarity is approximate.  
Re-ranking computes a stronger relevance score:

`relevance(Q, D) = CrossEncoder(Q ⊕ D)`

This improves evidence quality.

### Strict Output
Answers must include:
- answer
- supported_by_context flag
- citations
- quoted evidence

If unsupported → return `UNKNOWN`.

---

## 🧠 App 6 — Prompt Caching (Latency Optimization)

**File:** `applications/prompt_caching.py`

### What It Is
Prompt caching avoids recomputation by storing:

`hash(prompt + config) → response`

### What It Demonstrates
- Latency before caching
- Latency after caching
- Cache hits vs misses

### Key Lesson
Many LLM gains come from **systems engineering**, not larger models.

---

## ▶️ Running the Lab

Create and activate a virtual environment:

python -m venv llms-venv  
source llms-venv/bin/activate  

Install dependencies:

python -m pip install -r requirements.txt  

Run the launcher:

python -m streamlit run app.py  

Always use `python -m streamlit` to ensure the correct environment.

---

## 🤝 Contributing

Contributions are welcome.

- Keep implementations inspectable
- Prefer clarity over cleverness
- Add explanations when introducing new concepts

See `CONTRIBUTING.md` for details.

---

## 🛡 Security

This repository is for educational use.

- No secrets should be committed
- No production credentials required
- Report issues responsibly via GitHub Security Advisories

See `SECURITY.md`.

---

## 📜 Code of Conduct

All contributors are expected to follow the Code of Conduct.

See `CODE_OF_CONDUCT.md`.

---

## 🧠 Final Takeaway

This repository is not about making LLMs sound smart.

It is about understanding:
- why they fail
- how systems constrain them
- how engineers make them reliable

That is the difference between demos and production systems.

⭐ If this repo helped you learn something — consider starring it.  
💬 If you’re hiring — this repository reflects how I think about real-world AI systems.