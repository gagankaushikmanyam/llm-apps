Author: Gagan Kaushik Manyam  
---

# 🧪 LLM Lab — A Systems-First Playground for Modern LLM Engineering

**LLM Lab** is a modular, Streamlit-based experimentation environment for learning and demonstrating  
**how real-world LLM systems are designed, debugged, and extended** — beyond prompt demos and surface-level examples.

This repository focuses on **system behavior**, not model hype.

It explores:

• supervised fine-tuning  
• hallucination mitigation  
• retrieval-grounded generation (RAG-lite)  
• multi-step orchestration  
• tool-based (MCP-style) execution  
• prompt caching and latency optimization  

All examples are:

• CPU-friendly  
• fully inspectable  
• explicit about failure modes  

This is a **learning + research lab**, not a production framework.

---

## 👤 Who This Repository Is For

This repository is designed for:

• Aspiring **AI / LLM Engineers** entering industry roles  
• Software / ML engineers transitioning into **LLM systems**  
• Researchers who want to understand *why* LLMs fail or succeed  
• Recruiters and hiring managers evaluating **system-level thinking**

If you want to understand:
• why hallucinations happen  
• why prompting is not enough  
• how orchestration actually works  
• how tools and caching change LLM behavior  

this repository is for you.

---

## ✨ Design Principles

• Inspectability over magic  
• Concept-first demos (why things work or fail)  
• CPU-first, GPU optional  
• Reproducibility via explicit seeds  
• Plugin-style architecture  
• No hidden datasets, no black boxes  

---

## 🏗 Architecture Overview

The lab is built around a **single Streamlit launcher**.

Core entry point:
• app.py  

Applications are auto-discovered from:
• applications/  

### 🔌 Application Interface Contract

Each application must expose:

APP_NAME — Human-readable title  
APP_DESCRIPTION — Optional description  
run() — Streamlit entrypoint  

Adding a new app:
• Drop a file into applications/  
• Restart Streamlit  
• No launcher changes required  

This keeps the system clean, scalable, and extensible.

---

## 🧠 Application Overview

### App 1 — Hugging Face Fine-Tuning (Supervised)

File: applications/finetuning.py  

Demonstrates **end-to-end supervised fine-tuning** of a causal language model using Hugging Face Transformers.

What this app shows:
• True before-vs-after comparison  
• Validation loss with early stopping  
• Holdout benchmark not seen during training  
• Simple evaluation metrics  
  – Exact Match  
  – Token-level F1  
• Saved training artifacts  

Artifacts location:
• artifacts/finetuning/<timestamp>/  

Models:
• sshleifer/tiny-gpt2 — ultra-fast, educational  
• distilgpt2 — higher quality, still CPU-friendly  

Important note:
• This app performs **full fine-tuning**
• LoRA / QLoRA are planned extensions  

Key lesson:
Fine-tuning improves task alignment, **not knowledge**, and overfits easily on small datasets.

---

### App 2 — Hallucinations Lab (Prompting + RAG-lite)

File: applications/hallucinations.py  

Demonstrates **why hallucinations occur** and why **grounding with retrieved context** is the only reliable mitigation strategy.

Baseline behavior:
• No structure  
• No refusal  
• No grounding  

Result:
• Fluent answers  
• Confident tone  
• Fabricated facts  

Prompting techniques (JSON, refusal, self-consistency):
• Improve formatting  
• Improve stability  
• Do NOT guarantee correctness  

Key insight:
Prompting reduces chaos — **not hallucinations**.

Context-only answering (RAG-lite):
• Model may ONLY answer using retrieved context  
• Must return UNKNOWN if unsupported  

Knowledge base:
• Local text files under knowledge_base/  
• Fully explicit and inspectable  

Retrieval:
• TF-IDF via scikit-learn  
• Chunking + similarity ranking  
• Top-K context injection  

Key lesson:
Hallucinations are a **system design problem**, not a model bug.

---

### App 3 — LangChain Orchestration (Multi-Step Reasoning)

File: applications/langchain_orchestration.py  

Demonstrates **explicit multi-step orchestration** using LangChain.

Pipeline stages:
1. Classification  
2. Clarifying questions  
3. Checklist and required documents  
4. Optional email draft  

Each step:
• Executes independently  
• Consumes prior output  
• Is visible in the UI  

Key lesson:
Orchestration enables control, traceability, and debuggability — mirroring real enterprise workflows.

---

### App 4 — MCP Tools Lab (Tool-Based Systems)

File: applications/mcp_tax_tools.py  

Demonstrates **tool-driven LLM systems** inspired by the Model Context Protocol (MCP).

Tools implemented:
• classify_tax_case  
• build_prep_checklist  
• draft_tax_email  

Each tool is:
• Deterministic  
• Typed  
• Auditable  

UI shows:
• Live logs  
• Progress indicators  
• Intermediate tool calls  
• Final composed output  

Key lesson:
Tools turn LLMs from text generators into **inspectable systems**.

---

### App 5 — Prompt Caching Lab (Latency Optimization)

File: applications/prompt_caching.py  

Demonstrates **prompt caching and KV-cache reuse** for performance optimization.

What is measured:
• Latency without caching (full recomputation)  
• Latency with caching (shared prefix reused)  

UI displays:
• Before vs after latency  
• Per-query timings  
• Average speed-up  
• Output comparison  

Key lesson:
Prompt caching does not change correctness — it dramatically improves latency and scalability.

---

## 🧩 How Everything Fits Together

| Component          | What It Teaches                         |
|--------------------|------------------------------------------|
| Fine-tuning        | Weight adaptation                        |
| Hallucinations     | Why grounding is required                |
| Orchestration      | Structured reasoning                    |
| MCP Tools          | Controlled execution                    |
| Prompt Caching     | Performance and latency optimization    |

Together, these demonstrate **modern LLM system design**.

---

## ▶️ Running the Lab

Steps:
1. Create virtual environment  
2. Install dependencies  
3. Launch Streamlit  

Commands:

python -m venv llms-venv  
source llms-venv/bin/activate  
python -m pip install -r requirements.txt  
python -m streamlit run app.py  

Always run Streamlit via python -m to ensure the correct environment.

---

## 🚀 Roadmap

Planned additions:
• LoRA / QLoRA fine-tuning  
• Full RAG with embeddings  
• LangGraph workflows  
• MCP protocol integrations  
• Multi-agent coordination  
• ML & AI classics (trees, sparse regression, neural nets)  

---

## 🧠 Final Takeaway

This repository is not about making LLMs sound smart.

It is about understanding:
• why they fail  
• how systems constrain them  
• how engineers make them reliable  

That is the difference between **demos** and **production systems**.

---

⭐ If this repo helped you learn something — consider starring it.  
💬 If you’re hiring — this repo reflects how I think about AI systems.