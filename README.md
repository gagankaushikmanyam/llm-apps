

---
# 🧪 LLM Lab

**LLM Lab** is a modular, Streamlit-based experimentation environment for exploring core  
**Large Language Model (LLM) techniques** — starting with **supervised fine-tuning** and extending toward  
**hallucination mitigation, RAG-lite grounding, LoRA/QLoRA, and agent-style systems**.

This repository is intentionally designed as a **research and learning lab**, not a production system.

---

## ✨ Design Principles

- 🔍 **Clear, inspectable implementations**
- 🧠 **Concept-first demos** (why things work or fail)
- 💻 **CPU-friendly by default**, optional GPU acceleration
- 🎯 **Reproducibility** (explicit seed control)
- 🧩 **Plugin-style architecture** for adding new experiments

---

## 🏗 Architecture Overview

The lab is structured around a **single Streamlit launcher**:

```
app.py
```

The launcher dynamically discovers and loads applications from:

```
applications/
```

### 🔌 Application Interface Contract

Each application **must expose**:

```python
APP_NAME = "Human-readable name"
APP_DESCRIPTION = "Optional description"

def run() -> None:
    ...
```

➡️ New apps are added by simply dropping a file into `applications/`  
➡️ **No core launcher changes are required**

This keeps the lab scalable, clean, and easy to extend.

---

# 🧠 App 1 — Hugging Face Fine-tuning Demo

📄 **File:** `applications/finetuning.py`

This app demonstrates **end-to-end supervised fine-tuning** of a causal language model using  
the **Hugging Face Transformers** ecosystem.

The UI directly compares **pretrained vs fine-tuned behavior**.

---

## 🎯 Task

**Logistics email subject line generation** from short natural-language instructions.

Example:

```
Instruction: Write an email subject for a shipment delayed due to weather.
Subject: Weather Delay: Updated ETA for Shipment (Arrives Tomorrow)
```

---

## 📊 Data

- Small **in-repo toy dataset**
- Defined in: `utils/io.py`
- Format:  
  **instruction → subject**
- Intentionally small to keep training **fast and inspectable**

> This dataset is for **learning and experimentation**, not production use.

---

## 🤖 Model

- **Default:** `sshleifer/tiny-gpt2`
  - Extremely small
  - CPU-friendly
- **Optional:** `distilgpt2`
  - Better quality
  - Slower on CPU

All models are loaded via **Hugging Face Transformers**.

---

## ⚙️ Training & Evaluation

- Examples are formatted as:
  ```
  Instruction: ...
  Subject: ...
  ```
- Tokenization + training use the **causal LM objective**
- Training handled via **Hugging Face Trainer**
- UI displays:
  - Loss per epoch
  - Training time
- Fine-tuned artifacts saved under:
  ```
  artifacts/finetuning/<timestamp>/
  ```

---

## ✅ Expected Outcome

- “After” outputs become **more task-aligned** than “Before”
- With very small datasets:
  - Too many epochs can cause **overfitting**
  - Mitigated using:
    - Greedy / beam decoding
    - Repetition penalties

This app demonstrates **what fine-tuning can and cannot do**.

---

# 🧠 App 2 — Hallucinations Lab (Prompting + RAG-lite)

📄 **File:** `applications/hallucinations.py`

This app demonstrates:

- Why hallucinations happen
- Why prompting alone is insufficient
- Why **grounding with retrieved context** is the only reliable mitigation strategy

The goal is **not** to make a small model “know facts”, but to show  
**how systems enforce correctness even when models are unreliable**.

---

## ❌ Why Hallucinations Happen (Baseline)

LLMs are **probabilistic text generators**, not truth engines.

Without constraints, they will:
- Produce fluent answers
- Sound confident
- Hallucinate when uncertain

---

## 🔴 Baseline Mode (Free-form)

### Technique
- No structure
- No refusal
- No grounding

### Expected Behavior
- Model **always answers**
- Often **confidently wrong**
- No way to verify correctness

### How to Test
1. Select **Technique → Baseline (free-form)**
2. Ask:
   ```
   What year did Isaac Newton invent the smartphone?
   ```
3. Observe:
   - A confident but fabricated answer

This demonstrates the **default hallucination behavior** of LLMs.

---

## ⚠️ Why Prompting Alone Is Not Enough

### JSON-only / Refusal / Self-consistency Modes

These techniques improve **output control**, not truth.

They help with:
- Structured outputs
- Safer responses (`UNKNOWN`)
- Stability across generations

They **do not guarantee correctness** unless the model already knows the answer.

> **Key insight:**  
> Prompting reduces chaos — **not hallucinations**.

---

## 🟢 Context-Only Answering (Grounded Mode)

This is the **core hallucination mitigation strategy** in the lab.

### What “Context-Only” Means

- The model is **forbidden** from using internal knowledge
- It may answer **only using retrieved text**
- If unsupported → it **must return `UNKNOWN`**

This is a **RAG-lite system**.

---

## 📚 Knowledge Base (Local & Explicit)

You must create a local knowledge base manually.

### Folder Structure
```
knowledge_base/
  australia.txt
  logistics_faq.txt
  ...
```

### Example (`australia.txt`)
```
Australia's national government is based in Canberra, home to Parliament House.
Sydney is the largest city by population.
```

There is **no hidden dataset and no magic**.

This is intentional:
- You control the facts
- You inspect exactly what the model sees
- Failure cases are explicit and honest

---

## 🔎 How Retrieval Works (RAG-lite)

1. Documents are split into chunks
2. **TF-IDF (scikit-learn)** ranks chunks by similarity to the question
3. Top-K chunks are retrieved
4. The model may **only answer using those chunks**

### Why scikit-learn?
- Lightweight local retrieval
- No embeddings
- No vector databases
- CPU-friendly and transparent

---

## 🧪 How to Test Context-Only Correctness

### ✅ Correct Answer Case
1. Technique → **Context-only (RAG-lite grounded)**
2. Question:
   ```
   What is the capital of Australia?
   ```
3. Ensure `australia.txt` contains the answer
4. Expected:
   - `answer: Canberra`
   - `supported_by_context: true`
   - Evidence quoted from document

---

### 🚫 Forced UNKNOWN Case
1. Ask:
   ```
   Who is the president of Australia?
   ```
2. If not in the documents:
   - `answer: UNKNOWN`
   - `supported_by_context: false`

This verifies hallucinations are **blocked, not hidden**.

---

## 💡 Why Context-Only Feels “Obvious”

You might think:

> “We already put the answer in the context.”

That is **exactly the point**.

In real systems, context comes from:
- Databases
- Documents
- APIs
- Logs
- Contracts
- Internal knowledge bases

The model’s job is **not to invent**, but to:
- Read
- Extract
- Cite
- Refuse when unsupported

---

## 📌 Summary — What Each Mode Teaches

| Mode | What it Demonstrates |
|----|----|
| Baseline | Confident hallucinations |
| JSON-only | Structure without truth |
| Refusal | Safer uncertainty |
| Self-consistency | Stability, not correctness |
| Context-only (RAG-lite) | **Actual hallucination prevention** |

---

## 🧠 Key Takeaway

Hallucinations are **not a model bug**.  
They are a **system design problem**.

This lab shows:
- Prompting helps formatting
- Retrieval provides truth
- Grounding enforces correctness

Once this is understood, extending to:
- Full RAG
- Vector databases
- Citations
- Tool-augmented agents  
becomes straightforward.

---

## ▶️ Running the Lab

```bash
python -m venv llms-venv
source llms-venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

⚠️ Always use `python -m streamlit` to ensure the correct environment is used.

---

## ➕ Extending the Lab

To add a new application:

1. Create `applications/<new_app>.py`
2. Define `APP_NAME` and `run()`
3. Restart Streamlit

### Suggested Next Apps
- `applications/lora.py`
- `applications/qlora.py`
- `applications/rag.py`
- `applications/mcp.py`
```

---
