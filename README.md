
⸻

# LLM Lab 🧪

LLM Lab is a modular, Streamlit-based experimentation environment for exploring core **Large Language Model (LLM) techniques**, starting with **supervised fine-tuning** and extending toward hallucination mitigation, LoRA/QLoRA, RAG, and agent-style systems.

The repository emphasizes:
- clear, inspectable implementations
- CPU-friendly demos with optional GPU acceleration
- reproducibility (seed control)
- a plugin-style architecture for adding new apps

---

## Architecture Overview

The project is structured around a single Streamlit launcher (`app.py`) that dynamically discovers and loads applications from the `applications/` directory.

Each application must expose:

```python
APP_NAME = "Human-readable name"
APP_DESCRIPTION = "Optional description"

def run() -> None:
    ...

New experiments are added by dropping a new file into applications/—no core launcher modifications are required.

⸻

App: Hugging Face Fine-tuning Demo

File: applications/finetuning.py

This app demonstrates end-to-end supervised fine-tuning of a causal language model using the Hugging Face Transformers ecosystem, with a direct comparison between pretrained and fine-tuned behavior.

Task

Logistics email subject line generation from short instructions.

Data

A small in-repo toy dataset defined in utils/io.py (instruction → subject).

Model
	•	Default: sshleifer/tiny-gpt2 (CPU-friendly)
	•	Optional: distilgpt2 (higher quality, slower on CPU)

Training + Evaluation
	•	Examples are formatted as:
Instruction: ...\nSubject: ...
	•	Tokenization and training use the standard causal LM objective (labels = input_ids)
	•	Training runs via Hugging Face Trainer
	•	Loss per epoch is displayed
	•	Fine-tuned artifacts are saved under:
artifacts/finetuning/<timestamp>/

Expected outcome
	•	“After” output becomes more task-aligned than “Before”
	•	With very small datasets, excessive epochs can cause repetition (overfitting),
mitigated via decoding constraints (greedy/beam + repetition controls)

⸻

App: Hallucinations Lab — Prompting + RAG-lite Grounding

File: applications/hallucinations.py

This app demonstrates why hallucinations happen, why prompting alone is insufficient, and how grounding with retrieved context (RAG-lite) is the only reliable way to reduce hallucinations in practice.

The goal is not to make a small model “know facts”, but to show how systems enforce correctness even when the model is unreliable.

⸻

Why Hallucinations Happen (Baseline)

Large Language Models are probabilistic text generators, not truth engines.

When you ask a factual question without constraints, the model will:
	•	Produce a fluent answer
	•	Sound confident
	•	Hallucinate if it does not know

Baseline Mode (Free-form)

Technique:
	•	No structure
	•	No refusal
	•	No grounding

Expected behavior:
	•	The model always answers
	•	Often confidently wrong
	•	No way to verify correctness

How to test it:
	1.	Select Technique → Baseline (free-form)
	2.	Ask a nonsense or unknown question:
	•	“What year did Isaac Newton invent the smartphone?”
	3.	Observe:
	•	The model gives a confident but fabricated answer

This demonstrates the default hallucination behavior of LLMs.

⸻

Why Prompting Alone Is Not Enough

JSON-only / Refusal / Self-consistency Modes

These techniques improve output control, not truth.

They help with:
	•	Structured outputs
	•	Safer responses (UNKNOWN)
	•	Stability across multiple generations

They do not guarantee correctness unless the model already knows the answer.

How to test:
	•	Use JSON-only or JSON + refusal
	•	Ask factual questions the model may or may not know
	•	You may still get:
	•	Wrong answers
	•	Or inconsistent answers across runs

This shows:

Prompting reduces chaos, not hallucinations.

⸻

Context-Only Answering (Grounded Mode)

This is the core hallucination solution demonstrated in this app.

What “Context-Only” Actually Means
	•	The model is forbidden from using its internal knowledge
	•	It may only answer using retrieved text
	•	If the answer is not supported → it must return UNKNOWN

This is a RAG-lite system.

⸻

Knowledge Base (Local, Explicit, Transparent)

You must create a local knowledge base manually.

Folder structure:

knowledge_base/
  australia.txt
  logistics_faq.txt
  ...

Example (australia.txt):

Australia's national government is based in Canberra, home to Parliament House.
Sydney is the largest city by population.

There is no hidden dataset and no magic.

This is intentional:
	•	You control the knowledge
	•	You can inspect exactly what the model sees
	•	You can test failure cases honestly

⸻

How Retrieval Works (RAG-lite)
	1.	Documents are split into chunks
	2.	TF-IDF (scikit-learn) ranks chunks by similarity to the question
	3.	Top-K chunks are retrieved
	4.	The model is only allowed to answer using those chunks

This is why scikit-learn is installed:
	•	It powers local retrieval
	•	No embeddings, no vector DB, no cloud
	•	Simple, transparent, CPU-friendly

⸻

How to Test Context-Only Correctness

Correct Answer Case
	1.	Technique → Context-only (RAG-lite grounded)
	2.	Question:

What is the capital of Australia?


	3.	Ensure australia.txt contains the answer
	4.	Expected output:
	•	answer: Canberra
	•	supported_by_context: true
	•	Evidence quoted from the document

Forced UNKNOWN Case
	1.	Ask:

Who is the president of Australia?


	2.	If the answer is not in the documents
	3.	Expected output:
	•	answer: UNKNOWN
	•	supported_by_context: false

This verifies that hallucination is blocked, not hidden.

⸻

Why Context-Only May Feel “Obvious”

You may notice:

“We already put the answer in the context.”

That is the entire point.

In real systems:
	•	Context comes from databases
	•	Documents
	•	APIs
	•	Logs
	•	Contracts
	•	Internal knowledge bases

The model’s job is not to invent, but to:
	•	Read
	•	Extract
	•	Cite
	•	Refuse when unsupported

This app demonstrates that principle clearly.

⸻

Summary: What Each Mode Teaches You

Mode	What it demonstrates
Baseline	Confident hallucinations
JSON-only	Structured but not factual
Refusal	Safer uncertainty
Self-consistency	Stability, not truth
Context-only (RAG-lite)	Actual hallucination prevention


⸻

Key Takeaway

Hallucinations are not a “model bug”.
They are a system design problem.

This app shows that:
	•	Prompting helps formatting
	•	Retrieval provides truth
	•	Grounding enforces correctness

Once this is clear, extending to:
	•	Full RAG
	•	Vector databases
	•	Citations
	•	Tools & agents
becomes straightforward.

⸻

Running the Lab

python -m venv llms-venv
source llms-venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py

Always use python -m streamlit to ensure Streamlit runs inside the correct environment.

⸻

Extending the Lab

To add a new application:
	1.	Create applications/<new_app>.py
	2.	Define APP_NAME and run()
	3.	Restart Streamlit

Suggested next apps:
	•	applications/lora.py
	•	applications/qlora.py
	•	applications/rag.py
	•	applications/mcp.py