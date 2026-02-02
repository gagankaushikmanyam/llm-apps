
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

App 1: Hugging Face Fine-tuning Demo

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

App 2: Hallucinations Lab (Prompting Techniques)

File: applications/hallucinations.py

This app demonstrates prompt-level techniques to reduce hallucinations by improving output controllability and encouraging uncertainty. These approaches do not guarantee factual correctness without grounding (retrieval, citations, tools), but they are useful building blocks.

Techniques included
	1.	Baseline (free-form): unconstrained responses can sound confident even when wrong
	2.	JSON-only format: forces structured output and improves parseability
	3.	JSON + refusal policy: permits explicit uncertainty via UNKNOWN + confidence
	4.	Context-only answering: model must answer using provided context, else UNKNOWN
	5.	Self-consistency voting: sample multiple JSON answers and pick the most frequent

Expected outcome
	•	JSON prompting tends to reduce rambling and makes outputs machine-checkable
	•	Refusal policies reduce hallucinations when the model is uncertain
	•	Context-only prompts emulate a minimal “grounded answering” rule
	•	Self-consistency improves stability when single generations are noisy

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