
⸻


# LLM Lab 🧪

LLM Lab is a modular, Streamlit-based experimentation environment for exploring and demonstrating core **Large Language Model (LLM) techniques**, starting with **supervised fine-tuning** and designed to scale toward LoRA, QLoRA, RAG, and agent-based systems.

The repository prioritizes:
- clarity of implementation
- reproducibility
- CPU-friendly experimentation
- clean extensibility through a plugin-style app architecture

---

## Architecture Overview

The project is structured around a **single Streamlit launcher** (`app.py`) that dynamically discovers and loads LLM mini-applications from the `applications/` directory.

### Application discovery
- `app.py` scans all Python files in `applications/`
- Each file represents an independent LLM experiment
- No manual registration is required

Each application must expose the following interface:

```python
APP_NAME = "Human-readable application name"

def run() -> None:
    ...

Discovered applications are rendered automatically in the Streamlit sidebar, and their UI is displayed in the main panel upon selection.

This design enables seamless extension of the lab by adding new application files without modifying core infrastructure.

⸻

Included Application

App 1: Hugging Face Fine-tuning Demo

File: applications/finetuning.py

This application demonstrates end-to-end supervised fine-tuning of a causal language model using the Hugging Face Transformers ecosystem, with a clear comparison between pretrained and fine-tuned model behavior.

⸻

Background: What is Fine-tuning?

Pretrained language models (e.g., GPT-2) are trained on large, general-purpose corpora.
They exhibit strong linguistic competence but lack specialization for specific downstream tasks.

Fine-tuning refers to continuing training on a smaller, task-specific dataset in order to adapt the model’s internal weights to a particular domain or output style.

Key characteristics:
	•	updates model parameters (weights)
	•	differs fundamentally from prompt engineering
	•	enables task specialization with limited data

⸻

Fine-tuning in This Repository (Toy Example)

Task Definition

The fine-tuning task implemented here is logistics email subject line generation.

Given a short instruction describing an operational scenario, the model is trained to generate a concise, professional email subject line.

Example input

Write an email subject for a shipment delayed due to weather.
Mention the new ETA is tomorrow.

Expected output

Weather Delay: Updated ETA for Shipment (Arrives Tomorrow)


⸻

Implementation Details

Data
	•	Dataset is defined in utils/io.py
	•	Each training example consists of:
	•	instruction: textual description of the scenario
	•	subject: target email subject line
	•	The dataset is intentionally small to ensure:
	•	fast execution on CPU
	•	clear visibility of training effects

This dataset is designed for demonstration and learning, not for production-grade performance.

⸻

Model
	•	Default model: sshleifer/tiny-gpt2
	•	extremely lightweight
	•	CPU-friendly
	•	suitable for rapid experimentation
	•	Optional alternative: distilgpt2
	•	higher capacity
	•	improved output quality
	•	slower on CPU

Models are loaded via Hugging Face Transformers.

⸻

Training Procedure
	1.	Each example is formatted as a single causal language modeling sequence:

Instruction: <instruction text>
Subject: <subject text>

	2.	Text is tokenized and converted into model inputs
	3.	Labels are set equal to input IDs (standard causal LM objective)
	4.	Training is performed using Hugging Face’s Trainer API
	5.	Training runs for a small number of epochs to avoid overfitting
	6.	The fine-tuned model and tokenizer are saved locally under:

artifacts/finetuning/<timestamp>/


⸻

Evaluation and Comparison

The application performs side-by-side inference using the same instruction:
	•	once with the base pretrained model
	•	once with the fine-tuned model

This direct comparison highlights how fine-tuning alters model behavior for the target task.

Training loss per epoch is plotted to provide visibility into optimization dynamics.

⸻

Generation Strategy

Because the dataset is intentionally small, the generation pipeline applies stabilizing constraints to reduce repetition and overfitting artifacts:
	•	greedy or beam decoding (configurable)
	•	repetition penalty
	•	no-repeat n-gram constraints

These choices prioritize interpretability and consistency over creative diversity.

⸻

Expected Outcome

After fine-tuning:
	•	outputs become more structured and task-aligned
	•	subject lines exhibit clearer logistics-oriented phrasing
	•	differences between pretrained and fine-tuned behavior are immediately observable

It is expected—and instructive—that excessive epochs on small datasets can lead to repetition, illustrating common fine-tuning failure modes.

⸻

Running the Application

python -m venv llms-venv
source llms-venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py

Always use python -m streamlit to ensure the correct virtual environment is used.

⸻

Extensibility

The repository is designed to grow incrementally.

New experiments (e.g., LoRA, QLoRA, RAG, MCP tools) can be added by:
	1.	creating a new file in applications/
	2.	defining APP_NAME and run()
	3.	restarting the Streamlit app

No changes to the launcher are required.

⸻

Scope and Intent

LLM Lab is focused on mechanistic understanding and experimentation, not on maximizing text quality.

The goal is to provide a clean, inspectable foundation for:
	•	understanding how fine-tuning works in practice
	•	observing common training behaviors and failure modes
	•	extending toward more advanced LLM adaptation techniques

This foundation makes subsequent work on parameter-efficient fine-tuning, retrieval augmentation, and agent systems significantly easier to reason about.

---
