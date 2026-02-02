⸻


# LLM Lab 🧪

A lightweight, scalable **Streamlit-based playground** for experimenting with Large Language Model (LLM) techniques such as **fine-tuning**, with a structure that allows adding more apps (LoRA, QLoRA, RAG, MCP) later.

---

## Streamlit App Overview

This repository runs a **single Streamlit launcher (`app.py`)** that automatically discovers and loads mini-apps from the `applications/` folder.

### How it works
- `app.py` scans `applications/*.py`
- Each app must define:
  ```python
  APP_NAME = "Readable App Name"

  def run() -> None:
      ...

	•	Every discovered app appears automatically in the left sidebar
	•	Selecting an app renders its UI in the main panel

This design allows you to add new LLM experiments by simply adding a new file to applications/ — no changes to the launcher are required.

⸻

App Included: Hugging Face Fine-tuning Demo

File: applications/finetuning.py
Goal: Demonstrate before vs after behavior when fine-tuning a language model on a small, task-specific dataset.

The app shows:
	•	Model output before fine-tuning
	•	Model output after fine-tuning
	•	Training loss per epoch
	•	Saved fine-tuned model artifacts

⸻

What is Fine-tuning?

Pretrained language models (like GPT-2) are trained on large, general datasets.
They understand language broadly, but they are not specialized for your exact task.

Fine-tuning means:

Continuing training on a small, task-specific dataset so the model adapts its behavior.

Fine-tuning changes the model’s weights, unlike prompt engineering, which only changes the input text.

⸻

How Fine-tuning is done in this repo (Toy Example)

Task

Generate logistics email subject lines from short instructions.

Example input

Write an email subject for a shipment delayed due to weather.
Mention the new ETA is tomorrow.

Target output

Weather Delay: Updated ETA for Shipment (Arrives Tomorrow)


⸻

Dataset
	•	The dataset is defined in utils/io.py
	•	Each example contains:
	•	instruction – what the email is about
	•	subject – the correct subject line
	•	The dataset is intentionally very small so training runs quickly on CPU

This is a learning/demo dataset, not a production one.

⸻

Model
	•	Default model: sshleifer/tiny-gpt2
	•	Very small
	•	CPU-friendly
	•	Chosen to make fine-tuning fast and visible
	•	Optional upgrade: distilgpt2
	•	Better quality
	•	Slower on CPU

Models are loaded from Hugging Face Transformers.

⸻

Training process (simplified)
	1.	Each example is formatted as:

Instruction: <instruction>
Subject: <subject>


	2.	Text is tokenized into model inputs
	3.	The model is trained using Hugging Face’s Trainer API
	4.	Training runs for a small number of epochs
	5.	The fine-tuned model is saved locally under:

artifacts/finetuning/<timestamp>/



⸻

Generation (Before vs After)

The same instruction is run:
	•	Once with the base pretrained model
	•	Once with the fine-tuned model

This makes it easy to see how fine-tuning changes model behavior.

To reduce repetition caused by tiny datasets, generation uses:
	•	Greedy or beam decoding
	•	Repetition penalties
	•	No-repeat n-gram constraints

⸻

Running the App

python -m venv llms-venv
source llms-venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py

Always use python -m streamlit to ensure the correct virtual environment is used.

⸻

Extending the Lab

To add a new experiment:
	1.	Create a new file in applications/
	2.	Define APP_NAME and run()
	3.	Restart Streamlit

The README can be extended by adding short sections for new apps as they are added.

⸻

Purpose of this Repo

This project focuses on understanding the mechanics of LLM fine-tuning and experimentation — not on producing perfect text.

Once these fundamentals are clear, extending to LoRA, QLoRA, RAG, or agent systems becomes straightforward.

---

