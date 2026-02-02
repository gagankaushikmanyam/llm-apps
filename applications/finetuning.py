"""
Hugging Face Fine-tuning Demo (CPU-friendly)

Task: Email subject line generator for logistics/operations.

This Streamlit app:
- Loads a tiny in-repo toy dataset
- Runs inference on a base model ("before fine-tuning")
- Fine-tunes the model using Trainer API
- Saves the fine-tuned model to ./artifacts/finetuning/<timestamp>/
- Runs inference again ("after fine-tuning")
- Plots training loss per epoch
- Supports reproducibility via seed control

Model choice:
- Uses "sshleifer/tiny-gpt2" for CPU friendliness.
  You can switch to "distilgpt2" later if you want a larger model.

Important:
- With tiny datasets + many epochs, models can overfit and become repetitive.
- This file includes a patched generation method to reduce repetition using:
  - greedy/beam decoding options
  - no_repeat_ngram_size
  - repetition_penalty
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st

from utils.io import (
    SubjectExample,
    finetuning_artifact_dir,
    load_toy_subject_dataset,
    save_json,
)
from utils.seed import set_seed

APP_NAME = "HF Fine-tuning: Subject Line Generator"
APP_DESCRIPTION = (
    "Fine-tune a tiny Hugging Face causal LM to generate logistics email subject lines. "
    "Shows before vs after outputs, loss curve, and saves artifacts locally."
)


# -----------------------------
# Dependency checks
# -----------------------------
def _require_transformers() -> None:
    """Fail gracefully with helpful UI if required ML libraries are missing."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import datasets  # noqa: F401
    except Exception as exc:
        st.error("Missing required ML dependencies.")
        st.write("Install requirements with:")
        st.code("python -m pip install -r requirements.txt")
        st.exception(exc)
        st.stop()


def _get_device_label() -> str:
    """Return a user-friendly device label."""
    import torch

    if torch.cuda.is_available():
        return f"GPU (cuda:{torch.cuda.current_device()})"
    return "CPU"


# -----------------------------
# Data formatting
# -----------------------------
def _format_training_text(ex: SubjectExample) -> str:
    """
    Format a supervised example into a single training string.

    For causal LM fine-tuning, we train the model to continue the text after 'Subject:'.

    Example:
      Instruction: ...
      Subject: ...
    """
    return f"Instruction: {ex.instruction}\nSubject: {ex.subject}"


def _format_inference_prompt(instruction: str) -> str:
    """Format the prompt used for generation (without the answer)."""
    return f"Instruction: {instruction}\nSubject:"


# -----------------------------
# Model / tokenizer loading
# -----------------------------
@st.cache_resource(show_spinner=False)
def _load_tokenizer_and_base_model(model_name: str):
    """
    Cache tokenizer + base model so repeated UI interactions don't reload weights.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # GPT2-like models often have no pad token; define one to avoid warnings.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model


# -----------------------------
# Dataset building / tokenization
# -----------------------------
def _build_dataset(examples: List[SubjectExample]):
    """Build a Hugging Face Dataset object from in-memory examples."""
    from datasets import Dataset

    texts = [_format_training_text(ex) for ex in examples]
    return Dataset.from_dict({"text": texts})


def _tokenize_dataset(dataset, tokenizer, max_length: int = 192):
    """
    Tokenize dataset for causal language modeling.

    Steps:
    - Tokenize text
    - Create labels = input_ids (standard for causal LM)
    """

    def _tok(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        # For causal LM: labels are the same token IDs we want to predict
        out["labels"] = out["input_ids"].copy()
        return out

    return dataset.map(_tok, batched=True, remove_columns=["text"])


# -----------------------------
# Generation (patched to reduce repetition)
# -----------------------------
def _generate_subject(
    model,
    tokenizer,
    instruction: str,
    max_new_tokens: int,
    seed: int,
    *,
    decoding_mode: str = "greedy",
) -> str:
    """
    Generate a subject line given an instruction, with repetition controls.

    Why this exists:
    - Tiny models + tiny datasets often "loop" on high-frequency tokens after fine-tuning.
      Example: "Weather Delay: Weather Delay: Weather Delay..."

    Decoding modes:
    - "greedy" (default): deterministic; great for short structured outputs.
    - "beam": explores several candidate continuations and picks the best; often cleaner.
    - "sample": more variety, but can increase repetition if model is overfit.

    Repetition controls used:
    - no_repeat_ngram_size=3: blocks repeating 3-token phrases
    - repetition_penalty=1.2: penalizes repeating tokens too often
    """
    import torch

    # Seeding matters mostly for sampling, but it's safe to do always.
    set_seed(seed)

    prompt = _format_inference_prompt(instruction)
    inputs = tokenizer(prompt, return_tensors="pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen_common = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        repetition_penalty=1.2,
    )

    with torch.no_grad():
        if decoding_mode == "beam":
            output_ids = model.generate(
                **inputs,
                **gen_common,
                do_sample=False,
                num_beams=4,
                early_stopping=True,
            )
        elif decoding_mode == "sample":
            output_ids = model.generate(
                **inputs,
                **gen_common,
                do_sample=True,
                top_p=0.9,
                temperature=0.8,
            )
        else:
            output_ids = model.generate(
                **inputs,
                **gen_common,
                do_sample=False,
            )

    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    marker = "Subject:"
    if marker in decoded:
        subject = decoded.split(marker, 1)[1].strip()
    else:
        subject = decoded.strip()

    # Keep just the first line; normalize whitespace
    subject = subject.splitlines()[0].strip()
    subject = " ".join(subject.split())

    return subject


# -----------------------------
# Training compatibility helpers
# -----------------------------
def _build_training_arguments(
    *,
    output_dir: Path,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
    use_fp16: bool,
):
    """
    Build TrainingArguments in a version-compatible way.

    Transformers versions sometimes differ in parameter names:
    - evaluation_strategy vs eval_strategy
    - logging_strategy vs logging_steps

    We also avoid overwrite_output_dir entirely (timestamped folder makes it unnecessary).
    """
    from transformers import TrainingArguments

    args_common: Dict = dict(
        output_dir=str(output_dir / "trainer_runs"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        save_strategy="no",  # we save final model manually
        report_to="none",
        seed=seed,
        fp16=use_fp16,
    )

    # Attempt 1: modern API
    try:
        return TrainingArguments(
            **args_common,
            logging_strategy="epoch",
            evaluation_strategy="epoch",
        )
    except TypeError:
        pass

    # Attempt 2: eval_strategy variant
    try:
        return TrainingArguments(
            **args_common,
            logging_strategy="epoch",
            eval_strategy="epoch",
        )
    except TypeError:
        pass

    # Attempt 3: fallback logging_steps
    try:
        return TrainingArguments(
            **args_common,
            logging_steps=10,
            evaluation_strategy="epoch",
        )
    except TypeError:
        pass

    # Attempt 4: fallback logging_steps + eval_strategy
    return TrainingArguments(
        **args_common,
        logging_steps=10,
        eval_strategy="epoch",
    )


def _make_trainer_compat(**kwargs):
    """
    Create Trainer in a version-compatible way.

    Some older transformers versions do NOT accept `tokenizer=` in Trainer.__init__.
    We try with it first, then retry without it if needed.
    """
    from transformers import Trainer

    try:
        return Trainer(**kwargs)
    except TypeError as exc:
        msg = str(exc).lower()
        if "tokenizer" in msg and "unexpected keyword argument" in msg:
            kwargs.pop("tokenizer", None)
            return Trainer(**kwargs)
        raise


def _train(
    model_name: str,
    examples: List[SubjectExample],
    *,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
    output_dir: Path,
):
    """
    Fine-tune a causal LM using Hugging Face Trainer API.

    Returns:
        trainer, tokenizer, trained_model, train_seconds
    """
    _require_transformers()

    import torch
    from transformers import DataCollatorForLanguageModeling

    set_seed(seed)

    tokenizer, model = _load_tokenizer_and_base_model(model_name)

    raw_ds = _build_dataset(examples)
    split = raw_ds.train_test_split(test_size=0.2, seed=seed)
    train_ds = _tokenize_dataset(split["train"], tokenizer)
    eval_ds = _tokenize_dataset(split["test"], tokenizer)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    use_fp16 = bool(torch.cuda.is_available())

    training_args = _build_training_arguments(
        output_dir=output_dir,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        seed=seed,
        use_fp16=use_fp16,
    )

    trainer = _make_trainer_compat(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,  # dropped automatically if unsupported
    )

    t0 = time.perf_counter()
    trainer.train()
    t1 = time.perf_counter()
    train_seconds = float(t1 - t0)

    model_out = output_dir / "model"
    model_out.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(model_out))
    tokenizer.save_pretrained(str(model_out))

    return trainer, tokenizer, trainer.model, train_seconds


def _extract_loss_per_epoch(trainer) -> List[Tuple[int, float]]:
    """
    Extract (epoch_index, loss) points from Trainer's log history.
    We'll focus on training loss.
    """
    points: List[Tuple[int, float]] = []
    history = getattr(trainer.state, "log_history", []) or []

    for item in history:
        if "loss" in item and "epoch" in item:
            epoch_idx = int(round(float(item["epoch"])))
            points.append((epoch_idx, float(item["loss"])))

    # Deduplicate by epoch (keep last)
    by_epoch: Dict[int, float] = {}
    for e, l in points:
        by_epoch[e] = l

    return sorted(by_epoch.items(), key=lambda x: x[0])


# -----------------------------
# Streamlit UI
# -----------------------------
def run() -> None:
    """Render the fine-tuning demo UI inside Streamlit."""
    _require_transformers()

    st.markdown("### Overview")
    st.write(
        "This app fine-tunes a tiny Hugging Face causal LM to generate **logistics email subject lines**. "
        "You’ll see the model’s output **before vs after** fine-tuning, training loss per epoch, and saved artifacts."
    )

    col_a, col_b, col_c = st.columns([1.1, 1.1, 1.1])
    with col_a:
        model_name = st.selectbox(
            "Base model (CPU-friendly)",
            options=["sshleifer/tiny-gpt2", "distilgpt2"],
            index=0,
            help="tiny-gpt2 is much faster on CPU. distilgpt2 is larger and slower but often higher quality.",
        )
    with col_b:
        st.text_input("Detected device", value=_get_device_label(), disabled=True)
    with col_c:
        seed = int(
            st.number_input(
                "Random seed", min_value=0, max_value=10_000_000, value=42, step=1
            )
        )

    st.markdown("---")
    st.markdown("### Inputs")

    default_instruction = "Write an email subject for a shipment delayed due to weather. Mention the new ETA is tomorrow."
    instruction = st.text_area("Instruction", value=default_instruction, height=90)

    st.markdown("### Generation controls")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        max_new_tokens = int(st.slider("Max new tokens", 8, 64, 24, 1))
    with col2:
        decoding_mode = st.selectbox(
            "Decoding mode",
            options=["greedy", "beam", "sample"],
            index=0,
            help="Greedy is most stable. Beam can be cleaner. Sample gives variety but can repeat if overfit.",
        )
    with col3:
        batch_size = int(st.selectbox("Batch size", options=[1, 2, 4, 8], index=1))

    st.markdown("### Training controls")
    col4, col5 = st.columns([1, 1])
    with col4:
        epochs = int(st.slider("Epochs", 1, 10, 3, 1))
    with col5:
        learning_rate = float(
            st.select_slider(
                "Learning rate",
                options=[1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
                value=3e-4,
            )
        )

    st.markdown("---")
    st.markdown("### Toy dataset (in-repo)")
    examples = load_toy_subject_dataset()
    st.caption(
        f"Dataset size: {len(examples)} examples (tiny on purpose for CPU-friendly fine-tuning)."
    )

    with st.expander("View dataset examples"):
        for i, ex in enumerate(examples, start=1):
            st.markdown(f"**Example {i}**")
            st.write(f"Instruction: {ex.instruction}")
            st.write(f"Subject: {ex.subject}")
            st.markdown("---")

    st.markdown("---")
    st.markdown("### Before fine-tuning")
    try:
        tokenizer, base_model = _load_tokenizer_and_base_model(model_name)
        before_subject = _generate_subject(
            model=base_model,
            tokenizer=tokenizer,
            instruction=instruction,
            max_new_tokens=max_new_tokens,
            seed=seed,
            decoding_mode=decoding_mode,
        )
        st.code(before_subject or "(empty)", language="text")
    except Exception as exc:
        st.error("Base model inference failed.")
        st.exception(exc)
        st.stop()

    st.markdown("---")
    st.markdown("### Fine-tune and compare")

    col_run, col_note = st.columns([1, 2])
    with col_run:
        run_btn = st.button(
            "🚀 Run fine-tuning", type="primary", use_container_width=True
        )
    with col_note:
        st.info(
            "Tip: With a tiny dataset, more epochs can cause overfitting/repetition. "
            "Try 1–3 epochs first. Artifacts save to `./artifacts/finetuning/<timestamp>/`."
        )

    if not run_btn:
        return

    artifacts_dir = finetuning_artifact_dir("artifacts")
    save_json(
        artifacts_dir / "run_config.json",
        {
            "model_name": model_name,
            "seed": seed,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "max_new_tokens": max_new_tokens,
            "decoding_mode": decoding_mode,
            "instruction": instruction,
        },
    )

    with st.spinner("Fine-tuning in progress..."):
        try:
            trainer, tokenizer_ft, trained_model, train_seconds = _train(
                model_name=model_name,
                examples=examples,
                epochs=epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                seed=seed,
                output_dir=artifacts_dir,
            )
        except Exception as exc:
            st.error("Fine-tuning failed.")
            st.exception(exc)
            st.stop()

    st.markdown("#### Training loss per epoch")
    loss_points = _extract_loss_per_epoch(trainer)

    if loss_points:
        import matplotlib.pyplot as plt

        epochs_x = [p[0] for p in loss_points]
        losses_y = [p[1] for p in loss_points]

        fig = plt.figure()
        plt.plot(epochs_x, losses_y, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        st.pyplot(fig, clear_figure=True)
    else:
        st.warning("No loss logs found to plot (can happen with some Trainer configs).")

    st.markdown("### After fine-tuning")
    try:
        after_subject = _generate_subject(
            model=trained_model,
            tokenizer=tokenizer_ft,
            instruction=instruction,
            max_new_tokens=max_new_tokens,
            seed=seed,
            decoding_mode=decoding_mode,
        )
        st.code(after_subject or "(empty)", language="text")
    except Exception as exc:
        st.error("Fine-tuned model inference failed.")
        st.exception(exc)
        st.stop()

    st.markdown("---")
    st.markdown("### Run summary")
    st.write(f"**Training time:** {train_seconds:.2f} seconds")
    st.write(f"**Artifacts saved to:** `{artifacts_dir.as_posix()}`")
    st.write("Saved files include:")
    st.write("- `run_config.json` (your settings)")
    st.write("- `model/` (fine-tuned model + tokenizer)")
