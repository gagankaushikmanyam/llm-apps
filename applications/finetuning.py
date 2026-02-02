"""
Hugging Face Fine-tuning Demo (CPU-friendly, WOW-oriented)

Task: Logistics email subject line generation.

What this app demonstrates clearly:
- BEFORE vs AFTER: base pretrained model vs fine-tuned model on the same instruction
- Validation loss (eval_loss) with Early Stopping to prevent overfitting/repetition
- Holdout benchmark table (examples NOT used for training)
- Simple metrics to quantify improvement (Exact Match + Token F1)
- Artifacts saved under ./artifacts/finetuning/<timestamp>/

Critical fix vs your previous version:
- DO NOT cache model objects. Trainer mutates model weights in-place.
- We only cache the tokenizer (safe) and always create fresh models for base vs training vs eval.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import streamlit as st

from utils.io import (
    SubjectExample,
    finetuning_artifact_dir,
    load_holdout_subject_benchmark,
    load_toy_subject_dataset,
    save_json,
)
from utils.seed import set_seed

APP_NAME = "HF Fine-tuning: Subject Line Generator"
APP_DESCRIPTION = (
    "Fine-tune a small Hugging Face causal LM to generate logistics email subject lines. "
    "Shows BEFORE vs AFTER, eval loss curves, holdout benchmark, metrics, and saved artifacts."
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


def _filter_kwargs(callable_obj: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter kwargs to only those supported by callable signature.
    Helps compatibility across transformers versions.
    """
    import inspect

    allowed = set(inspect.signature(callable_obj).parameters.keys())
    return {k: v for k, v in kwargs.items() if k in allowed}


# -----------------------------
# Tokenizer/model loading
# -----------------------------
@st.cache_resource(show_spinner=False)
def _load_tokenizer(model_name: str):
    """
    Cache tokenizer only (safe).
    Models must NOT be cached because training mutates weights in-place.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _load_model(model_name: str):
    """Always return a fresh model instance (never cached)."""
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(model_name)


# -----------------------------
# Formatting helpers
# -----------------------------
def _format_training_text(ex: SubjectExample) -> str:
    """Instruction + target subject in one string for causal LM training."""
    return f"Instruction: {ex.instruction}\nSubject: {ex.subject}"


def _format_inference_prompt(instruction: str) -> str:
    """Prompt with instruction, leaving subject blank for generation."""
    return f"Instruction: {instruction}\nSubject:"


# -----------------------------
# Dataset + tokenization
# -----------------------------
def _build_dataset(examples: List[SubjectExample]):
    from datasets import Dataset

    texts = [_format_training_text(ex) for ex in examples]
    return Dataset.from_dict({"text": texts})


def _tokenize_dataset(dataset, tokenizer, max_length: int = 192):
    """
    Tokenize for causal LM:
    - input_ids from tokenizer
    - labels = input_ids (standard CLM objective)
    """

    def _tok(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        out["labels"] = out["input_ids"].copy()
        return out

    return dataset.map(_tok, batched=True, remove_columns=["text"])


# -----------------------------
# Generation (stable decoding)
# -----------------------------
def _extract_subject(decoded: str) -> str:
    """
    Extract a one-line subject after 'Subject:'.

    Safe against:
    - missing marker
    - empty output after marker
    - outputs that contain only whitespace/newlines
    """
    marker = "Subject:"
    if marker in decoded:
        tail = decoded.split(marker, 1)[1]
    else:
        tail = decoded

    # Normalize whitespace
    tail = tail.strip()
    if not tail:
        return "(empty)"

    lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
    if not lines:
        return "(empty)"

    return lines[0]


def _generate_subject(
    *,
    model,
    tokenizer,
    instruction: str,
    max_new_tokens: int,
    seed: int,
    decoding: str,
) -> str:
    """
    Generate subject with repetition controls.

    Defaults should be stable and "subject-like", not creative rambling.
    - beam/greedy are more stable than sampling for tiny fine-tunes
    """
    import torch

    set_seed(seed)

    prompt = _format_inference_prompt(instruction)
    inputs = tokenizer(prompt, return_tensors="pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen_common = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
    )

    with torch.no_grad():
        if decoding == "beam":
            out = model.generate(
                **inputs,
                **gen_common,
                do_sample=False,
                num_beams=4,
                early_stopping=True,
            )
        elif decoding == "greedy":
            out = model.generate(
                **inputs,
                **gen_common,
                do_sample=False,
            )
        else:
            # sampling option (not recommended as default for tiny fine-tunes)
            out = model.generate(
                **inputs,
                **gen_common,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
            )

    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    return _extract_subject(decoded)


# -----------------------------
# Simple metrics
# -----------------------------
_word_re = re.compile(r"[A-Za-z0-9]+")


def _normalize_text(s: str) -> str:
    return " ".join(_word_re.findall(s.lower())).strip()


def _exact_match(pred: str, target: str) -> float:
    return 1.0 if _normalize_text(pred) == _normalize_text(target) else 0.0


def _token_f1(pred: str, target: str) -> float:
    """
    Token-level F1 on normalized words (simple and interpretable).
    """
    p = _normalize_text(pred).split()
    t = _normalize_text(target).split()

    if not p and not t:
        return 1.0
    if not p or not t:
        return 0.0

    # bag-of-words overlap
    pc: Dict[str, int] = {}
    tc: Dict[str, int] = {}
    for w in p:
        pc[w] = pc.get(w, 0) + 1
    for w in t:
        tc[w] = tc.get(w, 0) + 1

    common = 0
    for w, c in pc.items():
        common += min(c, tc.get(w, 0))

    precision = common / max(1, len(p))
    recall = common / max(1, len(t))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _score_examples(
    *,
    model,
    tokenizer,
    items: List[SubjectExample],
    max_new_tokens: int,
    seed: int,
    decoding: str,
) -> Dict[str, float]:
    ems: List[float] = []
    f1s: List[float] = []

    for i, ex in enumerate(items):
        pred = _generate_subject(
            model=model,
            tokenizer=tokenizer,
            instruction=ex.instruction,
            max_new_tokens=max_new_tokens,
            seed=seed + i,
            decoding=decoding,
        )
        ems.append(_exact_match(pred, ex.subject))
        f1s.append(_token_f1(pred, ex.subject))

    return {
        "exact_match": float(sum(ems) / max(1, len(ems))),
        "token_f1": float(sum(f1s) / max(1, len(f1s))),
    }


# -----------------------------
# Training
# -----------------------------
def _train(
    *,
    model_name: str,
    train_examples: List[SubjectExample],
    epochs: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
    output_dir: Path,
    eval_split: float,
    early_stop_patience: int,
):
    """
    Fine-tune a causal LM using Trainer.

    Returns:
      trainer, tokenizer, trained_model, train_seconds
    """
    _require_transformers()

    import torch
    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
    from transformers.trainer_callback import EarlyStoppingCallback

    set_seed(seed)

    tokenizer = _load_tokenizer(model_name)
    model = _load_model(model_name)  # IMPORTANT: fresh model instance for training

    raw_ds = _build_dataset(train_examples)

    split = raw_ds.train_test_split(test_size=eval_split, seed=seed)
    train_ds = _tokenize_dataset(split["train"], tokenizer)
    eval_ds = _tokenize_dataset(split["test"], tokenizer)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    use_fp16 = bool(torch.cuda.is_available())

    ta_kwargs = dict(
        output_dir=str(output_dir / "trainer_runs"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        fp16=use_fp16,
        seed=seed,
    )
    training_args = TrainingArguments(**_filter_kwargs(TrainingArguments, ta_kwargs))

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )
    trainer = Trainer(**_filter_kwargs(Trainer, trainer_kwargs))

    if early_stop_patience > 0:
        trainer.add_callback(
            EarlyStoppingCallback(early_stopping_patience=early_stop_patience)
        )

    t0 = time.perf_counter()
    trainer.train()
    t1 = time.perf_counter()
    train_seconds = float(t1 - t0)

    # Save final/best model + tokenizer
    model_out = output_dir / "model"
    model_out.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(model_out))
    tokenizer.save_pretrained(str(model_out))

    return trainer, tokenizer, trainer.model, train_seconds


def _extract_epoch_losses(
    trainer,
) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """Extract train loss and eval loss points per epoch from log_history."""
    train_pts: List[Tuple[int, float]] = []
    eval_pts: List[Tuple[int, float]] = []

    history = getattr(trainer.state, "log_history", []) or []
    for item in history:
        if "epoch" not in item:
            continue
        e = int(round(float(item["epoch"])))
        if "loss" in item:
            train_pts.append((e, float(item["loss"])))
        if "eval_loss" in item:
            eval_pts.append((e, float(item["eval_loss"])))

    def _dedupe(pts: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        m: Dict[int, float] = {}
        for e, v in pts:
            m[e] = v
        return sorted(m.items(), key=lambda x: x[0])

    return _dedupe(train_pts), _dedupe(eval_pts)


def _export_benchmark_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    """Export benchmark results to CSV (no pandas dependency)."""
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["Instruction", "Target", "Before", "After"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})


# -----------------------------
# Streamlit UI
# -----------------------------
def run() -> None:
    _require_transformers()

    st.markdown("### Overview")
    st.write(
        "This app fine-tunes a small Hugging Face causal LM to generate **logistics email subject lines**.\n\n"
        "**Key idea:** we compare outputs on a holdout benchmark not used during training.\n"
        "If fine-tuning worked, you should see the model follow the `Delay/Tracking/Approval` style more often.\n"
    )

    col_a, col_b, col_c = st.columns([1.2, 1.0, 1.0])
    with col_a:
        model_name = st.selectbox(
            "Base model",
            options=["distilgpt2", "sshleifer/tiny-gpt2"],
            index=0,
            help="distilgpt2 looks much better; tiny-gpt2 is mainly for speed/debug.",
        )
    with col_b:
        st.text_input("Device", value=_get_device_label(), disabled=True)
    with col_c:
        seed = int(
            st.number_input("Seed", min_value=0, max_value=10_000_000, value=42, step=1)
        )

    st.markdown("---")
    st.markdown("### Dataset")
    col_d1, col_d2 = st.columns([1, 1])
    with col_d1:
        n_aug = int(st.slider("Synthetic augmentation size", 0, 300, 120, 10))
    with col_d2:
        eval_split = float(st.slider("Validation split", 0.1, 0.4, 0.2, 0.05))

    examples = load_toy_subject_dataset(n_augmented=n_aug)
    st.caption(f"Training dataset size: **{len(examples)}** examples")

    with st.expander("Preview training examples (first 12)"):
        for i, ex in enumerate(examples[:12], start=1):
            st.markdown(f"**Example {i}**")
            st.write(f"Instruction: {ex.instruction}")
            st.write(f"Subject: {ex.subject}")
            st.markdown("---")

    st.markdown("---")
    st.markdown("### Inputs")
    instruction = st.text_area(
        "Instruction",
        value="Write an email subject for a shipment delayed due to weather. Mention the new ETA is tomorrow.",
        height=90,
    )

    st.markdown("### Generation controls")
    col_g1, col_g2 = st.columns([1, 1])
    with col_g1:
        max_new_tokens = int(st.slider("Max new tokens", 8, 64, 20, 1))
    with col_g2:
        decoding = st.selectbox("Decoding", ["beam", "greedy", "sample"], index=0)

    st.markdown("### Training controls")
    col_t1, col_t2, col_t3 = st.columns([1, 1, 1])
    with col_t1:
        epochs = int(st.slider("Epochs", 1, 8, 3, 1))
    with col_t2:
        learning_rate = float(
            st.select_slider(
                "Learning rate",
                options=[1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
                value=3e-4,
            )
        )
    with col_t3:
        batch_size = int(st.selectbox("Batch size", options=[1, 2, 4, 8], index=1))

    early_stop_patience = int(
        st.selectbox(
            "Early stopping patience (epochs)",
            [0, 1, 2, 3],
            index=2,
            help="Stops training if eval loss stops improving. Helps reduce overfitting/repetition.",
        )
    )

    st.markdown("---")
    st.markdown("### Before fine-tuning (TRUE pretrained)")
    try:
        tokenizer = _load_tokenizer(model_name)
        base_model_for_preview = _load_model(model_name)  # fresh base model for preview
        before = _generate_subject(
            model=base_model_for_preview,
            tokenizer=tokenizer,
            instruction=instruction,
            max_new_tokens=max_new_tokens,
            seed=seed,
            decoding=decoding,
        )
        st.code(before or "(empty)", language="text")
    except Exception as exc:
        st.error("Base model inference failed.")
        st.exception(exc)
        st.stop()

    st.markdown("---")
    st.markdown("### Fine-tune and compare")

    run_btn = st.button("🚀 Run fine-tuning", type="primary", use_container_width=True)
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
            "decoding": decoding,
            "n_augmented": n_aug,
            "eval_split": eval_split,
            "early_stop_patience": early_stop_patience,
            "instruction": instruction,
        },
    )

    with st.spinner("Training..."):
        try:
            trainer, tokenizer_ft, trained_model, train_seconds = _train(
                model_name=model_name,
                train_examples=examples,
                epochs=epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                seed=seed,
                output_dir=artifacts_dir,
                eval_split=eval_split,
                early_stop_patience=early_stop_patience,
            )
        except Exception as exc:
            st.error("Fine-tuning failed.")
            st.exception(exc)
            st.stop()

    # Curves
    st.markdown("#### Training curves")
    train_pts, eval_pts = _extract_epoch_losses(trainer)
    if train_pts or eval_pts:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        if train_pts:
            plt.plot(
                [e for e, _ in train_pts],
                [v for _, v in train_pts],
                marker="o",
                label="train_loss",
            )
        if eval_pts:
            plt.plot(
                [e for e, _ in eval_pts],
                [v for _, v in eval_pts],
                marker="o",
                label="eval_loss",
            )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss per Epoch")
        plt.legend()
        st.pyplot(fig, clear_figure=True)
    else:
        st.info("No loss logs found to plot.")

    st.markdown("### After fine-tuning")
    try:
        after = _generate_subject(
            model=trained_model,
            tokenizer=tokenizer_ft,
            instruction=instruction,
            max_new_tokens=max_new_tokens,
            seed=seed,
            decoding=decoding,
        )
        st.code(after or "(empty)", language="text")
    except Exception as exc:
        st.error("Fine-tuned model inference failed.")
        st.exception(exc)
        st.stop()

    # Holdout benchmark (WOW table)
    st.markdown("---")
    st.markdown("### Holdout benchmark (NOT trained on)")

    holdout = load_holdout_subject_benchmark()

    # IMPORTANT: use a fresh base model instance for benchmark "Before"
    base_model_for_benchmark = _load_model(model_name)

    base_scores = _score_examples(
        model=base_model_for_benchmark,
        tokenizer=tokenizer,
        items=holdout,
        max_new_tokens=max_new_tokens,
        seed=seed,
        decoding=decoding,
    )
    ft_scores = _score_examples(
        model=trained_model,
        tokenizer=tokenizer_ft,
        items=holdout,
        max_new_tokens=max_new_tokens,
        seed=seed,
        decoding=decoding,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Base Exact Match", f"{base_scores['exact_match']:.2f}")
        st.metric("Base Token F1", f"{base_scores['token_f1']:.2f}")
    with c2:
        st.metric("Fine-tuned Exact Match", f"{ft_scores['exact_match']:.2f}")
        st.metric("Fine-tuned Token F1", f"{ft_scores['token_f1']:.2f}")

    st.caption(
        "Exact Match is strict; Token F1 is more forgiving and better for small demos."
    )

    rows: List[Dict[str, str]] = []
    for i, ex in enumerate(holdout):
        pred_before = _generate_subject(
            model=base_model_for_benchmark,
            tokenizer=tokenizer,
            instruction=ex.instruction,
            max_new_tokens=max_new_tokens,
            seed=seed + 100 + i,
            decoding=decoding,
        )
        pred_after = _generate_subject(
            model=trained_model,
            tokenizer=tokenizer_ft,
            instruction=ex.instruction,
            max_new_tokens=max_new_tokens,
            seed=seed + 200 + i,
            decoding=decoding,
        )
        rows.append(
            {
                "Instruction": ex.instruction,
                "Target": ex.subject,
                "Before": pred_before,
                "After": pred_after,
            }
        )

    st.dataframe(rows, use_container_width=True)

    # Export benchmark CSV for offline inspection
    csv_path = artifacts_dir / "benchmark_holdout.csv"
    _export_benchmark_csv(csv_path, rows)
    st.write(f"Saved holdout benchmark CSV to: `{csv_path.as_posix()}`")

    st.markdown("---")
    st.markdown("### Run summary")
    st.write(f"**Training time:** {train_seconds:.2f} seconds")
    st.write(f"**Artifacts saved to:** `{artifacts_dir.as_posix()}`")
    st.write("Saved files include:")
    st.write("- `run_config.json` (your settings)")
    st.write("- `model/` (fine-tuned model + tokenizer)")
    st.write("- `benchmark_holdout.csv` (before/after on fixed holdout)")
