"""
I/O and artifact helpers for the LLM Lab.

This module contains small, beginner-friendly utilities:
- Building an in-repo toy dataset for fine-tuning (with optional deterministic augmentation)
- Providing a fixed holdout benchmark (not used for training) to compare before vs after
- Managing timestamped artifact directories
- Saving JSON artifacts

Why augmentation?
- A tiny dataset (10 examples) often causes overfitting and repetition.
- Deterministic template augmentation creates diversity while staying reproducible and CPU-friendly.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


# -----------------------------
# Data containers
# -----------------------------
@dataclass(frozen=True)
class SubjectExample:
    """One supervised example: instruction -> subject line."""

    instruction: str
    subject: str


# -----------------------------
# Small filesystem helpers
# -----------------------------
def ensure_dir(path: Path) -> None:
    """Create a directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def timestamp_now() -> str:
    """Return a filesystem-friendly timestamp string. Example: 2026-02-02_10-15-30"""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def finetuning_artifact_dir(root: Path | str = "artifacts") -> Path:
    """
    Create and return a timestamped artifact directory for fine-tuning outputs.

    Structure:
      ./artifacts/finetuning/<timestamp>/
    """
    root_path = Path(root)
    out_dir = root_path / "finetuning" / timestamp_now()
    ensure_dir(out_dir)
    return out_dir


def save_json(path: Path, payload: Dict) -> None:
    """Save a dict to JSON (pretty-printed)."""
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


# -----------------------------
# Text helpers (for dataset hygiene)
# -----------------------------
_word_re = re.compile(r"\s+")


def _normalize_spaces(s: str) -> str:
    """Normalize whitespace to keep instructions/subjects consistent."""
    return _word_re.sub(" ", s.strip())


def _dedupe_by_instruction(examples: Iterable[SubjectExample]) -> List[SubjectExample]:
    """Remove duplicates by instruction (case-insensitive). Keep first occurrence."""
    seen = set()
    out: List[SubjectExample] = []
    for ex in examples:
        key = ex.instruction.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(ex)
    return out


# -----------------------------
# Seed dataset (handwritten)
# -----------------------------
def _seed_subject_dataset() -> List[SubjectExample]:
    """
    A small set of high-quality seed examples.

    These are the "gold" examples that define the style you want the model to learn.
    Augmentation expands this set deterministically to create diversity.
    """
    return [
        SubjectExample(
            instruction="Write an email subject for a shipment delayed due to weather. Mention the new ETA is tomorrow.",
            subject="Weather Delay: Updated ETA for Shipment (Tomorrow)",
        ),
        SubjectExample(
            instruction="Write an email subject requesting a POD for delivery completed yesterday.",
            subject="Request: Proof of Delivery (POD) for Yesterday’s Delivery",
        ),
        SubjectExample(
            instruction="Write an email subject confirming pickup appointment at 3 PM today at the Chicago warehouse.",
            subject="Pickup Confirmed: 3:00 PM Today at Chicago Warehouse",
        ),
        SubjectExample(
            instruction="Write an email subject asking the carrier for updated tracking on load 47219.",
            subject="Tracking Update Needed: Load 47219",
        ),
        SubjectExample(
            instruction="Write an email subject notifying a customer about a route change to avoid congestion.",
            subject="Route Update: Adjusted Path to Avoid Congestion",
        ),
        SubjectExample(
            instruction="Write an email subject for a rate confirmation attached for a new lane.",
            subject="Rate Confirmation Attached: New Lane Details",
        ),
        SubjectExample(
            instruction="Write an email subject escalating a missed pickup and asking for immediate reschedule options.",
            subject="Escalation: Missed Pickup — Immediate Reschedule Needed",
        ),
        SubjectExample(
            instruction="Write an email subject confirming delivery appointment scheduled for Friday 10 AM.",
            subject="Delivery Appointment Confirmed: Friday 10:00 AM",
        ),
        SubjectExample(
            instruction="Write an email subject requesting updated accessorial charges breakdown for an invoice.",
            subject="Request: Accessorial Charges Breakdown for Invoice",
        ),
        SubjectExample(
            instruction="Write an email subject notifying that a trailer is ready for drop-and-hook at dock 7.",
            subject="Trailer Ready: Drop-and-Hook Available at Dock 7",
        ),
    ]


# -----------------------------
# Deterministic augmentation
# -----------------------------
def build_subject_dataset(
    *,
    seed_examples: List[SubjectExample],
    n_augmented: int = 120,
) -> List[SubjectExample]:
    """
    Expand seed examples into a larger deterministic dataset using templates.

    Why deterministic?
    - Reproducible (no random sampling)
    - Still increases diversity dramatically
    - Makes "before vs after fine-tuning" much more obvious

    Args:
        seed_examples: the handwritten examples that define the desired style.
        n_augmented: how many synthetic examples to generate.

    Returns:
        A deduped list of SubjectExample containing seeds + augmented.
    """
    # Domain knobs (small, curated lists)
    delay_reasons: List[Tuple[str, str]] = [
        ("weather", "Weather Delay"),
        ("traffic", "Traffic Delay"),
        ("carrier capacity", "Carrier Capacity"),
        ("warehouse backlog", "Warehouse Backlog"),
        ("customs inspection", "Customs Hold"),
        ("equipment issue", "Equipment Issue"),
        ("port congestion", "Port Congestion"),
    ]
    etas: List[Tuple[str, str]] = [
        ("tomorrow", "Tomorrow"),
        ("next business day", "Next Business Day"),
        ("by end of day", "EOD"),
        ("within 48 hours", "48 Hours"),
    ]
    actions: List[Tuple[str, str]] = [
        ("confirm the updated ETA", "Updated ETA Confirmed"),
        ("request approval", "Approval Needed"),
        ("share a tracking update", "Tracking Update"),
        ("notify of a reschedule", "Rescheduled"),
        ("ask for a pickup window", "Pickup Window Needed"),
    ]
    objects = ["shipment", "load", "freight", "order", "delivery"]
    tones = ["professional", "clear", "short", "direct"]

    # (instruction template, subject template)
    templates: List[Tuple[str, str]] = [
        (
            "Write a {tone} email subject for a {obj} delayed due to {reason}. Mention the new ETA is {eta}.",
            "{reason_tag}: Updated ETA for {obj_cap} ({eta_tag})",
        ),
        (
            "Create a {tone} subject line to {action}. Context: {obj} impacted by {reason}. ETA: {eta}.",
            "{action_tag}: {reason_tag} ({eta_tag})",
        ),
        (
            "Generate a {tone} subject for a customer update: {obj} delayed from {reason}. New ETA is {eta}.",
            "Customer Update: {reason_tag} — {eta_tag}",
        ),
        (
            "Write a {tone} email subject to notify about {reason} affecting the {obj}. ETA now {eta}.",
            "{reason_tag}: {obj_cap} ETA {eta_tag}",
        ),
    ]

    augmented: List[SubjectExample] = []

    i = 0
    for reason, reason_tag in delay_reasons:
        for eta, eta_tag in etas:
            for action, action_tag in actions:
                obj = objects[i % len(objects)]
                tone = tones[i % len(tones)]
                instr_tpl, subj_tpl = templates[i % len(templates)]

                instruction = instr_tpl.format(
                    tone=tone,
                    obj=obj,
                    obj_cap=obj.capitalize(),
                    reason=reason,
                    eta=eta,
                    action=action,
                )
                subject = subj_tpl.format(
                    reason_tag=reason_tag,
                    eta_tag=eta_tag,
                    action_tag=action_tag,
                    obj_cap=obj.capitalize(),
                )

                augmented.append(
                    SubjectExample(
                        instruction=_normalize_spaces(instruction),
                        subject=_normalize_spaces(subject),
                    )
                )

                i += 1
                if len(augmented) >= n_augmented:
                    break
            if len(augmented) >= n_augmented:
                break
        if len(augmented) >= n_augmented:
            break

    combined = seed_examples + augmented
    combined = [
        SubjectExample(_normalize_spaces(ex.instruction), _normalize_spaces(ex.subject))
        for ex in combined
    ]
    return _dedupe_by_instruction(combined)


def load_toy_subject_dataset(n_augmented: int = 120) -> List[SubjectExample]:
    """
    Return an in-repo dataset for fine-tuning.

    This returns:
    - A small set of high-quality seed examples
    - Plus deterministic augmentation to reach a larger dataset

    Args:
        n_augmented: number of synthetic examples to add (0 disables augmentation).
    """
    seeds = _seed_subject_dataset()
    if n_augmented <= 0:
        return seeds
    return build_subject_dataset(seed_examples=seeds, n_augmented=n_augmented)


# -----------------------------
# Holdout benchmark (not for training)
# -----------------------------
def load_holdout_subject_benchmark() -> List[SubjectExample]:
    """
    Fixed benchmark examples NOT used for training.

    Why this exists:
    - You can show a clean before/after comparison on examples the model never trained on.
    - This makes the fine-tuning demo feel much more real.

    Keep this list small and stable. It should not overlap with training instructions.
    """
    return [
        SubjectExample(
            instruction="Write a subject for a delivery delayed due to carrier capacity. New ETA is next business day.",
            subject="Carrier Capacity: Updated ETA for Delivery (Next Business Day)",
        ),
        SubjectExample(
            instruction="Create a subject line to share a tracking update: load impacted by port congestion. ETA is by end of day.",
            subject="Tracking Update: Port Congestion (EOD)",
        ),
        SubjectExample(
            instruction="Generate a short customer update subject: shipment delayed from customs inspection. ETA within 48 hours.",
            subject="Customer Update: Customs Hold — 48 Hours",
        ),
        SubjectExample(
            instruction="Write a professional subject to notify about equipment issue affecting the freight. ETA is tomorrow.",
            subject="Equipment Issue: Freight ETA Tomorrow",
        ),
        SubjectExample(
            instruction="Create a clear subject to request approval related to traffic delay. ETA is next business day.",
            subject="Approval Needed: Traffic Delay (Next Business Day)",
        ),
        SubjectExample(
            instruction="Write a subject for weather impacting the order. Mention ETA is by end of day.",
            subject="Weather Delay: Order ETA EOD",
        ),
    ]
