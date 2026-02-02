"""
I/O and artifact helpers for the LLM Lab.

This module keeps small, beginner-friendly utilities:
- Creating a tiny in-repo toy dataset for fine-tuning
- Managing timestamped artifact directories
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class SubjectExample:
    """A single supervised example for subject line generation."""

    instruction: str
    subject: str


def load_toy_subject_dataset() -> List[SubjectExample]:
    """
    Return a tiny toy dataset for logistics/operations email subject generation.

    This is intentionally small to keep CPU fine-tuning fast.
    You can expand it later or replace it with a real dataset.
    """
    return [
        SubjectExample(
            instruction="Write an email subject for a shipment delayed due to weather. Mention the new ETA is tomorrow.",
            subject="Weather Delay: Updated ETA for Shipment (Arrives Tomorrow)",
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


def ensure_dir(path: Path) -> None:
    """Create a directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def timestamp_now() -> str:
    """Return a filesystem-friendly timestamp string."""
    # Example: 2026-02-02_10-15-30
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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
