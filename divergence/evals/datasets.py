"""Eval dataset loaders for divergence evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

CACHE_DIR = str(Path.home() / ".cache" / "llm-backend-divergence" / "datasets")

MMLU_SUBJECTS = [
    "high_school_mathematics",
    "professional_medicine",
    "formal_logic",
    "college_computer_science",
    "moral_scenarios",
]

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


class EvalItem(BaseModel):
    """A single evaluation item."""

    model_config = ConfigDict(frozen=True)

    id: str
    prompt: str
    reference_answer: str
    category: str
    precision_dimension: str | None = None


def load_gsm8k(n: int = 200) -> list[EvalItem]:
    """Load the first n test problems from the GSM8K dataset.

    Downloads from openai/gsm8k on HuggingFace Hub if not cached.
    """
    from datasets import load_dataset

    ds: Any = load_dataset("openai/gsm8k", "main", split="test", cache_dir=CACHE_DIR)

    items: list[EvalItem] = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        items.append(
            EvalItem(
                id=f"gsm8k-{i:04d}",
                prompt=row["question"],
                reference_answer=row["answer"],
                category="gsm8k",
            )
        )
    return items


def load_mmlu_subset(n_per_subject: int = 100) -> list[EvalItem]:
    """Load n_per_subject items from each of 5 MMLU subjects.

    Subjects chosen to stress different reasoning modes:
    - high_school_mathematics
    - professional_medicine
    - formal_logic
    - college_computer_science
    - moral_scenarios
    """
    from datasets import load_dataset

    items: list[EvalItem] = []
    choices_labels = ["A", "B", "C", "D"]

    for subject in MMLU_SUBJECTS:
        ds: Any = load_dataset("cais/mmlu", subject, split="test", cache_dir=CACHE_DIR)

        for i, row in enumerate(ds):
            if i >= n_per_subject:
                break

            choices_text = "\n".join(
                f"{choices_labels[j]}. {row['choices'][j]}"
                for j in range(len(row["choices"]))
            )
            prompt = (
                f"{row['question']}\n\n{choices_text}\n\n"
                f"Answer with the letter of the correct choice."
            )
            answer_idx: int = row["answer"]
            reference_answer = choices_labels[answer_idx]

            items.append(
                EvalItem(
                    id=f"mmlu-{subject}-{i:04d}",
                    prompt=prompt,
                    reference_answer=reference_answer,
                    category=f"mmlu-{subject}",
                )
            )
    return items


def load_canary_set() -> list[EvalItem]:
    """Load hand-crafted canary prompts from data/canary.jsonl.

    These prompts target known precision-sensitive operations:
    arithmetic, tokenization, logic, long_context, and formatting.
    """
    canary_path = DATA_DIR / "canary.jsonl"
    items: list[EvalItem] = []
    with canary_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data: dict[str, Any] = json.loads(line)
            items.append(EvalItem(**data))
    return items
