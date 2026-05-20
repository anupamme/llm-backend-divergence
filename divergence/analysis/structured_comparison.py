"""Structured within-framework vs cross-framework divergence analysis."""

from __future__ import annotations

import sqlite3
from collections import defaultdict

from pydantic import BaseModel, ConfigDict

from divergence.analysis.output_divergence import _extract_answer

WITHIN_FRAMEWORK: list[tuple[str, str]] = [
    ("mlx-fp16", "mlx-q4"),
    ("llamacpp-q8", "llamacpp-q4km"),
]

CROSS_FRAMEWORK_MATCHED: list[tuple[str, str]] = [
    ("mlx-fp16", "torch-mps"),
]

CROSS_FRAMEWORK_CONFOUNDED: list[tuple[str, str]] = [
    ("mlx-fp16", "llamacpp-q8"),
    ("mlx-fp16", "llamacpp-q4km"),
    ("mlx-q4", "llamacpp-q8"),
    ("mlx-q4", "llamacpp-q4km"),
    ("torch-mps", "llamacpp-q8"),
    ("torch-mps", "llamacpp-q4km"),
    ("mlx-q4", "torch-mps"),
]


class GroupAgreement(BaseModel):
    model_config = ConfigDict(frozen=True)
    group_name: str
    pairs: list[tuple[str, str]]
    agree: int
    total: int

    @property
    def rate(self) -> float:
        return self.agree / self.total if self.total > 0 else 0.0


class StructuredComparisonReport(BaseModel):
    model_config = ConfigDict(frozen=True)
    dataset_name: str
    within_framework: GroupAgreement
    cross_framework_matched: GroupAgreement
    cross_framework_confounded: GroupAgreement


def compute_structured_comparison(
    db_path: str,
    dataset_name: str,
) -> StructuredComparisonReport:
    """Compute extracted-answer agreement rates by comparison group."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """
        SELECT ir.item_id, ir.backend_name, ir.completion, ir.timestamp
        FROM inference_results ir
        JOIN runs r ON ir.run_id = r.run_id
        WHERE r.dataset_name = ?
          AND ir.error_message IS NULL
          AND ir.completion IS NOT NULL
        ORDER BY ir.timestamp DESC
        """,
        (dataset_name,),
    ).fetchall()
    conn.close()

    seen: set[tuple[str, str]] = set()
    items: dict[str, dict[str, str]] = defaultdict(dict)
    for item_id, backend, completion, _ts in rows:
        key = (item_id, backend)
        if key in seen:
            continue
        seen.add(key)
        items[item_id][backend] = _extract_answer(completion, dataset_name, item_id)

    def _agreement(pairs: list[tuple[str, str]]) -> tuple[int, int]:
        agree = 0
        total = 0
        for answers in items.values():
            for a, b in pairs:
                if a in answers and b in answers:
                    total += 1
                    if answers[a] == answers[b]:
                        agree += 1
        return agree, total

    w_agree, w_total = _agreement(WITHIN_FRAMEWORK)
    cm_agree, cm_total = _agreement(CROSS_FRAMEWORK_MATCHED)
    cc_agree, cc_total = _agreement(CROSS_FRAMEWORK_CONFOUNDED)

    return StructuredComparisonReport(
        dataset_name=dataset_name,
        within_framework=GroupAgreement(
            group_name="Within-framework",
            pairs=WITHIN_FRAMEWORK,
            agree=w_agree,
            total=w_total,
        ),
        cross_framework_matched=GroupAgreement(
            group_name="Cross-framework (matched precision)",
            pairs=CROSS_FRAMEWORK_MATCHED,
            agree=cm_agree,
            total=cm_total,
        ),
        cross_framework_confounded=GroupAgreement(
            group_name="Cross-framework (confounded)",
            pairs=CROSS_FRAMEWORK_CONFOUNDED,
            agree=cc_agree,
            total=cc_total,
        ),
    )
