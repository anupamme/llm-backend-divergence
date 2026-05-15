"""Pure data-loading helpers for the dashboard (no streamlit imports)."""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict

from pydantic import BaseModel, ConfigDict

from divergence.analysis.output_divergence import compute_output_divergence

_CANARY_PREFIX_MAP: dict[str, str] = {
    "arith": "arithmetic",
    "tok": "tokenization",
    "logic": "logic",
    "ctx": "long_context",
    "fmt": "formatting",
}


class LatencyStats(BaseModel):
    model_config = ConfigDict(frozen=True)
    backend_name: str
    ttft_values: list[float]
    itl_values: list[float]
    total_latency_values: list[float]
    error_count: int
    total_count: int
    total_tokens: int


class CanaryDimensionStats(BaseModel):
    model_config = ConfigDict(frozen=True)
    dimension: str
    total: int
    disagreements: int
    disagreement_rate: float


class MmluSubjectStats(BaseModel):
    model_config = ConfigDict(frozen=True)
    subject: str
    total: int
    disagreements: int
    disagreement_rate: float


def get_available_datasets(db_path: str) -> list[str]:
    """Return distinct dataset names from the runs table."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT DISTINCT dataset_name FROM runs ORDER BY dataset_name"
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


def get_available_backends(db_path: str) -> list[str]:
    """Return distinct backend names from the runs table."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT DISTINCT backend_name FROM runs ORDER BY backend_name"
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


def load_latency_stats(db_path: str) -> list[LatencyStats]:
    """Load per-backend latency statistics from inference results."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT ir.ttft_ms, ir.total_latency_ms, "
        "ir.per_token_latency_ms_json, ir.token_ids_json, "
        "ir.finish_reason, r.backend_name "
        "FROM inference_results ir "
        "JOIN runs r ON ir.run_id = r.run_id"
    ).fetchall()
    conn.close()

    stats_by_backend: dict[str, dict[str, list[float] | int]] = defaultdict(
        lambda: {
            "ttft": [],
            "itl": [],
            "total": [],
            "errors": 0,
            "count": 0,
            "tokens": 0,
        }
    )

    for row in rows:
        backend: str = row["backend_name"]
        entry = stats_by_backend[backend]
        entry["count"] += 1  # type: ignore[operator]

        if row["finish_reason"] == "error":
            entry["errors"] += 1  # type: ignore[operator]
            continue

        if row["ttft_ms"] is not None:
            entry["ttft"].append(row["ttft_ms"])  # type: ignore[union-attr]
        if row["total_latency_ms"] is not None:
            entry["total"].append(row["total_latency_ms"])  # type: ignore[union-attr]

        if row["per_token_latency_ms_json"]:
            per_token: list[float] = json.loads(row["per_token_latency_ms_json"])
            entry["itl"].extend(per_token)  # type: ignore[union-attr]

        if row["token_ids_json"]:
            token_ids: list[int] = json.loads(row["token_ids_json"])
            entry["tokens"] += len(token_ids)  # type: ignore[operator]

    results: list[LatencyStats] = []
    for backend in sorted(stats_by_backend.keys()):
        e = stats_by_backend[backend]
        results.append(
            LatencyStats(
                backend_name=backend,
                ttft_values=e["ttft"],  # type: ignore[arg-type]
                itl_values=e["itl"],  # type: ignore[arg-type]
                total_latency_values=e["total"],  # type: ignore[arg-type]
                error_count=e["errors"],  # type: ignore[arg-type]
                total_count=e["count"],  # type: ignore[arg-type]
                total_tokens=e["tokens"],  # type: ignore[arg-type]
            )
        )
    return results


def _item_id_to_canary_dimension(item_id: str) -> str | None:
    """Extract canary precision dimension from item ID prefix."""
    # Format: canary-{abbrev}-NNN
    parts = item_id.split("-")
    if len(parts) >= 3 and parts[0] == "canary":
        return _CANARY_PREFIX_MAP.get(parts[1])
    return None


def _item_id_to_mmlu_subject(item_id: str) -> str | None:
    """Extract MMLU subject from item ID prefix."""
    # Format: mmlu-{subject}-NNNN
    parts = item_id.split("-")
    if len(parts) >= 3 and parts[0] == "mmlu":
        return parts[1]
    return None


def load_canary_breakdown(db_path: str) -> list[CanaryDimensionStats]:
    """Compute disagreement rate per canary precision dimension."""
    report = compute_output_divergence(db_path, "canary")
    if not report.prompt_verdicts:
        return []

    dim_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "disagreements": 0}
    )

    for v in report.prompt_verdicts:
        dim = _item_id_to_canary_dimension(v.item_id)
        if dim is None:
            continue
        dim_counts[dim]["total"] += 1
        if v.verdict != "unanimous":
            dim_counts[dim]["disagreements"] += 1

    results: list[CanaryDimensionStats] = []
    for dim in sorted(dim_counts.keys()):
        total = dim_counts[dim]["total"]
        disagreements = dim_counts[dim]["disagreements"]
        results.append(
            CanaryDimensionStats(
                dimension=dim,
                total=total,
                disagreements=disagreements,
                disagreement_rate=disagreements / total if total > 0 else 0.0,
            )
        )
    return results


def load_mmlu_subject_stats(db_path: str) -> list[MmluSubjectStats]:
    """Compute disagreement rate per MMLU subject."""
    report = compute_output_divergence(db_path, "mmlu")
    if not report.prompt_verdicts:
        return []

    subj_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "disagreements": 0}
    )

    for v in report.prompt_verdicts:
        subject = _item_id_to_mmlu_subject(v.item_id)
        if subject is None:
            continue
        subj_counts[subject]["total"] += 1
        if v.verdict != "unanimous":
            subj_counts[subject]["disagreements"] += 1

    results: list[MmluSubjectStats] = []
    for subject in sorted(subj_counts.keys()):
        total = subj_counts[subject]["total"]
        disagreements = subj_counts[subject]["disagreements"]
        results.append(
            MmluSubjectStats(
                subject=subject,
                total=total,
                disagreements=disagreements,
                disagreement_rate=disagreements / total if total > 0 else 0.0,
            )
        )
    return results
