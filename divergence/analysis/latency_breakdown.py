"""Per-stage latency breakdown analysis."""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict

from pydantic import BaseModel, ConfigDict


class BackendLatency(BaseModel):
    model_config = ConfigDict(frozen=True)
    backend_name: str
    n_samples: int
    prefill_p50_ms: float
    prefill_p95_ms: float
    decode_per_token_p50_ms: float
    decode_per_token_p95_ms: float
    total_p50_ms: float
    total_p95_ms: float


class LatencyBreakdownReport(BaseModel):
    model_config = ConfigDict(frozen=True)
    backends: list[BackendLatency]


def _percentile(values: list[float], pct: int) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int(len(sorted_vals) * pct / 100)
    idx = min(idx, len(sorted_vals) - 1)
    return sorted_vals[idx]


def compute_latency_breakdown(db_path: str) -> LatencyBreakdownReport:
    """Compute per-stage latency statistics for each backend."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """
        SELECT backend_name, ttft_ms, total_latency_ms, per_token_latency_ms_json
        FROM inference_results
        WHERE error_message IS NULL AND ttft_ms IS NOT NULL
        """
    ).fetchall()
    conn.close()

    data: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"ttft": [], "decode_per_tok": [], "total": []}
    )

    for backend, ttft, total, ptl_json in rows:
        data[backend]["ttft"].append(ttft)
        data[backend]["total"].append(total)
        if ptl_json:
            ptl = json.loads(ptl_json)
            if len(ptl) > 1:
                data[backend]["decode_per_tok"].extend(ptl[1:])

    backends = []
    for name in sorted(data.keys()):
        d = data[name]
        backends.append(
            BackendLatency(
                backend_name=name,
                n_samples=len(d["ttft"]),
                prefill_p50_ms=_percentile(d["ttft"], 50),
                prefill_p95_ms=_percentile(d["ttft"], 95),
                decode_per_token_p50_ms=_percentile(d["decode_per_tok"], 50),
                decode_per_token_p95_ms=_percentile(d["decode_per_tok"], 95),
                total_p50_ms=_percentile(d["total"], 50),
                total_p95_ms=_percentile(d["total"], 95),
            )
        )

    return LatencyBreakdownReport(backends=backends)
