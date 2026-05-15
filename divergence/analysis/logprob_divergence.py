"""Logprob-level divergence detection across backends."""

from __future__ import annotations

import json
import math
import sqlite3
from itertools import combinations

from pydantic import BaseModel, ConfigDict

_APPROXIMATION_NOTE = (
    "KL divergence is approximated using only the logprob of the chosen token. "
    "For token t at position i, we compute max(0, exp(lp_a) * (lp_a - lp_b)) "
    "which is the contribution of t to KL(A || B), clamped to non-negative. "
    "This is a lower bound on true KL divergence since we cannot observe the "
    "full next-token distribution. The approximation is most accurate when the "
    "chosen token has high probability (logprob close to 0)."
)


class LogprobDivergenceConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    kl_threshold: float = 0.1
    top_n: int = 50


class TokenDelta(BaseModel):
    model_config = ConfigDict(frozen=True)
    token_id: int
    position: int
    logprob_a: float
    logprob_b: float
    delta: float
    kl_contribution: float


class PairwiseItemDivergence(BaseModel):
    model_config = ConfigDict(frozen=True)
    item_id: str
    prompt: str
    backend_a: str
    backend_b: str
    n_tokens: int
    mean_abs_delta: float
    max_abs_delta: float
    p95_abs_delta: float
    mean_kl_contribution: float
    max_kl_contribution: float
    p95_kl_contribution: float
    is_alert: bool


class TokenizationMismatch(BaseModel):
    model_config = ConfigDict(frozen=True)
    item_id: str
    prompt: str
    backend_a: str
    backend_b: str
    n_tokens_a: int
    n_tokens_b: int


class LogprobDivergenceReport(BaseModel):
    model_config = ConfigDict(frozen=True)
    backends: list[str]
    n_items_analyzed: int
    n_items_with_alerts: int
    n_tokenization_mismatches: int
    config: LogprobDivergenceConfig
    divergences: list[PairwiseItemDivergence]
    alerts: list[PairwiseItemDivergence]
    tokenization_mismatches: list[TokenizationMismatch]
    summary_markdown: str
    approximation_note: str


def _percentile(values: list[float], p: float) -> float:
    """Compute the p-th percentile (0-100) of a sorted list."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    idx = (p / 100.0) * (n - 1)
    lower = int(idx)
    upper = min(lower + 1, n - 1)
    frac = idx - lower
    return sorted_vals[lower] * (1 - frac) + sorted_vals[upper] * frac


def _compute_token_deltas(
    token_ids: list[int],
    logprobs_a: list[float],
    logprobs_b: list[float],
) -> list[TokenDelta]:
    """Compute per-token deltas and KL contributions for aligned sequences."""
    deltas: list[TokenDelta] = []
    for i, tid in enumerate(token_ids):
        lp_a = logprobs_a[i]
        lp_b = logprobs_b[i]
        delta = lp_a - lp_b
        kl_contrib = max(0.0, math.exp(lp_a) * delta)
        deltas.append(
            TokenDelta(
                token_id=tid,
                position=i,
                logprob_a=lp_a,
                logprob_b=lp_b,
                delta=delta,
                kl_contribution=kl_contrib,
            )
        )
    return deltas


def _generate_markdown_summary(
    backends: list[str],
    config: LogprobDivergenceConfig,
    divergences: list[PairwiseItemDivergence],
    alerts: list[PairwiseItemDivergence],
    mismatches: list[TokenizationMismatch],
    n_items_analyzed: int,
) -> str:
    """Generate markdown summary of logprob divergence analysis."""
    lines: list[str] = []
    lines.append("# Logprob Divergence Report")
    lines.append("")
    lines.append(f"**Backends:** {', '.join(backends)}")
    lines.append(f"**Items analyzed:** {n_items_analyzed}")
    lines.append(f"**KL threshold:** {config.kl_threshold}")
    lines.append(f"**Alerts:** {len(alerts)}")
    lines.append(f"**Tokenization mismatches:** {len(mismatches)}")
    lines.append("")

    if alerts:
        lines.append("## Divergence Alerts")
        lines.append("")
        lines.append(
            "| Item | Backend A | Backend B | Mean |delta| | Max |delta| | Mean KL |"
        )
        lines.append(
            "|------|-----------|-----------|------------|------------|---------|"
        )
        for a in alerts[:20]:
            lines.append(
                f"| {a.item_id} | {a.backend_a} | {a.backend_b} | "
                f"{a.mean_abs_delta:.4f} | {a.max_abs_delta:.4f} | "
                f"{a.mean_kl_contribution:.4f} |"
            )
        lines.append("")

    if divergences:
        lines.append("## Top Divergent Prompts")
        lines.append("")
        lines.append("| Item | Backend A | Backend B | Tokens | Mean KL | P95 KL |")
        lines.append("|------|-----------|-----------|--------|---------|--------|")
        for d in divergences[:20]:
            lines.append(
                f"| {d.item_id} | {d.backend_a} | {d.backend_b} | "
                f"{d.n_tokens} | {d.mean_kl_contribution:.4f} | "
                f"{d.p95_kl_contribution:.4f} |"
            )
        lines.append("")

    return "\n".join(lines)


def compute_logprob_divergence(
    db_path: str,
    *,
    config: LogprobDivergenceConfig | None = None,
) -> LogprobDivergenceReport:
    """Compute logprob-level divergence across backends.

    Analyzes scoring_results to detect subtle numerical drift between backends
    by comparing per-token logprobs of the chosen token.
    """
    if config is None:
        config = LogprobDivergenceConfig()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        "SELECT sr.item_id, sr.prompt, sr.completion, sr.token_ids_json, "
        "sr.logprobs_json, r.backend_name "
        "FROM scoring_results sr "
        "JOIN runs r ON sr.run_id = r.run_id "
        "WHERE sr.error_message IS NULL "
        "AND sr.logprobs_json IS NOT NULL "
        "AND sr.token_ids_json IS NOT NULL",
    ).fetchall()
    conn.close()

    # Group by (item_id, completion) → {backend: (token_ids, logprobs, prompt)}
    groups: dict[tuple[str, str], dict[str, tuple[list[int], list[float], str]]] = {}

    for row in rows:
        item_id: str = row["item_id"]
        completion: str = row["completion"] or ""
        backend: str = row["backend_name"]
        token_ids: list[int] = json.loads(row["token_ids_json"])
        logprobs: list[float] = json.loads(row["logprobs_json"])
        prompt: str = row["prompt"] or ""

        if not token_ids or not logprobs:
            continue

        key = (item_id, completion)
        if key not in groups:
            groups[key] = {}
        groups[key][backend] = (token_ids, logprobs, prompt)

    all_backends: set[str] = set()
    all_divergences: list[PairwiseItemDivergence] = []
    all_mismatches: list[TokenizationMismatch] = []
    items_analyzed: set[tuple[str, str]] = set()

    for (item_id, _completion), backend_data in groups.items():
        if len(backend_data) < 2:
            continue

        for ba, bb in combinations(sorted(backend_data.keys()), 2):
            all_backends.add(ba)
            all_backends.add(bb)
            items_analyzed.add((item_id, f"{ba}:{bb}"))

            token_ids_a, logprobs_a, prompt = backend_data[ba]
            token_ids_b, logprobs_b, _ = backend_data[bb]

            # Tokenization mismatch check
            if token_ids_a != token_ids_b:
                all_mismatches.append(
                    TokenizationMismatch(
                        item_id=item_id,
                        prompt=prompt,
                        backend_a=ba,
                        backend_b=bb,
                        n_tokens_a=len(token_ids_a),
                        n_tokens_b=len(token_ids_b),
                    )
                )
                continue

            # Compute per-token deltas
            token_deltas = _compute_token_deltas(token_ids_a, logprobs_a, logprobs_b)

            if not token_deltas:
                continue

            abs_deltas = [abs(td.delta) for td in token_deltas]
            kl_contribs = [td.kl_contribution for td in token_deltas]

            divergence = PairwiseItemDivergence(
                item_id=item_id,
                prompt=prompt,
                backend_a=ba,
                backend_b=bb,
                n_tokens=len(token_deltas),
                mean_abs_delta=sum(abs_deltas) / len(abs_deltas),
                max_abs_delta=max(abs_deltas),
                p95_abs_delta=_percentile(abs_deltas, 95),
                mean_kl_contribution=sum(kl_contribs) / len(kl_contribs),
                max_kl_contribution=max(kl_contribs),
                p95_kl_contribution=_percentile(kl_contribs, 95),
                is_alert=sum(kl_contribs) / len(kl_contribs) > config.kl_threshold,
            )
            all_divergences.append(divergence)

    # Sort by mean_kl descending, cap to top_n
    all_divergences.sort(key=lambda d: d.mean_kl_contribution, reverse=True)
    top_divergences = all_divergences[: config.top_n]
    alerts = [d for d in all_divergences if d.is_alert]

    sorted_backends = sorted(all_backends)
    summary_md = _generate_markdown_summary(
        sorted_backends,
        config,
        top_divergences,
        alerts,
        all_mismatches,
        len(items_analyzed),
    )

    return LogprobDivergenceReport(
        backends=sorted_backends,
        n_items_analyzed=len(items_analyzed),
        n_items_with_alerts=len(alerts),
        n_tokenization_mismatches=len(all_mismatches),
        config=config,
        divergences=top_divergences,
        alerts=alerts,
        tokenization_mismatches=all_mismatches,
        summary_markdown=summary_md,
        approximation_note=_APPROXIMATION_NOTE,
    )


def visualize_token_divergence(
    db_path: str,
    item_id: str,
    backend_a: str,
    backend_b: str,
) -> list[TokenDelta]:
    """Return per-token logprob deltas for a specific prompt and backend pair.

    For dashboard integration. Returns an empty list if data is missing
    or tokenization mismatches.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        "SELECT sr.token_ids_json, sr.logprobs_json, sr.completion, "
        "r.backend_name "
        "FROM scoring_results sr "
        "JOIN runs r ON sr.run_id = r.run_id "
        "WHERE sr.item_id = ? "
        "AND r.backend_name IN (?, ?) "
        "AND sr.error_message IS NULL "
        "AND sr.logprobs_json IS NOT NULL "
        "AND sr.token_ids_json IS NOT NULL",
        (item_id, backend_a, backend_b),
    ).fetchall()
    conn.close()

    # Group by backend, matching on completion
    data_a: tuple[list[int], list[float]] | None = None
    data_b: tuple[list[int], list[float]] | None = None

    # Find rows with matching completions
    by_backend: dict[str, list[tuple[str, list[int], list[float]]]] = {
        backend_a: [],
        backend_b: [],
    }
    for row in rows:
        backend: str = row["backend_name"]
        completion: str = row["completion"] or ""
        token_ids: list[int] = json.loads(row["token_ids_json"])
        logprobs: list[float] = json.loads(row["logprobs_json"])
        if backend in by_backend and token_ids and logprobs:
            by_backend[backend].append((completion, token_ids, logprobs))

    if not by_backend[backend_a] or not by_backend[backend_b]:
        return []

    # Match on same completion
    for comp_a, tids_a, lps_a in by_backend[backend_a]:
        for comp_b, tids_b, lps_b in by_backend[backend_b]:
            if comp_a == comp_b and tids_a == tids_b:
                data_a = (tids_a, lps_a)
                data_b = (tids_b, lps_b)
                break
        if data_a is not None:
            break

    if data_a is None or data_b is None:
        return []

    return _compute_token_deltas(data_a[0], data_a[1], data_b[1])
