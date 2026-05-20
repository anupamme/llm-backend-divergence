"""Output-level divergence detection across backends."""

from __future__ import annotations

import re
import sqlite3
from collections import Counter
from itertools import combinations
from typing import Literal

import sacrebleu
from pydantic import BaseModel, ConfigDict

Verdict = Literal["unanimous", "majority", "split", "dispersed"]


class PairwiseMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)
    backend_a: str
    backend_b: str
    exact_match_rate: float
    avg_levenshtein: float
    avg_bleu: float


class PromptVerdict(BaseModel):
    model_config = ConfigDict(frozen=True)
    item_id: str
    prompt: str
    verdict: Verdict
    extracted_answers: dict[str, str]
    completions: dict[str, str]


class DivergenceReport(BaseModel):
    model_config = ConfigDict(frozen=True)
    dataset_name: str
    backends: list[str]
    n_prompts: int
    pairwise_metrics: list[PairwiseMetrics]
    prompt_verdicts: list[PromptVerdict]
    summary_markdown: str


def _levenshtein(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)

    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insert = prev_row[j + 1] + 1
            delete = curr_row[j] + 1
            substitute = prev_row[j] + (0 if c1 == c2 else 1)
            curr_row.append(min(insert, delete, substitute))
        prev_row = curr_row

    return prev_row[-1]


def _extract_answer(completion: str, category: str, item_id: str = "") -> str:
    """Extract the final answer from a completion based on dataset category."""
    if category == "gsm8k" or category == "arithmetic":
        match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", completion)
        if match:
            return match.group(1).replace(",", "")
        numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", completion)
        if numbers:
            return str(numbers[-1]).replace(",", "")
        return completion.strip()

    if category == "mmlu":
        match = re.search(r"\b([A-D])\b", completion)
        if match:
            return match.group(1)
        return completion.strip()

    if category == "canary" and item_id.startswith("canary-arith-"):
        numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", completion)
        if numbers:
            return str(numbers[0]).replace(",", "")
        return completion.strip()[:100]

    return completion.strip()[:100]


def _classify_verdict(answers: dict[str, str]) -> Verdict:
    """Classify the agreement level among backends for a single prompt."""
    if len(answers) <= 1:
        return "unanimous"

    counts = Counter(answers.values())
    n = len(answers)
    most_common_count = counts.most_common(1)[0][1]

    if most_common_count == n:
        return "unanimous"
    if most_common_count >= n - 1 and n >= 3:
        return "majority"
    if most_common_count > n / 2:
        return "split"
    return "dispersed"


def _generate_markdown_summary(
    dataset_name: str,
    backends: list[str],
    pairwise: list[PairwiseMetrics],
    verdicts: list[PromptVerdict],
) -> str:
    """Generate a markdown summary of divergence analysis."""
    lines: list[str] = []
    lines.append(f"# Divergence Report: {dataset_name}")
    lines.append("")
    lines.append(f"**Backends:** {', '.join(backends)}")
    lines.append(f"**Prompts analyzed:** {len(verdicts)}")
    lines.append("")

    # Verdict distribution
    verdict_counts: Counter[str] = Counter(v.verdict for v in verdicts)
    lines.append("## Verdict Distribution")
    lines.append("")
    lines.append("| Verdict | Count | Percentage |")
    lines.append("|---------|-------|------------|")
    for verdict_type in ["unanimous", "majority", "split", "dispersed"]:
        count = verdict_counts.get(verdict_type, 0)
        pct = (count / len(verdicts) * 100) if verdicts else 0
        lines.append(f"| {verdict_type} | {count} | {pct:.1f}% |")
    lines.append("")

    # Pairwise metrics
    lines.append("## Pairwise Metrics")
    lines.append("")
    lines.append("| Backend A | Backend B | Exact Match | Avg Levenshtein | Avg BLEU |")
    lines.append("|-----------|-----------|-------------|-----------------|----------|")
    for m in pairwise:
        lines.append(
            f"| {m.backend_a} | {m.backend_b} | "
            f"{m.exact_match_rate:.2%} | {m.avg_levenshtein:.1f} | "
            f"{m.avg_bleu:.2f} |"
        )
    lines.append("")

    return "\n".join(lines)


def compute_output_divergence(db_path: str, dataset_name: str) -> DivergenceReport:
    """Compute output-level divergence across backends for a dataset.

    Args:
        db_path: Path to SQLite results database.
        dataset_name: Name of the dataset to analyze.

    Returns:
        DivergenceReport with pairwise metrics and per-prompt verdicts.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        "SELECT ir.item_id, ir.prompt, ir.completion, ir.token_ids_json, "
        "ir.finish_reason, r.backend_name "
        "FROM inference_results ir "
        "JOIN runs r ON ir.run_id = r.run_id "
        "WHERE r.dataset_name = ? AND ir.finish_reason != 'error'",
        (dataset_name,),
    ).fetchall()
    conn.close()

    # Group by item_id → {backend_name: completion}
    completions_by_item: dict[str, dict[str, str]] = {}
    tokens_by_item: dict[str, dict[str, str]] = {}
    prompts_by_item: dict[str, str] = {}

    for row in rows:
        item_id = row["item_id"]
        backend = row["backend_name"]
        if item_id not in completions_by_item:
            completions_by_item[item_id] = {}
            tokens_by_item[item_id] = {}
        completions_by_item[item_id][backend] = row["completion"] or ""
        tokens_by_item[item_id][backend] = row["token_ids_json"] or "[]"
        prompts_by_item[item_id] = row["prompt"] or ""

    # Discover all backends
    all_backends = sorted({b for comps in completions_by_item.values() for b in comps})

    # Compute pairwise metrics
    pairwise_metrics: list[PairwiseMetrics] = []
    for ba, bb in combinations(all_backends, 2):
        exact_matches = 0
        lev_distances: list[int] = []
        bleu_scores: list[float] = []
        pair_count = 0

        for item_id, comps in completions_by_item.items():
            if ba not in comps or bb not in comps:
                continue
            pair_count += 1
            comp_a = comps[ba]
            comp_b = comps[bb]

            # Exact match (token-level)
            tok_a = tokens_by_item[item_id].get(ba, "[]")
            tok_b = tokens_by_item[item_id].get(bb, "[]")
            if tok_a == tok_b:
                exact_matches += 1

            # Levenshtein
            lev_distances.append(_levenshtein(comp_a, comp_b))

            # BLEU
            if comp_a and comp_b:
                bleu = sacrebleu.sentence_bleu(comp_a, [comp_b])
                bleu_scores.append(bleu.score)
            else:
                bleu_scores.append(0.0 if comp_a != comp_b else 100.0)

        if pair_count > 0:
            pairwise_metrics.append(
                PairwiseMetrics(
                    backend_a=ba,
                    backend_b=bb,
                    exact_match_rate=exact_matches / pair_count,
                    avg_levenshtein=sum(lev_distances) / pair_count,
                    avg_bleu=sum(bleu_scores) / pair_count,
                )
            )

    # Compute per-prompt verdicts
    prompt_verdicts: list[PromptVerdict] = []

    # Infer category from dataset_name
    category = dataset_name

    for item_id, comps in completions_by_item.items():
        extracted: dict[str, str] = {}
        for backend, completion in comps.items():
            extracted[backend] = _extract_answer(completion, category, item_id)

        verdict = _classify_verdict(extracted)
        prompt_verdicts.append(
            PromptVerdict(
                item_id=item_id,
                prompt=prompts_by_item.get(item_id, ""),
                verdict=verdict,
                extracted_answers=extracted,
                completions=comps,
            )
        )

    summary_md = _generate_markdown_summary(
        dataset_name, all_backends, pairwise_metrics, prompt_verdicts
    )

    return DivergenceReport(
        dataset_name=dataset_name,
        backends=all_backends,
        n_prompts=len(prompt_verdicts),
        pairwise_metrics=pairwise_metrics,
        prompt_verdicts=prompt_verdicts,
        summary_markdown=summary_md,
    )
