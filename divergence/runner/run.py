"""Eval runner — execute one backend across one dataset."""

from __future__ import annotations

import logging
import math
import time
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from divergence.backends.base import Backend
from divergence.backends.schema import InferenceResult
from divergence.evals.datasets import EvalItem
from divergence.runner.db import (
    get_completed_item_ids,
    init_db,
    insert_inference_result,
    insert_run,
    insert_scoring_result,
)

logger = logging.getLogger(__name__)


def _default_run_id() -> str:
    return str(uuid4())


class RunConfig(BaseModel):
    """Configuration for an evaluation run."""

    model_config = ConfigDict(frozen=True)

    max_tokens: int = 256
    temperature: float = 0.0
    seed: int = 42
    output_db_path: str = "results.db"
    run_id: str = Field(default_factory=_default_run_id)
    resume: bool = False
    score_completion: Literal["model", "reference"] = "model"
    dataset_name: str = "unknown"


class RunSummary(BaseModel):
    """Summary statistics from a completed evaluation run."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    completed: int
    errors: int
    total_wall_time_s: float
    ttft_p50_ms: float
    ttft_p95_ms: float
    ttft_p99_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float


def _percentile(values: list[float], p: float) -> float:
    """Compute the p-th percentile (0-100) from a sorted list."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (p / 100.0) * (len(sorted_vals) - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def run_eval(
    backend: Backend,
    dataset: list[EvalItem],
    config: RunConfig,
) -> RunSummary:
    """Execute an evaluation run: iterate dataset, generate, score, persist.

    Returns a RunSummary with completion counts and latency percentiles.
    """
    from tqdm import tqdm

    conn = init_db(config.output_db_path)

    insert_run(
        conn,
        run_id=config.run_id,
        backend_name=backend.name,
        model_id=backend.metadata.name,
        dataset_name=config.dataset_name,
        started_at=datetime.now(tz=UTC).isoformat(),
        config_json=config.model_dump_json(),
    )

    items_to_run = dataset
    if config.resume:
        completed_ids = get_completed_item_ids(conn, config.run_id)
        items_to_run = [item for item in dataset if item.id not in completed_ids]

    completed = 0
    errors = 0
    ttft_values: list[float] = []
    latency_values: list[float] = []

    start_wall = time.perf_counter()

    for item in tqdm(items_to_run, desc=f"Running {backend.name}"):
        result: InferenceResult | None = None
        error_msg: str | None = None

        try:
            result = backend.generate(
                item.prompt,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                seed=config.seed,
            )
            ttft_values.append(result.ttft_ms)
            latency_values.append(result.total_latency_ms)
            completed += 1
        except Exception as e:
            error_msg = str(e)
            errors += 1
            logger.warning("Error on item %s: %s", item.id, error_msg)

        insert_inference_result(conn, config.run_id, item.id, result, error_msg)

        # Scoring
        try:
            if result is not None:
                if config.score_completion == "model":
                    score_text = result.completion
                else:
                    score_text = item.reference_answer
                score_result = backend.score(item.prompt, score_text)
                insert_scoring_result(conn, config.run_id, item.id, score_result, None)
        except NotImplementedError:
            pass
        except Exception as e:
            insert_scoring_result(conn, config.run_id, item.id, None, str(e))

    total_wall_time = time.perf_counter() - start_wall
    conn.close()

    return RunSummary(
        run_id=config.run_id,
        completed=completed,
        errors=errors,
        total_wall_time_s=total_wall_time,
        ttft_p50_ms=_percentile(ttft_values, 50),
        ttft_p95_ms=_percentile(ttft_values, 95),
        ttft_p99_ms=_percentile(ttft_values, 99),
        latency_p50_ms=_percentile(latency_values, 50),
        latency_p95_ms=_percentile(latency_values, 95),
        latency_p99_ms=_percentile(latency_values, 99),
    )
