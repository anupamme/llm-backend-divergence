"""SQLite persistence for eval run results."""

from __future__ import annotations

import json
import sqlite3

from divergence.backends.schema import InferenceResult, ScoringResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    backend_name TEXT NOT NULL,
    model_id TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    started_at TEXT NOT NULL,
    config_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS inference_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    prompt TEXT NOT NULL,
    completion TEXT,
    token_ids_json TEXT,
    per_token_latency_ms_json TEXT,
    ttft_ms REAL,
    total_latency_ms REAL,
    finish_reason TEXT,
    backend_name TEXT,
    model_id TEXT,
    seed INTEGER,
    temperature REAL,
    timestamp TEXT,
    error_message TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS scoring_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    prompt TEXT NOT NULL,
    completion TEXT,
    token_ids_json TEXT,
    logprobs_json TEXT,
    backend_name TEXT,
    model_id TEXT,
    error_message TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
"""


def init_db(db_path: str) -> sqlite3.Connection:
    """Create or open the SQLite database and ensure schema exists."""
    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def insert_run(
    conn: sqlite3.Connection,
    run_id: str,
    backend_name: str,
    model_id: str,
    dataset_name: str,
    started_at: str,
    config_json: str,
) -> None:
    """Insert a run metadata row (skips if run_id already exists for resume)."""
    conn.execute(
        "INSERT OR IGNORE INTO runs (run_id, backend_name, model_id, dataset_name, "
        "started_at, config_json) VALUES (?, ?, ?, ?, ?, ?)",
        (run_id, backend_name, model_id, dataset_name, started_at, config_json),
    )
    conn.commit()


def insert_inference_result(
    conn: sqlite3.Connection,
    run_id: str,
    item_id: str,
    result: InferenceResult | None,
    error_message: str | None = None,
) -> None:
    """Insert an inference result row."""
    if result is not None:
        conn.execute(
            "INSERT INTO inference_results (run_id, item_id, prompt, completion, "
            "token_ids_json, per_token_latency_ms_json, ttft_ms, total_latency_ms, "
            "finish_reason, backend_name, model_id, seed, temperature, timestamp, "
            "error_message) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                item_id,
                result.prompt,
                result.completion,
                json.dumps(result.token_ids),
                json.dumps(result.per_token_latency_ms),
                result.ttft_ms,
                result.total_latency_ms,
                result.finish_reason,
                result.backend_name,
                result.model_id,
                result.seed,
                result.temperature,
                result.timestamp.isoformat(),
                error_message,
            ),
        )
    else:
        conn.execute(
            "INSERT INTO inference_results (run_id, item_id, prompt, "
            "finish_reason, error_message) VALUES (?, ?, ?, ?, ?)",
            (run_id, item_id, "", "error", error_message),
        )
    conn.commit()


def insert_scoring_result(
    conn: sqlite3.Connection,
    run_id: str,
    item_id: str,
    result: ScoringResult | None,
    error_message: str | None = None,
) -> None:
    """Insert a scoring result row."""
    if result is not None:
        conn.execute(
            "INSERT INTO scoring_results (run_id, item_id, prompt, completion, "
            "token_ids_json, logprobs_json, backend_name, model_id, error_message) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                item_id,
                result.prompt,
                result.completion,
                json.dumps(result.token_ids),
                json.dumps(result.logprobs),
                result.backend_name,
                result.model_id,
                error_message,
            ),
        )
    else:
        conn.execute(
            "INSERT INTO scoring_results (run_id, item_id, prompt, "
            "error_message) VALUES (?, ?, ?, ?)",
            (run_id, item_id, "", error_message),
        )
    conn.commit()


def get_completed_item_ids(conn: sqlite3.Connection, run_id: str) -> set[str]:
    """Get item IDs already completed for a given run (for resume support)."""
    cursor = conn.execute(
        "SELECT item_id FROM inference_results WHERE run_id = ?",
        (run_id,),
    )
    return {row[0] for row in cursor.fetchall()}
