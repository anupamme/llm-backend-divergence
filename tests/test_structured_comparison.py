"""Tests for structured within-framework vs cross-framework comparison."""

from __future__ import annotations

import sqlite3
import tempfile

from divergence.analysis.structured_comparison import (
    StructuredComparisonReport,
    compute_structured_comparison,
)


def _create_test_db(db_path: str) -> None:
    """Create a DB where within-framework backends agree, cross-framework diverge."""
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE runs (
            run_id TEXT PRIMARY KEY,
            backend_name TEXT NOT NULL,
            model_id TEXT NOT NULL,
            dataset_name TEXT NOT NULL,
            started_at TEXT NOT NULL,
            config_json TEXT NOT NULL
        );
        CREATE TABLE inference_results (
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
            error_message TEXT
        );
    """)

    backends_answers = {
        "mlx-fp16": "The answer is B",
        "mlx-q4": "The answer is B",
        "llamacpp-q8": "The answer is C",
        "llamacpp-q4km": "The answer is C",
        "torch-mps": "The answer is B",
    }

    for i, (backend, answer) in enumerate(backends_answers.items()):
        run_id = f"run-{backend}"
        conn.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, backend, "test-model", "mmlu", "2024-01-01", "{}"),
        )
        conn.execute(
            """INSERT INTO inference_results
            (run_id, item_id, prompt, completion, backend_name, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (run_id, "item-001", "What is X?", answer, backend, f"2024-01-01T00:0{i}"),
        )

    conn.commit()
    conn.close()


class TestStructuredComparison:
    def test_within_framework_agreement(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            _create_test_db(f.name)
            report = compute_structured_comparison(f.name, "mmlu")

            assert report.within_framework.agree == 2
            assert report.within_framework.total == 2
            assert report.within_framework.rate == 1.0

    def test_cross_framework_matched_agreement(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            _create_test_db(f.name)
            report = compute_structured_comparison(f.name, "mmlu")

            assert report.cross_framework_matched.agree == 1
            assert report.cross_framework_matched.total == 1

    def test_cross_framework_confounded(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            _create_test_db(f.name)
            report = compute_structured_comparison(f.name, "mmlu")

            assert report.cross_framework_confounded.total == 7
            # mlx-fp16 vs llamacpp-q8: B vs C = disagree
            # mlx-fp16 vs llamacpp-q4km: B vs C = disagree
            # mlx-q4 vs llamacpp-q8: B vs C = disagree
            # mlx-q4 vs llamacpp-q4km: B vs C = disagree
            # torch-mps vs llamacpp-q8: B vs C = disagree
            # torch-mps vs llamacpp-q4km: B vs C = disagree
            # mlx-q4 vs torch-mps: B vs B = agree
            assert report.cross_framework_confounded.agree == 1

    def test_report_structure(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            _create_test_db(f.name)
            report = compute_structured_comparison(f.name, "mmlu")

            assert isinstance(report, StructuredComparisonReport)
            assert report.dataset_name == "mmlu"

    def test_empty_dataset(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            _create_test_db(f.name)
            report = compute_structured_comparison(f.name, "nonexistent")

            assert report.within_framework.total == 0
            assert report.within_framework.rate == 0.0
