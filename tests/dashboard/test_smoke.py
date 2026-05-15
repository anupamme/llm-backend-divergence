"""Smoke tests for the dashboard data-loading helpers."""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

from divergence.dashboard.data_loader import (
    CanaryDimensionStats,
    LatencyStats,
    MmluSubjectStats,
    get_available_backends,
    get_available_datasets,
    load_canary_breakdown,
    load_latency_stats,
    load_mmlu_subject_stats,
)


def _create_empty_db(db_path: str) -> None:
    """Create an empty DB with the correct schema."""
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
        CREATE TABLE scoring_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            item_id TEXT NOT NULL,
            prompt TEXT NOT NULL,
            completion TEXT,
            token_ids_json TEXT,
            logprobs_json TEXT,
            backend_name TEXT,
            model_id TEXT,
            error_message TEXT
        );
    """)
    conn.commit()
    conn.close()


def _create_populated_db(db_path: str) -> None:
    """Create DB with synthetic canary/mmlu data for 2 backends."""
    _create_empty_db(db_path)
    conn = sqlite3.connect(db_path)

    # Runs
    runs = [
        ("run-a", "backend-a", "model-1", "canary", "2024-01-01", "{}"),
        ("run-b", "backend-b", "model-1", "canary", "2024-01-01", "{}"),
        ("run-c", "backend-a", "model-1", "mmlu", "2024-01-01", "{}"),
        ("run-d", "backend-b", "model-1", "mmlu", "2024-01-01", "{}"),
    ]
    conn.executemany("INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?)", runs)

    tids = json.dumps([1, 2, 3])
    per_token = json.dumps([10.0, 5.0, 5.0])

    # Canary items across dimensions
    canary_items = [
        ("canary-arith-001", "What is 1+1?", "2"),
        ("canary-arith-002", "What is 2+2?", "4"),
        ("canary-tok-001", "Tokenize this", "tokens"),
        ("canary-logic-001", "If A then B", "B"),
        ("canary-ctx-001", "Long passage", "summary"),
        ("canary-fmt-001", "Format JSON", "{}"),
    ]

    for item_id, prompt, completion in canary_items:
        # Backend A
        conn.execute(
            "INSERT INTO inference_results "
            "(run_id, item_id, prompt, completion, token_ids_json, "
            "per_token_latency_ms_json, ttft_ms, total_latency_ms, "
            "finish_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("run-a", item_id, prompt, completion, tids, per_token, 15.0, 20.0, "stop"),
        )
        # Backend B — same completion for most, different for one
        comp_b = "different" if item_id == "canary-arith-002" else completion
        conn.execute(
            "INSERT INTO inference_results "
            "(run_id, item_id, prompt, completion, token_ids_json, "
            "per_token_latency_ms_json, ttft_ms, total_latency_ms, "
            "finish_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("run-b", item_id, prompt, comp_b, tids, per_token, 18.0, 25.0, "stop"),
        )

    # MMLU items
    mmlu_items = [
        ("mmlu-physics-001", "Force equals?", "A"),
        ("mmlu-physics-002", "Energy is?", "B"),
        ("mmlu-history-001", "Year of?", "C"),
    ]

    for item_id, prompt, completion in mmlu_items:
        conn.execute(
            "INSERT INTO inference_results "
            "(run_id, item_id, prompt, completion, token_ids_json, "
            "per_token_latency_ms_json, ttft_ms, total_latency_ms, "
            "finish_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("run-c", item_id, prompt, completion, tids, per_token, 12.0, 18.0, "stop"),
        )
        comp_b = "D" if item_id == "mmlu-history-001" else completion
        conn.execute(
            "INSERT INTO inference_results "
            "(run_id, item_id, prompt, completion, token_ids_json, "
            "per_token_latency_ms_json, ttft_ms, total_latency_ms, "
            "finish_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("run-d", item_id, prompt, comp_b, tids, per_token, 14.0, 22.0, "stop"),
        )

    # Add an error row
    conn.execute(
        "INSERT INTO inference_results "
        "(run_id, item_id, prompt, finish_reason, error_message) "
        "VALUES (?, ?, ?, ?, ?)",
        ("run-a", "canary-arith-003", "fail", "error", "timeout"),
    )

    conn.commit()
    conn.close()


class TestDataLoaderEmptyDb:
    def test_empty_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            _create_empty_db(db_path)
            assert get_available_datasets(db_path) == []

    def test_empty_backends(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            _create_empty_db(db_path)
            assert get_available_backends(db_path) == []

    def test_empty_latency_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            _create_empty_db(db_path)
            assert load_latency_stats(db_path) == []

    def test_empty_canary_breakdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            _create_empty_db(db_path)
            assert load_canary_breakdown(db_path) == []

    def test_empty_mmlu_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            _create_empty_db(db_path)
            assert load_mmlu_subject_stats(db_path) == []


class TestDataLoaderWithData:
    def test_available_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            _create_populated_db(db_path)
            datasets = get_available_datasets(db_path)
        assert "canary" in datasets
        assert "mmlu" in datasets

    def test_available_backends(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            _create_populated_db(db_path)
            backends = get_available_backends(db_path)
        assert "backend-a" in backends
        assert "backend-b" in backends

    def test_latency_stats_structure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            _create_populated_db(db_path)
            stats = load_latency_stats(db_path)

        assert len(stats) == 2
        assert all(isinstance(s, LatencyStats) for s in stats)

        # Backend A has canary + mmlu items + 1 error
        ba = next(s for s in stats if s.backend_name == "backend-a")
        assert ba.error_count == 1
        assert ba.total_count > 0
        assert len(ba.ttft_values) > 0
        assert len(ba.itl_values) > 0
        assert ba.total_tokens > 0

    def test_latency_itl_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            _create_populated_db(db_path)
            stats = load_latency_stats(db_path)

        ba = next(s for s in stats if s.backend_name == "backend-a")
        # Each non-error item has per_token=[10.0, 5.0, 5.0]
        assert 5.0 in ba.itl_values
        assert 10.0 in ba.itl_values

    def test_canary_breakdown_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            _create_populated_db(db_path)
            breakdown = load_canary_breakdown(db_path)

        assert len(breakdown) > 0
        assert all(isinstance(s, CanaryDimensionStats) for s in breakdown)
        dimensions = {s.dimension for s in breakdown}
        assert "arithmetic" in dimensions

    def test_canary_disagreement_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            _create_populated_db(db_path)
            breakdown = load_canary_breakdown(db_path)

        # canary-arith-002 has different completions → disagreement
        arith = next((s for s in breakdown if s.dimension == "arithmetic"), None)
        assert arith is not None
        assert arith.disagreements > 0

    def test_mmlu_subject_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            _create_populated_db(db_path)
            stats = load_mmlu_subject_stats(db_path)

        assert len(stats) > 0
        assert all(isinstance(s, MmluSubjectStats) for s in stats)
        subjects = {s.subject for s in stats}
        assert "physics" in subjects
        assert "history" in subjects

    def test_mmlu_disagreement_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            _create_populated_db(db_path)
            stats = load_mmlu_subject_stats(db_path)

        # mmlu-history-001 has different completions
        history = next((s for s in stats if s.subject == "history"), None)
        assert history is not None
        assert history.disagreements > 0


def test_import_app() -> None:
    """Verify app module is importable without side effects."""
    import divergence.dashboard.app  # noqa: F401
