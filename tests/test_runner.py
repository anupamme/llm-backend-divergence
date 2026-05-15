"""Tests for the eval runner."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from divergence.backends import MockBackend
from divergence.evals import EvalItem
from divergence.runner import RunConfig, RunSummary, run_eval


def _make_dataset(n: int = 5) -> list[EvalItem]:
    """Create a synthetic n-item dataset for testing."""
    return [
        EvalItem(
            id=f"test-{i:03d}",
            prompt=f"What is {i} + {i}?",
            reference_answer=str(i + i),
            category="test",
        )
        for i in range(n)
    ]


class TestRunConfig:
    def test_defaults(self) -> None:
        config = RunConfig()
        assert config.max_tokens == 256
        assert config.temperature == 0.0
        assert config.seed == 42
        assert config.output_db_path == "results.db"
        assert config.resume is False
        assert config.score_completion == "model"
        assert config.dataset_name == "unknown"
        assert len(config.run_id) == 36  # UUID format

    def test_custom_values(self) -> None:
        config = RunConfig(
            max_tokens=100,
            temperature=0.5,
            seed=123,
            run_id="custom-id",
            dataset_name="gsm8k",
        )
        assert config.max_tokens == 100
        assert config.run_id == "custom-id"

    def test_auto_generated_run_id_unique(self) -> None:
        c1 = RunConfig()
        c2 = RunConfig()
        assert c1.run_id != c2.run_id


class TestRunSummary:
    def test_construction(self) -> None:
        summary = RunSummary(
            run_id="test-run",
            completed=10,
            errors=2,
            total_wall_time_s=5.0,
            ttft_p50_ms=100.0,
            ttft_p95_ms=200.0,
            ttft_p99_ms=300.0,
            latency_p50_ms=500.0,
            latency_p95_ms=800.0,
            latency_p99_ms=900.0,
        )
        assert summary.completed == 10
        assert summary.errors == 2

    def test_frozen(self) -> None:
        summary = RunSummary(
            run_id="test-run",
            completed=10,
            errors=0,
            total_wall_time_s=1.0,
            ttft_p50_ms=10.0,
            ttft_p95_ms=20.0,
            ttft_p99_ms=30.0,
            latency_p50_ms=50.0,
            latency_p95_ms=80.0,
            latency_p99_ms=90.0,
        )
        with pytest.raises(Exception):
            summary.completed = 99  # type: ignore[misc]


class TestRunEval:
    def test_basic_run(self) -> None:
        backend = MockBackend()
        backend.load("test-model")
        dataset = _make_dataset(5)

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            config = RunConfig(
                output_db_path=db_path,
                max_tokens=16,
                dataset_name="test",
            )
            summary = run_eval(backend, dataset, config)

        assert summary.completed == 5
        assert summary.errors == 0
        assert summary.total_wall_time_s > 0
        assert summary.ttft_p50_ms >= 0
        assert summary.latency_p50_ms >= 0
        backend.unload()

    def test_results_persisted_to_db(self) -> None:
        backend = MockBackend()
        backend.load("test-model")
        dataset = _make_dataset(5)

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            config = RunConfig(
                output_db_path=db_path,
                max_tokens=16,
                run_id="test-run-001",
                dataset_name="test",
            )
            run_eval(backend, dataset, config)

            conn = sqlite3.connect(db_path)
            # Check runs table
            runs = conn.execute("SELECT * FROM runs").fetchall()
            assert len(runs) == 1
            assert runs[0][0] == "test-run-001"

            # Check inference_results table
            inference = conn.execute("SELECT * FROM inference_results").fetchall()
            assert len(inference) == 5

            # Check scoring_results table (MockBackend supports score)
            scoring = conn.execute("SELECT * FROM scoring_results").fetchall()
            assert len(scoring) == 5

            conn.close()
        backend.unload()

    def test_resume_skips_completed_items(self) -> None:
        backend = MockBackend()
        backend.load("test-model")
        dataset = _make_dataset(5)

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            config = RunConfig(
                output_db_path=db_path,
                max_tokens=16,
                run_id="resume-run",
                resume=True,
                dataset_name="test",
            )
            # First run
            summary1 = run_eval(backend, dataset, config)
            assert summary1.completed == 5

            # Second run with resume — should skip all items
            summary2 = run_eval(backend, dataset, config)
            assert summary2.completed == 0
            assert summary2.errors == 0

            # DB should still only have 5 inference results for this run
            conn = sqlite3.connect(db_path)
            inference = conn.execute(
                "SELECT * FROM inference_results WHERE run_id = ?",
                ("resume-run",),
            ).fetchall()
            assert len(inference) == 5
            conn.close()
        backend.unload()

    def test_error_handling(self) -> None:
        backend = MockBackend()
        # Don't call load() — generate will raise RuntimeError
        dataset = _make_dataset(3)

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            config = RunConfig(
                output_db_path=db_path,
                max_tokens=16,
                run_id="error-run",
                dataset_name="test",
            )
            summary = run_eval(backend, dataset, config)

        assert summary.completed == 0
        assert summary.errors == 3
        assert summary.total_wall_time_s >= 0

    def test_error_persisted_to_db(self) -> None:
        backend = MockBackend()
        # Don't call load()
        dataset = _make_dataset(2)

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            config = RunConfig(
                output_db_path=db_path,
                max_tokens=16,
                run_id="error-persist-run",
                dataset_name="test",
            )
            run_eval(backend, dataset, config)

            conn = sqlite3.connect(db_path)
            rows = conn.execute(
                "SELECT finish_reason, error_message FROM inference_results"
            ).fetchall()
            assert len(rows) == 2
            for row in rows:
                assert row[0] == "error"
                assert row[1] is not None
                assert "Model not loaded" in row[1]
            conn.close()

    def test_score_reference_answer(self) -> None:
        backend = MockBackend()
        backend.load("test-model")
        dataset = _make_dataset(3)

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            config = RunConfig(
                output_db_path=db_path,
                max_tokens=16,
                run_id="ref-score-run",
                score_completion="reference",
                dataset_name="test",
            )
            run_eval(backend, dataset, config)

            conn = sqlite3.connect(db_path)
            scoring = conn.execute(
                "SELECT prompt, completion FROM scoring_results"
            ).fetchall()
            assert len(scoring) == 3
            # Verify prompts match our dataset items
            prompts = {row[0] for row in scoring}
            assert "What is 0 + 0?" in prompts
            assert "What is 1 + 1?" in prompts
            assert "What is 2 + 2?" in prompts
            conn.close()
        backend.unload()

    def test_latency_percentiles(self) -> None:
        backend = MockBackend()
        backend.load("test-model")
        dataset = _make_dataset(10)

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            config = RunConfig(
                output_db_path=db_path,
                max_tokens=16,
                dataset_name="test",
            )
            summary = run_eval(backend, dataset, config)

        assert summary.completed == 10
        assert summary.ttft_p50_ms >= 0
        assert summary.ttft_p95_ms >= summary.ttft_p50_ms
        assert summary.ttft_p99_ms >= summary.ttft_p95_ms
        assert summary.latency_p50_ms >= 0
        assert summary.latency_p95_ms >= summary.latency_p50_ms
        assert summary.latency_p99_ms >= summary.latency_p95_ms
        backend.unload()
