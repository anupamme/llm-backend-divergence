"""Tests for the multi-backend orchestrator CLI."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

from typer.testing import CliRunner

from divergence.cli.main import app

runner = CliRunner()


class TestListCommands:
    def test_list_backends(self) -> None:
        result = runner.invoke(app, ["list-backends"])
        assert result.exit_code == 0
        assert "mlx-fp16" in result.output
        assert "llamacpp-q8" in result.output
        assert "torch-mps" in result.output
        assert "mock" in result.output

    def test_list_datasets(self) -> None:
        result = runner.invoke(app, ["list-datasets"])
        assert result.exit_code == 0
        assert "gsm8k" in result.output
        assert "mmlu" in result.output
        assert "canary" in result.output


class TestRunCommand:
    def test_smoke_test_mock_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            result = runner.invoke(
                app,
                [
                    "run",
                    "--backends",
                    "mock,mock",
                    "--datasets",
                    "canary",
                    "--output",
                    db_path,
                    "--max-tokens",
                    "8",
                    "--model-id",
                    "test-model",
                ],
            )
            assert result.exit_code == 0, result.output

            conn = sqlite3.connect(db_path)
            # 2 backends × 100 canary items = 200 inference results
            inference_count = conn.execute(
                "SELECT COUNT(*) FROM inference_results"
            ).fetchone()[0]
            assert inference_count == 200

            # Check runs table has 2 entries (mock × canary, twice)
            run_count = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            assert run_count == 2

            # Scoring results also present (MockBackend supports score)
            scoring_count = conn.execute(
                "SELECT COUNT(*) FROM scoring_results"
            ).fetchone()[0]
            assert scoring_count == 200

            conn.close()

    def test_unknown_backend_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            result = runner.invoke(
                app,
                [
                    "run",
                    "--backends",
                    "nonexistent",
                    "--datasets",
                    "canary",
                    "--output",
                    db_path,
                ],
            )
            assert result.exit_code == 1

    def test_unknown_dataset_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            result = runner.invoke(
                app,
                [
                    "run",
                    "--backends",
                    "mock",
                    "--datasets",
                    "nonexistent",
                    "--output",
                    db_path,
                ],
            )
            assert result.exit_code == 1


class TestSummarizeCommand:
    def test_summarize_existing_db(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            # First create a run
            runner.invoke(
                app,
                [
                    "run",
                    "--backends",
                    "mock",
                    "--datasets",
                    "canary",
                    "--output",
                    db_path,
                    "--max-tokens",
                    "8",
                    "--model-id",
                    "test-model",
                ],
            )
            # Then summarize
            result = runner.invoke(app, ["summarize", db_path])
            assert result.exit_code == 0
            assert "mock" in result.output
            assert "canary" in result.output

    def test_summarize_missing_db(self) -> None:
        result = runner.invoke(app, ["summarize", "/nonexistent/path.db"])
        assert result.exit_code == 1
