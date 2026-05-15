"""Tests for output-level divergence detection."""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

from divergence.analysis.output_divergence import (
    DivergenceReport,
    _classify_verdict,
    _extract_answer,
    _levenshtein,
    compute_output_divergence,
)


def _create_test_db(db_path: str) -> None:
    """Create a synthetic DB with known divergence patterns."""
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

    # 3 backends
    runs = [
        ("run-a", "backend-a", "model-1", "canary", "2024-01-01", "{}"),
        ("run-b", "backend-b", "model-1", "canary", "2024-01-01", "{}"),
        ("run-c", "backend-c", "model-1", "canary", "2024-01-01", "{}"),
    ]
    conn.executemany("INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?)", runs)

    tok_same = json.dumps([1, 2, 3])
    tok_diff = json.dumps([4, 5, 6])
    tok_c = json.dumps([7, 8, 9])
    tok_hello = json.dumps([10, 20, 30])
    tok_hi = json.dumps([40, 50])

    results = [
        # Item 1: all same → unanimous
        ("run-a", "item-1", "What is 1+1?", "The answer is 2", tok_same, "stop"),
        ("run-b", "item-1", "What is 1+1?", "The answer is 2", tok_same, "stop"),
        ("run-c", "item-1", "What is 1+1?", "The answer is 2", tok_same, "stop"),
        # Item 2: A and B agree, C differs → majority
        ("run-a", "item-2", "What is 2+2?", "The answer is 4", tok_same, "stop"),
        ("run-b", "item-2", "What is 2+2?", "The answer is 4", tok_same, "stop"),
        ("run-c", "item-2", "What is 2+2?", "The answer is 5", tok_diff, "stop"),
        # Item 3: all different → dispersed
        ("run-a", "item-3", "Tell a joke", "Chicken crossed", tok_same, "stop"),
        ("run-b", "item-3", "Tell a joke", "Knock knock", tok_diff, "stop"),
        ("run-c", "item-3", "Tell a joke", "Priest bar", tok_c, "stop"),
        # Item 4: A=B, C different (pairwise exact match)
        ("run-a", "item-4", "Say hello", "Hello world!", tok_hello, "stop"),
        ("run-b", "item-4", "Say hello", "Hello world!", tok_hello, "stop"),
        ("run-c", "item-4", "Say hello", "Hi there!", tok_hi, "stop"),
        # Item 5: one backend has error (excluded for that backend)
        ("run-a", "item-5", "Count to 3", "1 2 3", tok_same, "stop"),
        ("run-b", "item-5", "Count to 3", "1 2 3", tok_same, "stop"),
        ("run-c", "item-5", "Count to 3", None, None, "error"),
    ]

    for r in results:
        conn.execute(
            "INSERT INTO inference_results "
            "(run_id, item_id, prompt, completion, token_ids_json, finish_reason) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            r,
        )

    conn.commit()
    conn.close()


class TestLevenshtein:
    def test_identical(self) -> None:
        assert _levenshtein("hello", "hello") == 0

    def test_empty(self) -> None:
        assert _levenshtein("", "abc") == 3
        assert _levenshtein("abc", "") == 3

    def test_single_edit(self) -> None:
        assert _levenshtein("cat", "bat") == 1
        assert _levenshtein("cat", "cats") == 1
        assert _levenshtein("cats", "cat") == 1

    def test_multi_edit(self) -> None:
        assert _levenshtein("kitten", "sitting") == 3


class TestExtractAnswer:
    def test_gsm8k_with_hash(self) -> None:
        text = "Let me calculate... 5 + 3 = 8. #### 8"
        assert _extract_answer(text, "gsm8k") == "8"

    def test_gsm8k_without_hash(self) -> None:
        text = "The total is 42"
        assert _extract_answer(text, "gsm8k") == "42"

    def test_arithmetic_category(self) -> None:
        text = "5 + 5 = 10"
        assert _extract_answer(text, "arithmetic") == "10"

    def test_mmlu_letter(self) -> None:
        text = "The answer is B because..."
        assert _extract_answer(text, "mmlu") == "B"

    def test_general_category(self) -> None:
        text = "  Hello world  "
        assert _extract_answer(text, "canary") == "Hello world"


class TestClassifyVerdict:
    def test_unanimous(self) -> None:
        answers = {"a": "42", "b": "42", "c": "42"}
        assert _classify_verdict(answers) == "unanimous"

    def test_majority(self) -> None:
        answers = {"a": "42", "b": "42", "c": "43"}
        assert _classify_verdict(answers) == "majority"

    def test_split(self) -> None:
        answers = {"a": "42", "b": "42", "c": "42", "d": "43", "e": "44"}
        assert _classify_verdict(answers) == "split"

    def test_dispersed(self) -> None:
        answers = {"a": "1", "b": "2", "c": "3"}
        assert _classify_verdict(answers) == "dispersed"

    def test_single_backend(self) -> None:
        answers = {"a": "42"}
        assert _classify_verdict(answers) == "unanimous"


class TestComputeOutputDivergence:
    def test_full_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            _create_test_db(db_path)

            report = compute_output_divergence(db_path, "canary")

        assert isinstance(report, DivergenceReport)
        assert report.dataset_name == "canary"
        assert sorted(report.backends) == ["backend-a", "backend-b", "backend-c"]
        assert report.n_prompts == 5

    def test_verdicts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            _create_test_db(db_path)

            report = compute_output_divergence(db_path, "canary")

        verdicts_by_item = {v.item_id: v.verdict for v in report.prompt_verdicts}
        assert verdicts_by_item["item-1"] == "unanimous"
        assert verdicts_by_item["item-2"] == "majority"
        assert verdicts_by_item["item-3"] == "dispersed"
        assert verdicts_by_item["item-4"] == "majority"
        # Item 5: only A and B have results (C is error), both agree
        assert verdicts_by_item["item-5"] == "unanimous"

    def test_pairwise_exact_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            _create_test_db(db_path)

            report = compute_output_divergence(db_path, "canary")

        # Find A-B pair
        ab_metrics = None
        for m in report.pairwise_metrics:
            if m.backend_a == "backend-a" and m.backend_b == "backend-b":
                ab_metrics = m
                break

        assert ab_metrics is not None
        # A-B: 5 items present. 1,2,4,5=match, 3=diff → 4/5
        assert ab_metrics.exact_match_rate == 4 / 5

    def test_pairwise_levenshtein(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            _create_test_db(db_path)

            report = compute_output_divergence(db_path, "canary")

        # A-B pair: items 1,2,4,5 have same text → lev=0. Item 3 differs.
        ab_metrics = None
        for m in report.pairwise_metrics:
            if m.backend_a == "backend-a" and m.backend_b == "backend-b":
                ab_metrics = m
                break

        assert ab_metrics is not None
        assert ab_metrics.avg_levenshtein > 0  # item 3 contributes

    def test_bleu_scores_in_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            _create_test_db(db_path)

            report = compute_output_divergence(db_path, "canary")

        for m in report.pairwise_metrics:
            assert 0.0 <= m.avg_bleu <= 100.0

    def test_markdown_summary_nonempty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            _create_test_db(db_path)

            report = compute_output_divergence(db_path, "canary")

        assert len(report.summary_markdown) > 0
        assert "Divergence Report" in report.summary_markdown
        assert "canary" in report.summary_markdown
        assert "unanimous" in report.summary_markdown

    def test_error_rows_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            _create_test_db(db_path)

            report = compute_output_divergence(db_path, "canary")

        # Item 5: backend-c has error, so should only have A and B completions
        item5 = None
        for v in report.prompt_verdicts:
            if v.item_id == "item-5":
                item5 = v
                break

        assert item5 is not None
        assert "backend-c" not in item5.completions
        assert len(item5.completions) == 2

    def test_nonexistent_dataset_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            _create_test_db(db_path)

            report = compute_output_divergence(db_path, "nonexistent")

        assert report.n_prompts == 0
        assert report.pairwise_metrics == []
        assert report.prompt_verdicts == []
