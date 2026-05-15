"""Tests for logprob-level divergence detection."""

from __future__ import annotations

import json
import math
import sqlite3
import tempfile
from pathlib import Path

from divergence.analysis.logprob_divergence import (
    LogprobDivergenceConfig,
    LogprobDivergenceReport,
    TokenDelta,
    _compute_token_deltas,
    _percentile,
    compute_logprob_divergence,
    visualize_token_divergence,
)


def _create_scoring_test_db(db_path: str) -> None:
    """Create a synthetic DB with known logprob divergence patterns."""
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

    runs = [
        ("run-a", "backend-a", "model-1", "test", "2024-01-01", "{}"),
        ("run-b", "backend-b", "model-1", "test", "2024-01-01", "{}"),
        ("run-c", "backend-c", "model-1", "test", "2024-01-01", "{}"),
    ]
    conn.executemany("INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?)", runs)

    tids = json.dumps([1, 2, 3])
    tids_diff = json.dumps([4, 5, 6])
    tids_short = json.dumps([1, 2])
    lp_base = json.dumps([-0.1, -0.2, -0.3])
    lp_small_drift = json.dumps([-0.12, -0.22, -0.28])
    lp_large_drift = json.dumps([-0.5, -1.0, -2.0])
    lp_short = json.dumps([-0.1, -0.2])

    scoring = [
        # item-1: all same → zero divergence
        ("run-a", "item-1", "Q1", "answer1", tids, lp_base, None),
        ("run-b", "item-1", "Q1", "answer1", tids, lp_base, None),
        ("run-c", "item-1", "Q1", "answer1", tids, lp_base, None),
        # item-2: small drift between A and B → below threshold
        ("run-a", "item-2", "Q2", "answer2", tids, lp_base, None),
        ("run-b", "item-2", "Q2", "answer2", tids, lp_small_drift, None),
        # item-3: large drift between A and B → alert
        ("run-a", "item-3", "Q3", "answer3", tids, lp_base, None),
        ("run-b", "item-3", "Q3", "answer3", tids, lp_large_drift, None),
        # item-4: token mismatch between A and C
        ("run-a", "item-4", "Q4", "answer4", tids, lp_base, None),
        ("run-c", "item-4", "Q4", "answer4", tids_diff, lp_base, None),
        # item-5: length mismatch between A and C
        ("run-a", "item-5", "Q5", "answer5", tids, lp_base, None),
        ("run-c", "item-5", "Q5", "answer5", tids_short, lp_short, None),
        # item-6: error row for C (should be skipped)
        ("run-a", "item-6", "Q6", "answer6", tids, lp_base, None),
        ("run-c", "item-6", "Q6", "answer6", tids, lp_base, "score failed"),
    ]

    for s in scoring:
        conn.execute(
            "INSERT INTO scoring_results "
            "(run_id, item_id, prompt, completion, token_ids_json, "
            "logprobs_json, error_message) VALUES (?, ?, ?, ?, ?, ?, ?)",
            s,
        )

    conn.commit()
    conn.close()


class TestPercentile:
    def test_p50(self) -> None:
        assert _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50) == 3.0

    def test_p95(self) -> None:
        values = list(range(1, 101))
        result = _percentile([float(v) for v in values], 95)
        assert abs(result - 95.05) < 0.1

    def test_single_value(self) -> None:
        assert _percentile([5.0], 95) == 5.0

    def test_empty(self) -> None:
        assert _percentile([], 95) == 0.0


class TestComputeTokenDeltas:
    def test_identical_logprobs(self) -> None:
        deltas = _compute_token_deltas(
            [1, 2, 3], [-0.1, -0.2, -0.3], [-0.1, -0.2, -0.3]
        )
        assert len(deltas) == 3
        for d in deltas:
            assert d.delta == 0.0
            assert d.kl_contribution == 0.0

    def test_known_kl_contribution(self) -> None:
        # exp(-0.1) * (−0.1 − (−0.2)) = exp(-0.1) * 0.1
        deltas = _compute_token_deltas([1], [-0.1], [-0.2])
        expected_kl = math.exp(-0.1) * 0.1
        assert abs(deltas[0].kl_contribution - expected_kl) < 1e-10
        assert abs(deltas[0].delta - 0.1) < 1e-10

    def test_negative_kl_clamped_to_zero(self) -> None:
        # When lp_b > lp_a (B more confident), delta < 0 → KL clamped to 0
        deltas = _compute_token_deltas([1], [-0.5], [-0.1])
        assert deltas[0].delta < 0
        assert deltas[0].kl_contribution == 0.0

    def test_multi_token_analytics(self) -> None:
        # Verify analytically:
        # pos 0: exp(-0.1) * (-0.1 - (-0.5)) = 0.9048 * 0.4 = 0.36193
        # pos 1: exp(-0.2) * (-0.2 - (-1.0)) = 0.8187 * 0.8 = 0.65500
        # pos 2: exp(-0.3) * (-0.3 - (-2.0)) = 0.7408 * 1.7 = 1.25934
        deltas = _compute_token_deltas(
            [1, 2, 3], [-0.1, -0.2, -0.3], [-0.5, -1.0, -2.0]
        )
        assert abs(deltas[0].kl_contribution - 0.36193) < 0.001
        assert abs(deltas[1].kl_contribution - 0.65500) < 0.001
        assert abs(deltas[2].kl_contribution - 1.25934) < 0.001

    def test_position_and_token_id(self) -> None:
        lps = [-0.1, -0.2, -0.3]
        deltas = _compute_token_deltas([10, 20, 30], lps, lps)
        assert deltas[0].position == 0
        assert deltas[0].token_id == 10
        assert deltas[2].position == 2
        assert deltas[2].token_id == 30


class TestComputeLogprobDivergence:
    def test_full_report_structure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            _create_scoring_test_db(db_path)

            report = compute_logprob_divergence(db_path)

        assert isinstance(report, LogprobDivergenceReport)
        assert len(report.backends) > 0
        assert report.n_items_analyzed > 0
        assert report.approximation_note != ""
        assert report.config.kl_threshold == 0.1

    def test_zero_divergence_no_alert(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            _create_scoring_test_db(db_path)

            report = compute_logprob_divergence(db_path)

        # item-1 has zero divergence across all pairs
        item1_divs = [d for d in report.divergences if d.item_id == "item-1"]
        for d in item1_divs:
            assert d.mean_kl_contribution == 0.0
            assert d.is_alert is False

    def test_large_divergence_triggers_alert(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            _create_scoring_test_db(db_path)

            report = compute_logprob_divergence(db_path)

        # item-3 has large drift → should be an alert
        item3_alerts = [a for a in report.alerts if a.item_id == "item-3"]
        assert len(item3_alerts) == 1
        assert item3_alerts[0].backend_a == "backend-a"
        assert item3_alerts[0].backend_b == "backend-b"
        assert item3_alerts[0].mean_kl_contribution > 0.1

    def test_tokenization_mismatch_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            _create_scoring_test_db(db_path)

            report = compute_logprob_divergence(db_path)

        # item-4 and item-5 have token mismatches with backend-c
        mismatch_items = {m.item_id for m in report.tokenization_mismatches}
        assert "item-4" in mismatch_items
        assert "item-5" in mismatch_items
        assert report.n_tokenization_mismatches >= 2

    def test_error_rows_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            _create_scoring_test_db(db_path)

            report = compute_logprob_divergence(db_path)

        # item-6 should not appear as a divergence (C has error)
        item6_divs = [d for d in report.divergences if d.item_id == "item-6"]
        assert len(item6_divs) == 0
        item6_mismatches = [
            m for m in report.tokenization_mismatches if m.item_id == "item-6"
        ]
        assert len(item6_mismatches) == 0

    def test_custom_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            _create_scoring_test_db(db_path)

            # Very high threshold → no alerts
            cfg = LogprobDivergenceConfig(kl_threshold=100.0)
            report = compute_logprob_divergence(db_path, config=cfg)

        assert report.n_items_with_alerts == 0

    def test_top_n_limits_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            _create_scoring_test_db(db_path)

            cfg = LogprobDivergenceConfig(top_n=2)
            report = compute_logprob_divergence(db_path, config=cfg)

        assert len(report.divergences) <= 2

    def test_single_backend_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
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
            conn.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?)",
                ("run-a", "backend-a", "m", "t", "2024-01-01", "{}"),
            )
            conn.execute(
                "INSERT INTO scoring_results "
                "(run_id, item_id, prompt, completion, "
                "token_ids_json, logprobs_json, error_message) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("run-a", "i1", "Q", "A", "[1]", "[-0.5]", None),
            )
            conn.commit()
            conn.close()

            report = compute_logprob_divergence(db_path)

        assert report.n_items_analyzed == 0
        assert report.divergences == []

    def test_markdown_summary_nonempty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            _create_scoring_test_db(db_path)

            report = compute_logprob_divergence(db_path)

        assert "Logprob Divergence Report" in report.summary_markdown
        assert len(report.summary_markdown) > 50

    def test_sorted_by_mean_kl_descending(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            _create_scoring_test_db(db_path)

            report = compute_logprob_divergence(db_path)

        if len(report.divergences) >= 2:
            for i in range(len(report.divergences) - 1):
                assert (
                    report.divergences[i].mean_kl_contribution
                    >= report.divergences[i + 1].mean_kl_contribution
                )


class TestVisualizeTokenDivergence:
    def test_returns_token_deltas(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            _create_scoring_test_db(db_path)

            deltas = visualize_token_divergence(
                db_path, "item-3", "backend-a", "backend-b"
            )

        assert len(deltas) == 3
        assert all(isinstance(d, TokenDelta) for d in deltas)
        # Verify values match expected large drift
        assert abs(deltas[0].logprob_a - (-0.1)) < 1e-10
        assert abs(deltas[0].logprob_b - (-0.5)) < 1e-10

    def test_mismatch_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            _create_scoring_test_db(db_path)

            deltas = visualize_token_divergence(
                db_path, "item-4", "backend-a", "backend-c"
            )

        assert deltas == []

    def test_nonexistent_item_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            _create_scoring_test_db(db_path)

            deltas = visualize_token_divergence(
                db_path, "no-such-item", "backend-a", "backend-b"
            )

        assert deltas == []

    def test_missing_backend_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            _create_scoring_test_db(db_path)

            deltas = visualize_token_divergence(
                db_path, "item-1", "backend-a", "nonexistent"
            )

        assert deltas == []
