"""Tests for eval dataset loaders."""

from __future__ import annotations

from collections import Counter

import pytest
from pydantic import ValidationError

from divergence.evals import EvalItem, load_canary_set, load_gsm8k, load_mmlu_subset


class TestEvalItemModel:
    def test_valid_construction(self) -> None:
        item = EvalItem(
            id="test-001",
            prompt="What is 1+1?",
            reference_answer="2",
            category="math",
        )
        assert item.id == "test-001"
        assert item.precision_dimension is None

    def test_with_precision_dimension(self) -> None:
        item = EvalItem(
            id="test-001",
            prompt="prompt",
            reference_answer="answer",
            category="canary",
            precision_dimension="arithmetic",
        )
        assert item.precision_dimension == "arithmetic"

    def test_frozen(self) -> None:
        item = EvalItem(
            id="test-001",
            prompt="prompt",
            reference_answer="answer",
            category="test",
        )
        with pytest.raises(ValidationError):
            item.id = "changed"  # type: ignore[misc]

    def test_missing_required_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            EvalItem(id="x", prompt="y", category="z")  # type: ignore[call-arg]


class TestCanarySet:
    def test_load_canary_set_returns_100_items(self) -> None:
        items = load_canary_set()
        assert len(items) == 100

    def test_all_items_are_eval_items(self) -> None:
        items = load_canary_set()
        assert all(isinstance(item, EvalItem) for item in items)

    def test_all_items_have_category_canary(self) -> None:
        items = load_canary_set()
        assert all(item.category == "canary" for item in items)

    def test_all_items_have_precision_dimension(self) -> None:
        items = load_canary_set()
        valid_dimensions = {
            "arithmetic",
            "tokenization",
            "logic",
            "long_context",
            "formatting",
        }
        for item in items:
            assert item.precision_dimension in valid_dimensions

    def test_distribution(self) -> None:
        items = load_canary_set()
        counts = Counter(item.precision_dimension for item in items)
        assert counts["arithmetic"] == 40
        assert counts["tokenization"] == 20
        assert counts["logic"] == 20
        assert counts["long_context"] == 10
        assert counts["formatting"] == 10

    def test_unique_ids(self) -> None:
        items = load_canary_set()
        ids = [item.id for item in items]
        assert len(ids) == len(set(ids))

    def test_non_empty_prompts_and_answers(self) -> None:
        items = load_canary_set()
        for item in items:
            assert item.prompt.strip() != ""
            assert item.reference_answer.strip() != ""


@pytest.mark.slow
class TestGsm8k:
    def test_load_default(self) -> None:
        items = load_gsm8k()
        assert len(items) == 200

    def test_load_custom_n(self) -> None:
        items = load_gsm8k(n=10)
        assert len(items) == 10

    def test_items_have_correct_fields(self) -> None:
        items = load_gsm8k(n=5)
        for item in items:
            assert isinstance(item, EvalItem)
            assert item.category == "gsm8k"
            assert item.id.startswith("gsm8k-")
            assert item.prompt.strip() != ""
            assert item.reference_answer.strip() != ""
            assert item.precision_dimension is None


@pytest.mark.slow
class TestMmluSubset:
    def test_load_default(self) -> None:
        items = load_mmlu_subset()
        assert len(items) == 500  # 5 subjects * 100

    def test_load_custom_n(self) -> None:
        items = load_mmlu_subset(n_per_subject=5)
        assert len(items) == 25  # 5 subjects * 5

    def test_items_have_correct_fields(self) -> None:
        items = load_mmlu_subset(n_per_subject=3)
        for item in items:
            assert isinstance(item, EvalItem)
            assert item.category.startswith("mmlu-")
            assert item.id.startswith("mmlu-")
            assert item.reference_answer in ("A", "B", "C", "D")
            assert item.precision_dimension is None

    def test_all_subjects_represented(self) -> None:
        items = load_mmlu_subset(n_per_subject=2)
        categories = {item.category for item in items}
        expected = {
            "mmlu-high_school_mathematics",
            "mmlu-professional_medicine",
            "mmlu-formal_logic",
            "mmlu-college_computer_science",
            "mmlu-moral_scenarios",
        }
        assert categories == expected
