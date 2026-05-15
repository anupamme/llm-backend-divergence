"""Tests verifying documentation completeness."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class TestReadme:
    def test_exists(self) -> None:
        assert (ROOT / "README.md").is_file()

    def test_has_mermaid_diagram(self) -> None:
        content = (ROOT / "README.md").read_text()
        assert "```mermaid" in content

    def test_has_quickstart(self) -> None:
        content = (ROOT / "README.md").read_text()
        assert "## Quickstart" in content

    def test_has_limitations(self) -> None:
        content = (ROOT / "README.md").read_text()
        assert "## Limitations" in content

    def test_has_license(self) -> None:
        content = (ROOT / "README.md").read_text()
        assert "## License" in content

    def test_has_architecture(self) -> None:
        content = (ROOT / "README.md").read_text()
        assert "## Architecture" in content

    def test_mentions_motivation(self) -> None:
        content = (ROOT / "README.md").read_text()
        assert "## Motivation" in content


class TestFindings:
    def test_exists(self) -> None:
        assert (ROOT / "docs" / "findings.md").is_file()

    def test_has_setup_section(self) -> None:
        content = (ROOT / "docs" / "findings.md").read_text()
        assert "## Setup" in content

    def test_has_headline_numbers(self) -> None:
        content = (ROOT / "docs" / "findings.md").read_text()
        assert "## Headline Numbers" in content

    def test_has_divergence_examples(self) -> None:
        content = (ROOT / "docs" / "findings.md").read_text()
        assert "## Divergence Examples" in content

    def test_has_at_least_three_examples(self) -> None:
        content = (ROOT / "docs" / "findings.md").read_text()
        assert content.count("### Example") >= 3

    def test_has_discussion(self) -> None:
        content = (ROOT / "docs" / "findings.md").read_text()
        assert "## Discussion" in content

    def test_has_implications(self) -> None:
        content = (ROOT / "docs" / "findings.md").read_text()
        assert "## Implications for AI Reliability Engineering" in content

    def test_mentions_hardware(self) -> None:
        content = (ROOT / "docs" / "findings.md").read_text()
        assert "M4" in content

    def test_mentions_model(self) -> None:
        content = (ROOT / "docs" / "findings.md").read_text()
        assert "Qwen2.5-7B-Instruct" in content


class TestMethodology:
    def test_exists(self) -> None:
        assert (ROOT / "docs" / "methodology.md").is_file()

    def test_has_seed_section(self) -> None:
        content = (ROOT / "docs" / "methodology.md").read_text()
        assert "## Seed Management" in content

    def test_has_ttft_section(self) -> None:
        content = (ROOT / "docs" / "methodology.md").read_text()
        assert "## TTFT Measurement" in content

    def test_has_kl_section(self) -> None:
        content = (ROOT / "docs" / "methodology.md").read_text()
        assert "## Why Pairwise KL Divergence" in content

    def test_has_tokenizer_mismatch_section(self) -> None:
        content = (ROOT / "docs" / "methodology.md").read_text()
        assert "## Tokenizer-Mismatch Handling" in content

    def test_explains_kl_formula(self) -> None:
        content = (ROOT / "docs" / "methodology.md").read_text()
        assert "max(0, exp(lp_a" in content

    def test_references_mps_limitations(self) -> None:
        content = (ROOT / "docs" / "methodology.md").read_text()
        assert "torch-mps-limitations.md" in content
