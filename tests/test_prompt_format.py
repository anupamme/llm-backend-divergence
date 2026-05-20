"""Tests for chat template formatting."""

from __future__ import annotations

from divergence.prompt_format import format_prompt

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


class TestFormatPrompt:
    def test_applies_chat_template(self) -> None:
        result = format_prompt("What is 2+2?", MODEL_ID)
        assert "<|im_start|>user" in result
        assert "What is 2+2?" in result
        assert "<|im_start|>assistant" in result

    def test_includes_generation_prompt(self) -> None:
        result = format_prompt("Hello", MODEL_ID)
        assert result.rstrip().endswith("<|im_start|>assistant")

    def test_no_template_returns_raw(self) -> None:
        raw = "Just a plain prompt"
        result = format_prompt(raw, MODEL_ID, apply_template=False)
        assert result == raw

    def test_preserves_prompt_content(self) -> None:
        prompt = "Decode this Caesar cipher: 'Khoor Zruog'"
        result = format_prompt(prompt, MODEL_ID)
        assert prompt in result

    def test_system_message_present(self) -> None:
        result = format_prompt("test", MODEL_ID)
        assert "<|im_start|>system" in result
