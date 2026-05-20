"""Chat template formatting for instruct models."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def _get_tokenizer(model_id: str) -> Any | None:
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(model_id)
    except (OSError, ValueError):
        logger.warning(
            "Could not load tokenizer for '%s'; skipping chat template.",
            model_id,
        )
        return None


def format_prompt(
    raw_prompt: str,
    model_id: str,
    *,
    apply_template: bool = True,
) -> str:
    """Apply the model's chat template to a raw prompt.

    Wraps the prompt in the model's expected conversation format
    (e.g., <|im_start|>user/assistant markers for Qwen2.5).
    Returns the raw prompt unchanged if apply_template=False or
    if the tokenizer cannot be loaded.
    """
    if not apply_template:
        return raw_prompt

    tokenizer = _get_tokenizer(model_id)
    if tokenizer is None:
        return raw_prompt

    messages = [{"role": "user", "content": raw_prompt}]
    formatted: str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return formatted
