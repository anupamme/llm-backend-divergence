"""llama.cpp Q8_0 backend for Qwen2.5-7B-Instruct."""

from __future__ import annotations

from divergence.backends.llamacpp_base import LlamaCppBackend
from divergence.backends.llamacpp_constants import GGUF_Q8_0


class LlamaCppQ8Backend(LlamaCppBackend):
    """llama.cpp backend using Q8_0 quantization."""

    def __init__(self) -> None:
        super().__init__(GGUF_Q8_0, "Q8_0")

    @property
    def name(self) -> str:
        return "llamacpp-q8"
