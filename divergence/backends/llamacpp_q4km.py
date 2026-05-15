"""llama.cpp Q4_K_M backend for Qwen2.5-7B-Instruct."""

from __future__ import annotations

from divergence.backends.llamacpp_base import LlamaCppBackend
from divergence.backends.llamacpp_constants import GGUF_Q4_K_M


class LlamaCppQ4KMBackend(LlamaCppBackend):
    """llama.cpp backend using Q4_K_M quantization."""

    def __init__(self) -> None:
        super().__init__(GGUF_Q4_K_M, "Q4_K_M")

    @property
    def name(self) -> str:
        return "llamacpp-q4km"
