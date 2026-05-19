"""PyTorch MPS backend using HuggingFace Transformers for FP16 inference."""

from __future__ import annotations

import logging
import os
import time
from datetime import UTC, datetime
from typing import Any, Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria

from divergence.backends.base import Backend
from divergence.backends.schema import (
    BackendMetadata,
    Hardware,
    InferenceResult,
    ScoringResult,
)

logger = logging.getLogger(__name__)


class _TimestampCriteria(StoppingCriteria):
    """Records wall-clock time at each generated token for latency measurement."""

    def __init__(self) -> None:
        self.timestamps: list[float] = []

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs: Any,
    ) -> bool:
        self.timestamps.append(time.perf_counter())
        return False


class TorchMpsBackend(Backend):
    """Backend using PyTorch + HuggingFace Transformers on MPS (FP16)."""

    def __init__(self) -> None:
        self._model: Any = None
        self._tokenizer: Any = None
        self._model_id: str | None = None

    @property
    def name(self) -> str:
        return "torch-mps"

    def load(self, model_id: str) -> None:
        if not torch.backends.mps.is_available():
            msg = "MPS backend is not available on this system."
            raise RuntimeError(msg)

        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
            logger.warning(
                "PYTORCH_ENABLE_MPS_FALLBACK=1 is set. Some operations may "
                "fall back to CPU, which can affect performance and determinism."
            )

        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map=None,
        )
        self._model = model.to("mps")  # type: ignore[arg-type]
        self._model_id = model_id

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        seed: int,
    ) -> InferenceResult:
        if self._model is None or self._tokenizer is None or self._model_id is None:
            msg = "Model not loaded. Call load() first."
            raise RuntimeError(msg)

        torch.manual_seed(seed)
        torch.mps.manual_seed(seed)

        input_ids = self._tokenizer.encode(prompt, return_tensors="pt").to("mps")
        prompt_len: int = input_ids.shape[1]

        timer = _TimestampCriteria()

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "stopping_criteria": [timer],
        }
        if temperature > 0.0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False

        start = time.perf_counter()

        with torch.no_grad():
            output_ids = self._model.generate(input_ids, **gen_kwargs)

        torch.mps.synchronize()
        total_latency_ms = (time.perf_counter() - start) * 1000.0

        generated_ids: list[int] = output_ids[0, prompt_len:].tolist()

        timestamps = timer.timestamps
        if timestamps:
            ttft_ms = (timestamps[0] - start) * 1000.0
            per_token_latency_ms = [(timestamps[0] - start) * 1000.0] + [
                (timestamps[i] - timestamps[i - 1]) * 1000.0
                for i in range(1, len(timestamps))
            ]
        else:
            ttft_ms = total_latency_ms
            per_token_latency_ms = []

        finish_reason: Literal["stop", "length", "error"] = "length"
        eos_id = self._tokenizer.eos_token_id
        if generated_ids and generated_ids[-1] == eos_id:
            finish_reason = "stop"
            generated_ids = generated_ids[:-1]
            if per_token_latency_ms:
                per_token_latency_ms = per_token_latency_ms[:-1]

        # Align per_token_latency_ms with token_ids length
        if len(per_token_latency_ms) != len(generated_ids):
            if generated_ids:
                avg = total_latency_ms / len(generated_ids)
                per_token_latency_ms = [avg] * len(generated_ids)
                per_token_latency_ms[0] = ttft_ms
            else:
                per_token_latency_ms = []

        completion: str = self._tokenizer.decode(
            generated_ids, skip_special_tokens=True
        )

        return InferenceResult(
            prompt=prompt,
            completion=completion,
            token_ids=generated_ids,
            per_token_latency_ms=per_token_latency_ms,
            ttft_ms=ttft_ms,
            total_latency_ms=total_latency_ms,
            finish_reason=finish_reason,
            backend_name=self.name,
            model_id=self._model_id,
            seed=seed,
            temperature=temperature,
            timestamp=datetime.now(tz=UTC),
        )

    def score(self, prompt: str, completion: str) -> ScoringResult:
        if self._model is None or self._tokenizer is None or self._model_id is None:
            msg = "Model not loaded. Call load() first."
            raise RuntimeError(msg)

        prompt_tokens: list[int] = self._tokenizer.encode(prompt)
        completion_tokens: list[int] = self._tokenizer.encode(
            completion, add_special_tokens=False
        )

        full_tokens = prompt_tokens + completion_tokens
        input_ids = torch.tensor([full_tokens], device="mps")

        with torch.no_grad():
            logits = self._model(input_ids).logits

        prompt_len = len(prompt_tokens)
        completion_logits = logits[0, prompt_len - 1 : -1, :]

        log_probs = torch.nn.functional.log_softmax(completion_logits.float(), dim=-1)

        comp_tensor = torch.tensor(completion_tokens, device="mps")
        token_logprobs: list[float] = log_probs[
            torch.arange(len(completion_tokens), device="mps"), comp_tensor
        ].tolist()

        return ScoringResult(
            prompt=prompt,
            completion=completion,
            token_ids=completion_tokens,
            logprobs=token_logprobs,
            backend_name=self.name,
            model_id=self._model_id,
        )

    def unload(self) -> None:
        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None
        self._model_id = None
        torch.mps.empty_cache()

    @property
    def metadata(self) -> BackendMetadata:
        return BackendMetadata(
            name=self.name,
            framework_version=torch.__version__,
            quantization="none",
            dtype="float16",
            hardware=Hardware.MPS,
            extra={"device_map": "mps"},
        )
