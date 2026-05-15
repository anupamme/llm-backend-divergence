"""Pydantic models for backend inference and scoring results."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator


class Hardware(StrEnum):
    """Supported hardware accelerators."""

    MPS = "mps"
    METAL = "metal"
    CPU = "cpu"


class BackendMetadata(BaseModel):
    """Metadata describing a backend's configuration."""

    model_config = ConfigDict(frozen=True)

    name: str
    framework_version: str
    quantization: str | None
    dtype: str
    hardware: Hardware
    extra: dict[str, str | int | float | bool | None]


class InferenceResult(BaseModel):
    """Result of a single inference (generation) call."""

    model_config = ConfigDict(frozen=True)

    prompt: str
    completion: str
    token_ids: list[int]
    per_token_latency_ms: list[float]
    ttft_ms: float
    total_latency_ms: float
    finish_reason: Literal["stop", "length", "error"]
    backend_name: str
    model_id: str
    seed: int
    temperature: float
    timestamp: datetime

    @model_validator(mode="after")
    def _validate_token_completion_consistency(self) -> InferenceResult:
        if not self.token_ids and self.completion:
            msg = "token_ids must not be empty when completion is non-empty"
            raise ValueError(msg)
        if self.token_ids and not self.completion:
            msg = "completion must not be empty when token_ids is non-empty"
            raise ValueError(msg)
        if len(self.per_token_latency_ms) != len(self.token_ids):
            msg = "per_token_latency_ms length must match token_ids length"
            raise ValueError(msg)
        return self


class ScoringResult(BaseModel):
    """Result of scoring (log-probability evaluation) a completion."""

    model_config = ConfigDict(frozen=True)

    prompt: str
    completion: str
    token_ids: list[int]
    logprobs: list[float]
    backend_name: str
    model_id: str

    @model_validator(mode="after")
    def _validate_token_completion_consistency(self) -> ScoringResult:
        if not self.token_ids and self.completion:
            msg = "token_ids must not be empty when completion is non-empty"
            raise ValueError(msg)
        if len(self.logprobs) != len(self.token_ids):
            msg = "logprobs length must match token_ids length"
            raise ValueError(msg)
        return self
