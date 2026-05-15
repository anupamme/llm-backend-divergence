"""Abstract base class for LLM backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

from divergence.backends.schema import (
    BackendMetadata,
    InferenceResult,
    ScoringResult,
)


class Backend(ABC):
    """Abstract base class that all LLM backends must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a stable string identifier for this backend."""

    @abstractmethod
    def load(self, model_id: str) -> None:
        """Load model weights into memory."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        seed: int,
    ) -> InferenceResult:
        """Generate a completion for the given prompt."""

    def score(self, prompt: str, completion: str) -> ScoringResult:
        """Score a completion given a prompt (compute log-probabilities).

        Default implementation raises NotImplementedError. Backends that
        support logprob extraction should override this method.
        """
        raise NotImplementedError(f"Backend '{self.name}' does not support scoring")

    @abstractmethod
    def unload(self) -> None:
        """Release model weights and free memory."""

    @property
    @abstractmethod
    def metadata(self) -> BackendMetadata:
        """Return metadata describing this backend's configuration."""
