"""Analysis subpackage."""

from divergence.analysis.latency_breakdown import (
    BackendLatency,
    LatencyBreakdownReport,
    compute_latency_breakdown,
)
from divergence.analysis.logprob_divergence import (
    LogprobDivergenceConfig,
    LogprobDivergenceReport,
    PairwiseItemDivergence,
    TokenDelta,
    TokenizationMismatch,
    compute_logprob_divergence,
    visualize_token_divergence,
)
from divergence.analysis.output_divergence import (
    DivergenceReport,
    PairwiseMetrics,
    PromptVerdict,
    Verdict,
    compute_output_divergence,
)
from divergence.analysis.structured_comparison import (
    GroupAgreement,
    StructuredComparisonReport,
    compute_structured_comparison,
)

__all__ = [
    "BackendLatency",
    "DivergenceReport",
    "GroupAgreement",
    "LatencyBreakdownReport",
    "LogprobDivergenceConfig",
    "LogprobDivergenceReport",
    "PairwiseItemDivergence",
    "PairwiseMetrics",
    "PromptVerdict",
    "StructuredComparisonReport",
    "TokenDelta",
    "TokenizationMismatch",
    "Verdict",
    "compute_latency_breakdown",
    "compute_logprob_divergence",
    "compute_output_divergence",
    "compute_structured_comparison",
    "visualize_token_divergence",
]
