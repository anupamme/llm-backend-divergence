# Measurement Methodology

This document describes the measurement decisions behind the divergence evaluation framework: how chat templates are applied, how seeds are managed for reproducibility, how latency is measured, why pairwise KL divergence was chosen for logprob analysis, and how tokenizer mismatches are handled.

## Chat Template Formatting

### Why It Matters

Instruct-tuned models (like Qwen2.5-7B-Instruct) are trained to expect structured input with role markers — `<|im_start|>user`, `<|im_end|>`, `<|im_start|>assistant`. Without these markers, the model treats raw text as a base-model continuation prompt and generates based on memorized training data patterns rather than following instructions.

### Implementation

The pipeline applies the model's chat template via `divergence/prompt_format.py`:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_id)
messages = [{"role": "user", "content": raw_prompt}]
formatted = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
```

For Qwen2.5-Instruct, this produces:
```
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
{raw_prompt}<|im_end|>
<|im_start|>assistant
```

### Design Decisions

1. **Applied in the runner, not per-backend.** Template formatting is centralized in `run_eval()` before calling `backend.generate()`. This ensures all backends receive byte-identical input strings, preserving the key experimental invariant.

2. **`--no-chat-template` flag for backward compatibility.** The CLI accepts `--no-chat-template` to reproduce the initial (template-free) run for comparison purposes.

3. **Graceful fallback.** If the tokenizer cannot be loaded (e.g., mock backends in tests), the raw prompt is passed through unchanged.

## Seed Management

Deterministic output requires controlling randomness at the backend level. Each backend receives the same seed value (default: 42) for every prompt, set immediately before generation.

### Per-Backend Strategy

**MLX (FP16 and Q4):**
```python
mx.random.seed(seed)  # Set before each generate() call
```
MLX's random state is global. Setting it before each call ensures identical sampling decisions across repeated runs.

**llama.cpp (Q8 and Q4_K_M):**
```python
self._model.set_seed(seed)  # Set before each generate() call
```
llama-cpp-python exposes seed via the model's `set_seed()` method. This seeds the internal sampling RNG. With temperature=0 (greedy), the seed is redundant but set regardless for consistency.

**PyTorch MPS:**
```python
torch.manual_seed(seed)      # CPU RNG (affects some operations)
torch.mps.manual_seed(seed)  # MPS device RNG
```
Both CPU and device seeds are set because HuggingFace `generate()` may use CPU-side operations during sampling. However, MPS does not guarantee bitwise reproducibility even with fixed seeds — see [PyTorch MPS Limitations](torch-mps-limitations.md). This non-determinism is itself a finding: MPS-based serving cannot be relied upon for exact reproducibility.

### Why Per-Prompt Seeding

The seed is reset before every prompt rather than once per run. This ensures that:
1. Each prompt is independently reproducible regardless of run order.
2. Resume (restarting an interrupted run) produces identical results for remaining items.
3. Divergence at prompt N does not cascade into apparent divergence at prompt N+1.

## TTFT Measurement

Time to First Token (TTFT) measures the latency from the start of a generation request to the emission of the first output token. This includes prompt processing (prefill) time.

### Approach

All backends use `time.perf_counter()` for high-resolution monotonic timing:

```python
start = time.perf_counter()
for token in generate_stream(...):
    now = time.perf_counter()
    if first_token:
        ttft_ms = (now - start) * 1000.0
    per_token_latency_ms.append((now - prev) * 1000.0)
    prev = now
total_latency_ms = (time.perf_counter() - start) * 1000.0
```

### MPS Synchronization

PyTorch MPS dispatches operations asynchronously. Without synchronization, timing measurements reflect dispatch time, not execution time. The MPS backend calls `torch.mps.synchronize()` before recording `total_latency_ms`:

```python
torch.mps.synchronize()  # Wait for all MPS operations to complete
total_latency_ms = (time.perf_counter() - start) * 1000.0
```

For TTFT, the streamer callback fires on CPU after each token is produced, so TTFT timing is naturally synchronized (the callback cannot fire until the token is actually computed).

### What TTFT Includes

TTFT includes:
- Tokenization of the input prompt
- KV-cache prefill (processing all prompt tokens)
- First decode step (generating the first output token)

It does not include model loading or warmup. Models are loaded once via `backend.load()` before any generation calls.

## Why Pairwise KL Divergence

### The Problem

We want to quantify how differently two backends behave at the token level. The natural metric is KL divergence between their next-token probability distributions. However, inference APIs (and most backends) only return the logprob of the token that was actually sampled — not the full vocabulary distribution.

### The Approximation

Given two backends A and B that produced the same token sequence, for each token position i with chosen token t:

```
KL_contribution(i) = max(0, exp(lp_a[i]) * (lp_a[i] - lp_b[i]))
```

Where:
- `lp_a[i]` = log-probability of token t under backend A at position i
- `lp_b[i]` = log-probability of token t under backend B at position i
- `exp(lp_a[i])` = probability of token t under backend A (the weighting term)
- `(lp_a[i] - lp_b[i])` = log-ratio contribution

This is the contribution of token t to `KL(A || B)` from the definition `KL(A||B) = sum_x P_A(x) * log(P_A(x)/P_B(x))`, restricted to the single observed token.

### Why This Works

1. **Lower bound on true KL.** Since we only observe one token per position, the true KL (summing over all vocabulary items) is necessarily >= our approximation. A non-zero value here guarantees real divergence exists.

2. **Sufficient for detection.** We don't need exact KL values — we need to rank prompt/backend pairs by divergence severity and set alert thresholds. Relative ordering is preserved even with the single-token approximation.

3. **Most accurate when it matters most.** The approximation is tightest when the chosen token has high probability (logprob close to 0), which is precisely the case for greedy decoding (temperature=0). In greedy mode, the chosen token typically concentrates most of the probability mass.

### Why Not Alternatives

- **Jensen-Shannon divergence**: requires full distributions from both sides. Same observability problem.
- **L1/L2 on logprobs**: treats all differences equally regardless of probability mass. A 0.1 difference at logprob=-0.01 (very confident) is more meaningful than at logprob=-5.0 (very unlikely). KL naturally weights by probability.
- **Exact match + edit distance only**: misses cases where outputs agree but confidence profiles diverge significantly.

## Tokenizer-Mismatch Handling

### The Problem

Logprob comparison requires aligned token sequences: position i in backend A must correspond to the same text span as position i in backend B. If backends use different tokenizers (or different tokenizer versions), the same text can decompose into different token sequences of different lengths.

### Detection

Before computing pairwise KL, the analysis compares `token_ids` lists from each backend's scoring result:

```python
if len(token_ids_a) != len(token_ids_b):
    # Tokenization mismatch — cannot align positions
    report as TokenizationMismatch
    skip KL computation for this pair
```

Length inequality is a sufficient (though not necessary) condition for misalignment. Same-length sequences with different token IDs at some positions could still represent different segmentations, but in practice this only occurs with genuinely different tokenizers.

### Reporting

Tokenization mismatches are:
1. Excluded from KL divergence computation (no meaningful per-position comparison is possible).
2. Reported separately in the analysis output as `TokenizationMismatch` records with both backends' token counts.
3. Counted in the report summary (`n_tokenization_mismatches`).

### When Mismatches Occur

In this project, all backends use the same model (Qwen2.5-7B-Instruct) and therefore the same tokenizer. Mismatches should not occur under normal operation. Their presence indicates:
- A backend loaded the wrong model or tokenizer
- A tokenizer version mismatch between backend libraries
- Corruption in the scoring results

The mismatch detection exists as a safety check and to support future work with heterogeneous model deployments where different backends may serve different model versions.
