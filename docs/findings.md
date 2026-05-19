# First Run Findings

Results from the initial full evaluation run comparing four inference backends on Apple Silicon.

## Setup

| Parameter | Value |
|-----------|-------|
| Hardware | Apple M4 Pro, 24 GB unified memory |
| Model | Qwen/Qwen2.5-7B-Instruct |
| Temperature | 0.0 (greedy decoding) |
| Seed | 42 |
| Max tokens | 256 |

**Backends:**
- `mlx-fp16` — MLX framework, FP16 weights
- `mlx-q4` — MLX framework, 4-bit quantized
- `llamacpp-q8` — llama.cpp, Q8_0 quantization (Metal)
- `llamacpp-q4km` — llama.cpp, Q4_K_M quantization (Metal)

Note: `torch-mps` (PyTorch FP16 on MPS) could not complete the run due to a 14.19 GiB buffer allocation failure on the 24 GB system. This is itself a finding — MPS FP16 inference for 7B models requires more contiguous memory than a 24 GB Apple Silicon machine can reliably provide under normal system load.

**Datasets:**
- GSM8K: 200 grade-school math problems
- MMLU: 500 items across 5 subjects (high school math, medicine, formal logic, computer science, moral scenarios)
- Canary: 100 hand-crafted precision probes (arithmetic, tokenization, logic, long context, formatting)

## Headline Numbers

### Agreement Rates by Dataset

| Dataset | Unanimous | Majority | Split | Dispersed |
|---------|-----------|----------|-------|-----------|
| GSM8K | 42.5% | 26.0% | 0.0% | 31.5% |
| MMLU | 60.2% | 10.0% | 0.0% | 29.8% |
| Canary | 4.0% | 4.0% | 0.0% | 92.0% |

The canary dataset — designed to probe precision-sensitive operations — shows extreme divergence (92% dispersed). This is driven by long-form generation where backends produce semantically similar but textually different responses. MMLU shows the highest agreement at 60.2% unanimous, reflecting constrained factual recall tasks. The absence of "split" verdicts (0% across all datasets) indicates that with 4 backends, disagreements tend to be either minor (majority) or total (dispersed) rather than evenly divided.

### Latency by Backend

| Backend | TTFT p50 (ms) | TTFT p95 (ms) | Total p50 (ms) | Total p95 (ms) |
|---------|---------------|---------------|----------------|----------------|
| mlx-fp16 | 816 | 1,316 | 36,243 | 37,039 |
| mlx-q4 | 792 | 1,272 | 36,196 | 36,969 |
| llamacpp-q8 | 475 | 1,120 | 32,726 | 35,251 |
| llamacpp-q4km | 495 | 1,234 | 24,791 | 27,522 |

llama.cpp Q4_K_M is fastest for total latency (24.8s p50 vs 36.2s for MLX FP16) — a 31% speedup from aggressive quantization. llama.cpp also has lower TTFT than MLX, suggesting faster prompt processing (prefill) in the Metal-accelerated llama.cpp stack. MLX FP16 and Q4 are nearly identical in latency, indicating that MLX's quantization overhead is minimal but doesn't translate to meaningful speedup for generation-bound workloads at max_tokens=256.

## Divergence Examples

### Example 1: Hallucinated Web Content (Canary)

**Prompt:** `What is 2^32?`

| Backend | Completion (first 100 chars) |
|---------|------------------------------|
| mlx-fp16 | `- Quora\nWhat is 2^32?\nAd by The Penny Hoarder\nShould you leave more than $1,000 in a checking acc...` |
| mlx-q4 | `- Quora\nWhat is 2^32?\nAd by The Penny Hoarder\nShould you leave more than $1,000 in a checking acc...` |
| llamacpp-q8 | `- Quora\nWhat is 2^32?\nAd by Masterworks\nWhat's a good investment for 2022?\nThis might sound unc...` |
| llamacpp-q4km | `- Quora\nWhat is 2^32?\nAd by Masterworks\nWhat's a good investment for 2022?\nThis might sound unc...` |

**Verdict:** dispersed

**What changed:** All backends hallucinate a Quora-style web page rather than answering "4,294,967,296". The divergence is in the *content* of the hallucination: MLX backends hallucinate a "Penny Hoarder" ad, while llama.cpp backends hallucinate a "Masterworks" ad. This reveals that quantization framework (not just quantization level) influences which memorized training data surfaces. Both FP16 and Q4 within the same framework agree, but cross-framework comparison diverges — suggesting the divergence is driven by subtle differences in how MLX vs llama.cpp handle floating-point accumulation during attention computation.

### Example 2: Caesar Cipher Decoding Strategy (Canary)

**Prompt:** `The following is an encoded message using a simple Caesar cipher with shift 3: 'Wkh txlfn eurzq ira...'`

| Backend | Completion (first 100 chars) |
|---------|------------------------------|
| mlx-fp16 | `To decode the message, we need to reverse the Caesar cipher with a shift of 3. This means we will sh...` |
| mlx-q4 | `To decode the message, we need to reverse the Caesar cipher with a shift of 3. This means we will sh...` |
| llamacpp-q8 | `To decode the message, we need to reverse the Caesar cipher by shifting each letter 3 positions bac...` |
| llamacpp-q4km | `To decode the message, we need to reverse the Caesar cipher with a shift of 3. This means we will s...` |

**Verdict:** majority (3 backends agree, llamacpp-q8 diverges on phrasing)

**What changed:** Three backends use "reverse the Caesar cipher with a shift of 3", while llamacpp-q8 phrases it as "shifting each letter 3 positions back". The answer is semantically identical but textually different. This illustrates how the output divergence detector correctly classifies this as "majority" — most backends agree, one deviates. In production, this kind of benign paraphrase divergence is noise, not signal.

### Example 3: MMLU Math Problem — Reasoning Chain (MMLU)

**Prompt:** `Jane's quiz scores were 98, 97, 92, 85 and 93. What was her mean score? A. 92 B. 93 C. 94.5 D. 95`

| Backend | Completion (first 90 chars) |
|---------|------------------------------|
| mlx-fp16 | `To find Jane's mean quiz score, we need to follow these steps:\n\n1. **Sum the sco...` |
| mlx-q4 | `To find Jane's mean quiz score, we need to follow these steps:\n\n1. **Sum the sco...` |
| llamacpp-q8 | `To find Jane's mean quiz score, we need to follow these steps:\n\n1. **Sum the sc...` |
| llamacpp-q4km | `To find Jane's mean score, we need to follow these steps:\n\n1. **Sum the scores*...` |

**Verdict:** majority

**What changed:** llamacpp-q4km says "mean score" where the other three say "mean quiz score", and uses slightly different formatting in the step labels. All arrive at the correct answer (B. 93). The Q4_K_M quantization produces the most abbreviated phrasing — a pattern consistent with reduced precision causing the model to assign slightly different probabilities to filler words like "quiz".

## Discussion

### What to Alert On

Based on these findings, a production divergence monitoring system should alert on:

1. **Cross-framework disagreement on factual answers.** When MLX and llama.cpp backends give different extracted answers (not just different phrasing), this indicates a meaningful behavioral divergence that could affect users.
2. **Hallucination divergence.** The "2^32" example shows that different backends hallucinate different content. If one backend starts hallucinating where others don't, that's a quality regression.
3. **Canary disagreement on deterministic tasks.** For prompts with unambiguous correct answers (arithmetic, logic), any disagreement signals a problem.

### Proposed SLOs

| Metric | Warning | Critical |
|--------|---------|----------|
| MMLU answer disagreement (extracted) | > 15% | > 25% |
| GSM8K answer disagreement (extracted) | > 20% | > 35% |
| Canary unanimous rate | < 10% | < 5% |
| Cross-framework agreement (same answer) | < 80% | < 70% |

Note: these thresholds are calibrated to the observed rates. The high dispersed rates (30-92%) reflect textual divergence in long-form generation, not answer-level disagreement. SLOs should be set on *extracted answer* agreement, not raw text match.

### Eval Cadence

- **Per-deploy**: run canary set (100 items, ~50 minutes per backend). Gate deployment on canary answer-level agreement.
- **Nightly**: full GSM8K + MMLU + canary suite. Generate trend dashboard. Flag any new items that shift from unanimous to dispersed.
- **Weekly**: logprob-level analysis on full suite. Review top-50 divergent items for new patterns.

## Implications for AI Reliability Engineering

This project demonstrates that "the model works" and "the model works the same way" are different claims requiring different evidence. Traditional reliability engineering focuses on availability (is the service up?) and latency (is it fast enough?). AI reliability must additionally track behavioral consistency: does the same input produce semantically equivalent output regardless of which serving node handles the request?

Key takeaways:

1. **Framework matters more than quantization level.** The sharpest divergence boundary is between MLX and llama.cpp, not between FP16 and Q4. Within the same framework, different quantization levels produce near-identical outputs for most prompts. Across frameworks, even at the same precision (Q8 vs FP16), outputs diverge significantly. This suggests that implementation details in attention computation, KV-cache management, and sampling logic have more behavioral impact than weight precision.

2. **Long-form generation amplifies divergence.** With max_tokens=256 and greedy decoding, small per-token probability differences compound over the sequence. A token-level divergence at position 20 cascades into entirely different text by position 100. This is why the canary set (which allows long-form responses) shows 92% dispersed, while short-answer extraction from MMLU shows 60% unanimous.

3. **Canary sets need answer extraction.** Raw text comparison classifies most outputs as "dispersed" because generation is inherently sensitive to initial conditions. The meaningful signal is in extracted answers — does the model get the same final answer regardless of backend? This requires dataset-specific extraction logic (regex for math, letter matching for MMLU).

4. **llama.cpp is fastest, MLX is most consistent.** llama.cpp Q4_K_M delivers 31% lower latency than MLX FP16, but the two llama.cpp backends show slightly more inter-backend divergence than the two MLX backends. The choice between frameworks is a speed-vs-consistency tradeoff.

5. **MPS FP16 is not viable for 7B models on 24 GB.** The buffer allocation failure means PyTorch MPS cannot be used as a serving backend for this model size without either reducing precision further or using a machine with more memory. This eliminates one potential backend from the heterogeneous serving pool.

The gap between "model works" and "model works the same way across all serving infrastructure" is where silent quality regressions live. Closing that gap requires treating behavioral consistency as a first-class reliability metric, measured continuously and enforced automatically.
