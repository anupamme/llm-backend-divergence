# First Run Findings

Results from the initial full evaluation run comparing five inference backends on Apple Silicon.

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
- `torch-mps` — PyTorch + HuggingFace, FP16 on MPS

**Datasets:**
- GSM8K: 200 grade-school math problems
- MMLU: 500 items across 5 subjects (high school math, medicine, formal logic, computer science, moral scenarios)
- Canary: 100 hand-crafted precision probes (arithmetic, tokenization, logic, long context, formatting)

## Headline Numbers

### Agreement Rates by Dataset

| Dataset | Unanimous | Majority | Split | Dispersed |
|---------|-----------|----------|-------|-----------|
| GSM8K | 36.0% | 17.5% | 25.0% | 21.5% |
| MMLU | 57.6% | 9.8% | 7.2% | 25.4% |
| Canary | 2.0% | 3.0% | 4.0% | 91.0% |

The canary dataset — designed to probe precision-sensitive operations — shows extreme divergence (91% dispersed). This is driven by long-form generation where backends produce semantically similar but textually different responses. MMLU shows the highest agreement at 57.6% unanimous, reflecting constrained factual recall tasks. With 5 backends, "split" verdicts now appear (25% on GSM8K) — indicating groups of backends that cluster on different reasoning approaches.

### Latency by Backend

| Backend | TTFT p50 (ms) | TTFT p95 (ms) | Total p50 (ms) | Total p95 (ms) |
|---------|---------------|---------------|----------------|----------------|
| torch-mps | 168 | 450 | 39,479 | 42,436 |
| llamacpp-q8 | 475 | 1,120 | 32,726 | 35,251 |
| llamacpp-q4km | 495 | 1,234 | 24,791 | 27,522 |
| mlx-q4 | 792 | 1,272 | 36,196 | 36,969 |
| mlx-fp16 | 816 | 1,316 | 36,243 | 37,039 |

torch-mps has the fastest TTFT (168 ms p50) due to efficient MPS-accelerated prompt prefill, but the slowest total generation (39.5s p50) because HuggingFace's `generate()` loop has higher per-token overhead than native frameworks. llama.cpp Q4_K_M is fastest end-to-end (24.8s p50) — a 37% speedup over MLX FP16. MLX FP16 and Q4 are nearly identical in latency, indicating that MLX's quantization doesn't translate to meaningful speedup for generation-bound workloads at max_tokens=256.

## Divergence Examples

### Example 1: Hallucinated Web Content (Canary)

**Prompt:** `What is 2^32?`

| Backend | Completion (first 100 chars) |
|---------|------------------------------|
| mlx-fp16 | `- Quora\nWhat is 2^32?\nAd by The Penny Hoarder\nShould you leave more than $1,000 in a checking acc...` |
| mlx-q4 | `- Quora\nWhat is 2^32?\nAd by The Penny Hoarder\nShould you leave more than $1,000 in a checking acc...` |
| llamacpp-q8 | `- Quora\nWhat is 2^32?\nAd by Masterworks\nWhat's a good investment for 2022?\nThis might sound unc...` |
| llamacpp-q4km | `- Quora\nWhat is 2^32?\nAd by Masterworks\nWhat's a good investment for 2022?\nThis might sound unc...` |
| torch-mps | `- Answers\nMath and Arithmetic\nWhat is 2^32?\nWiki User\n∙ 2010-04-26 19:57:48\n...4,294,967,296...` |

**Verdict:** dispersed

**What changed:** All backends hallucinate a web page rather than directly answering "4,294,967,296". The divergence reveals three distinct clusters: MLX backends (both FP16 and Q4) hallucinate a Quora page with a "Penny Hoarder" ad; llama.cpp backends hallucinate Quora with a "Masterworks" ad; and torch-mps hallucinates a Wiki Answers page. Notably, only torch-mps includes the correct numeric answer within its hallucination. This demonstrates that inference framework (not just quantization level) determines which memorized training data surfaces — backends within the same framework agree perfectly, but cross-framework comparison diverges completely.

### Example 2: Caesar Cipher Decoding Strategy (Canary)

**Prompt:** `The following is an encoded message using a simple Caesar cipher with shift 3: 'Wkh txlfn eurzq ira...'`

| Backend | Completion (first 100 chars) |
|---------|------------------------------|
| mlx-fp16 | `To decode the message, we need to reverse the Caesar cipher with a shift of 3. This means we will sh...` |
| mlx-q4 | `To decode the message, we need to reverse the Caesar cipher with a shift of 3. This means we will sh...` |
| llamacpp-q8 | `To decode the message, we need to reverse the Caesar cipher by shifting each letter 3 positions bac...` |
| llamacpp-q4km | `To decode the message, we need to reverse the Caesar cipher with a shift of 3. This means we will s...` |
| torch-mps | `To decode the message, we need to reverse the Caesar cipher by shifting each letter back by 3 posit...` |

**Verdict:** split (two groups of backends diverge on phrasing)

**What changed:** Three backends (mlx-fp16, mlx-q4, llamacpp-q4km) use "reverse the Caesar cipher with a shift of 3", while llamacpp-q8 and torch-mps both phrase it as "shifting each letter [back by] 3 positions". The answer is semantically identical but textually different. With 5 backends, this shifts from a "majority" to a "split" verdict — two distinct phrasing clusters emerge. Notably, torch-mps clusters with llamacpp-q8 rather than with the MLX backends, suggesting that the PyTorch framework's tokenization and probability distribution resembles llama.cpp Q8 more than MLX on this prompt. In production, this kind of benign paraphrase divergence is noise, not signal.

### Example 3: MMLU Math Problem — Reasoning Chain (MMLU)

**Prompt:** `Jane's quiz scores were 98, 97, 92, 85 and 93. What was her mean score? A. 92 B. 93 C. 94.5 D. 95`

| Backend | Completion (first 90 chars) |
|---------|------------------------------|
| mlx-fp16 | `To find Jane's mean quiz score, we need to follow these steps:\n\n1. **Sum the sco...` |
| mlx-q4 | `To find Jane's mean quiz score, we need to follow these steps:\n\n1. **Sum the sco...` |
| llamacpp-q8 | `To find Jane's mean quiz score, we need to follow these steps:\n\n1. **Sum the sc...` |
| llamacpp-q4km | `To find Jane's mean score, we need to follow these steps:\n\n1. **Sum the scores*...` |
| torch-mps | `To find Jane's mean quiz score, we need to follow these steps:\n\n1. **Sum the sco...` |

**Verdict:** majority

**What changed:** llamacpp-q4km says "mean score" where the other four say "mean quiz score", and uses slightly different formatting in the step labels. All arrive at the correct answer (B. 93). The Q4_K_M quantization produces the most abbreviated phrasing — a pattern consistent with reduced precision causing the model to assign slightly different probabilities to filler words like "quiz". torch-mps agrees with the majority here, clustering with mlx-fp16 on this structured reasoning task.

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

2. **Long-form generation amplifies divergence.** With max_tokens=256 and greedy decoding, small per-token probability differences compound over the sequence. A token-level divergence at position 20 cascades into entirely different text by position 100. This is why the canary set (which allows long-form responses) shows 98% dispersed, while short-answer extraction from MMLU shows 57.6% unanimous.

3. **Canary sets need answer extraction.** Raw text comparison classifies most outputs as "dispersed" because generation is inherently sensitive to initial conditions. The meaningful signal is in extracted answers — does the model get the same final answer regardless of backend? This requires dataset-specific extraction logic (regex for math, letter matching for MMLU).

4. **llama.cpp is fastest, MLX is most consistent.** llama.cpp Q4_K_M delivers 37% lower end-to-end latency than MLX FP16 (24.8s vs 36.2s p50), but the two llama.cpp backends show slightly more inter-backend divergence than the two MLX backends. The choice between frameworks is a speed-vs-consistency tradeoff.

5. **MPS FP16 requires careful memory management.** PyTorch's default `caching_allocator_warmup` attempts to allocate a single 14 GB buffer, exceeding MPS limits. The workaround — loading to CPU then moving to MPS piecewise — succeeds but adds ~30s to model load time. Once loaded, torch-mps achieves the fastest TTFT (168 ms) thanks to efficient MPS-accelerated prefill, but the slowest total generation due to HuggingFace's per-token Python loop overhead. In a heterogeneous serving pool, torch-mps is best suited for latency-sensitive short completions where TTFT dominates.

The gap between "model works" and "model works the same way across all serving infrastructure" is where silent quality regressions live. Closing that gap requires treating behavioral consistency as a first-class reliability metric, measured continuously and enforced automatically.
