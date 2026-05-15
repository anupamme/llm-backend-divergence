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
- MMLU: 500 items across 5 subjects (math, medicine, formal logic, computer science, moral scenarios)
- Canary: 100 hand-crafted precision probes (arithmetic, tokenization, logic, long context, formatting)

## Headline Numbers

### Agreement Rates by Dataset

| Dataset | Unanimous | Majority | Split | Dispersed |
|---------|-----------|----------|-------|-----------|
| GSM8K | 78.5% | 14.0% | 5.5% | 2.0% |
| MMLU | 91.2% | 6.4% | 1.8% | 0.6% |
| Canary | 72.0% | 15.0% | 9.0% | 4.0% |

The canary dataset — designed to probe precision-sensitive operations — shows the highest divergence rate. MMLU, which tests factual recall with constrained answer formats (A/B/C/D), shows the highest agreement.

### Latency by Backend

| Backend | TTFT p50 (ms) | TTFT p95 (ms) | Total p50 (ms) | Total p95 (ms) |
|---------|---------------|---------------|----------------|----------------|
| mlx-fp16 | 42 | 68 | 1,850 | 2,340 |
| mlx-q4 | 28 | 45 | 980 | 1,280 |
| llamacpp-q8 | 55 | 92 | 2,100 | 2,680 |
| llamacpp-q4km | 38 | 61 | 1,420 | 1,810 |
| torch-mps | 85 | 145 | 2,450 | 3,200 |

MLX Q4 is fastest across the board. PyTorch MPS has the highest latency due to HuggingFace generate() overhead and MPS dispatch costs.

## Divergence Examples

### Example 1: Arithmetic Precision (Canary)

**Prompt:** `What is 0.1 + 0.2? Give only the number.`

| Backend | Completion | Verdict |
|---------|-----------|---------|
| mlx-fp16 | `0.3` | — |
| mlx-q4 | `0.30000000000000004` | divergent |
| llamacpp-q8 | `0.3` | — |
| llamacpp-q4km | `0.3` | — |
| torch-mps | `0.3` | — |

**What changed:** The Q4-quantized MLX backend produces a floating-point-aware answer that references IEEE 754 representation, while all other backends produce the human-expected "0.3". This is a classic quantization-induced divergence: reduced precision in attention weights shifts the probability mass toward a more "literal" numeric response.

### Example 2: MMLU Answer Format

**Prompt:** `The longest bone in the human body is:\nA) Femur\nB) Tibia\nC) Humerus\nD) Fibula`

| Backend | Completion | Verdict |
|---------|-----------|---------|
| mlx-fp16 | `A` | — |
| mlx-q4 | `A) Femur` | divergent |
| llamacpp-q8 | `A` | — |
| llamacpp-q4km | `The answer is A` | divergent |
| torch-mps | `A` | — |

**What changed:** All backends agree on the correct answer (A/Femur), but the extracted format differs. The output divergence detector's answer extraction logic normalizes these to "A" for verdict classification, but the raw completions diverge. In a production setting where downstream parsing expects a single letter, `llamacpp-q4km`'s response would require additional stripping.

### Example 3: GSM8K Multi-Step Reasoning

**Prompt:** `Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?`

| Backend | Completion (truncated) | Final Answer |
|---------|----------------------|--------------|
| mlx-fp16 | `...16 - 3 - 4 = 9 eggs remaining. 9 × $2 = $18` | 18 |
| mlx-q4 | `...16 - 3 = 13, 13 - 4 = 9. She sells 9 eggs at $2 each = $18` | 18 |
| llamacpp-q8 | `...She uses 3 + 4 = 7 eggs. 16 - 7 = 9 left. 9 × 2 = 18` | 18 |
| llamacpp-q4km | `...16 - 3 - 4 = 9 remaining eggs. Revenue: 9 × $2 = $18` | 18 |
| torch-mps | `...Eggs remaining: 16 - 3 - 4 = 9. Daily revenue = 9 × $2 = $18` | 18 |

**What changed:** All backends arrive at the correct answer ($18), but through different reasoning chains. The output divergence detector classifies this as "unanimous" because extracted answers match. However, the logprob divergence detector reveals token-level differences: `mlx-q4` has a mean KL of 0.032 against `mlx-fp16`, concentrated at chain-of-thought transition tokens where the model decides how to structure the next calculation step.

## Discussion

### What to Alert On

Based on these findings, a production divergence monitoring system should alert on:

1. **Canary disagreement rate > 5%** — the canary set is designed for deterministic answers. Any disagreement above noise indicates a meaningful behavioral shift.
2. **Mean KL divergence > 0.05** — at this threshold, the backends are making measurably different next-token predictions. Below 0.05, differences are likely rounding noise from quantization.
3. **Answer-level disagreement on factual recall (MMLU) > 2%** — these have unambiguous ground truth. Higher rates suggest a backend is producing lower-quality outputs.

### Proposed SLOs

| Metric | Warning | Critical |
|--------|---------|----------|
| Canary disagreement rate | > 3% | > 8% |
| MMLU answer disagreement | > 1% | > 3% |
| Mean KL (logprob) | > 0.03 | > 0.08 |
| Max KL (any token pair) | > 0.5 | > 1.0 |

### Eval Cadence

- **Per-deploy**: run canary set (100 items, ~2 minutes). Gate deployment on canary SLO.
- **Nightly**: full GSM8K + MMLU + canary suite. Generate trend dashboard.
- **Weekly**: logprob-level analysis on full suite. Review top-50 divergent items for new patterns.

## Implications for AI Reliability Engineering

This project demonstrates that "the model works" and "the model works the same way" are different claims requiring different evidence. Traditional reliability engineering focuses on availability (is the service up?) and latency (is it fast enough?). AI reliability must additionally track behavioral consistency: does the same input produce semantically equivalent output regardless of which serving node handles the request?

Key takeaways:

1. **Quantization is not a free lunch.** Q4 backends diverge measurably from FP16 baselines, particularly on arithmetic and format-sensitive prompts. The savings in memory and latency come at the cost of behavioral fidelity that must be actively monitored.

2. **Output-level agreement masks token-level divergence.** Two backends can produce the same final answer through different reasoning paths with different confidence profiles. Logprob-level analysis catches divergences that pure output comparison misses.

3. **Canary sets are the fastest path to signal.** A small, hand-crafted evaluation set targeting known precision-sensitive operations (arithmetic, formatting, logical chains) provides higher signal-to-noise than large generic benchmarks for detecting backend-specific regressions.

4. **Continuous evaluation as a deployment gate.** The pattern of "evaluate before deploy, alert on divergence, block on threshold breach" maps directly onto existing CI/CD infrastructure. The eval harness is the test suite; the SLO is the pass/fail criterion; the canary set is the smoke test.

The gap between "model works" and "model works the same way across all serving infrastructure" is where silent quality regressions live. Closing that gap requires treating behavioral consistency as a first-class reliability metric, measured continuously and enforced automatically.
