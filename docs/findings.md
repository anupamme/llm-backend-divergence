# First Run Findings

Results from the initial evaluation run comparing five inference backends on Apple Silicon.

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

## Methodology Note: Chat Template

This initial run passed raw prompt strings to all backends **without applying the Qwen2.5-Instruct chat template** (`<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`). This means the model operated in base-model completion mode rather than instruction-following mode for all backends identically.

This is a methodological flaw: the hallucination behavior (e.g., generating Quora pages for "What is 2^32?") is characteristic of prompting an instruct model without its chat template. However, because the same raw prompt was passed to all five backends, the *relative comparison* between backends remains valid — we are measuring how backends diverge given identical input, even if that input is suboptimal. The absolute quality of outputs does not affect the divergence signal.

A follow-up run with chat templates applied is documented in the "Before/After: Chat Template" section below. The pipeline now applies chat templates by default (`divergence/prompt_format.py`).

## Structured Divergence Analysis

### Extracted-Answer Agreement by Comparison Group

Rather than presenting a single agreement table that conflates quantization effects with framework effects, we separate comparisons into three tiers:

| Group | GSM8K | MMLU | Canary (overall) | Canary (arithmetic subset) |
|-------|-------|------|------------------|---------------------------|
| **Within-framework** (FP16 vs Q4, same framework) | 78.2% | 92.3% | 81.0% | 96.2% |
| **Cross-framework, matched precision** (MLX FP16 vs torch-mps FP16) | 53.5% | 86.0% | 61.0% | 92.5% |
| **Cross-framework, confounded** (different framework + precision) | 53.1% | 84.3% | 58.3% | 91.4% |

These are extracted-answer agreement rates — not raw text match. Extraction logic varies by dataset: for GSM8K, the final numeric answer (after `####` or last number); for MMLU, the letter choice (A-D); for canary arithmetic items (40/100), the first number in the completion; for non-arithmetic canary items (60/100), the first 100 characters of stripped output (no reliable answer format exists for logic, formatting, and long-context dimensions). The "overall" canary column blends both tiers.

**Key observation:** Within-framework agreement is 20-25 percentage points higher than cross-framework for GSM8K and canary overall. For MMLU (short factual answers), the gap narrows to 6-8 points. For canary arithmetic items specifically, all groups exceed 91% agreement — numeric extraction collapses textual variation, confirming that much of the "divergence" on canary is paraphrase noise, not answer disagreement. This pattern is consistent with the hypothesis that framework implementation details (attention kernels, KV-cache management, sampling code) introduce more behavioral divergence than weight quantization.

### Raw Text Verdict Distribution

For completeness, the raw text comparison verdicts (which conflate semantic agreement with phrasing variation):

| Dataset | Unanimous | Majority | Split | Dispersed |
|---------|-----------|----------|-------|-----------|
| GSM8K | 36.0% | 17.5% | 25.0% | 21.5% |
| MMLU | 57.6% | 9.8% | 7.2% | 25.4% |
| Canary | 2.0% | 3.0% | 4.0% | 91.0% |

**Caveat:** The 91% "dispersed" on canary does not mean backends disagree on answers — it means long-form generation produces textually different outputs even when the semantic content is equivalent. The arithmetic subset achieves 96% within-framework agreement when numeric answer extraction is applied, vs raw-text "dispersed" verdicts on those same items. This metric is useful for detecting copy-paste equivalence but not for measuring behavioral consistency. The extracted-answer table above is the operationally meaningful metric.

## Latency Breakdown

### Per-Stage Latency

| Backend | Prefill (TTFT) p50 | Decode/token p50 | Total p50 | Total p95 |
|---------|--------------------|--------------------|-----------|-----------|
| torch-mps | 168 ms | 154.1 ms | 39,479 ms | 42,436 ms |
| llamacpp-q8 | 475 ms | 130.2 ms | 32,726 ms | 35,251 ms |
| llamacpp-q4km | 495 ms | 97.7 ms | 24,791 ms | 27,522 ms |
| mlx-q4 | 792 ms | 138.8 ms | 36,196 ms | 36,969 ms |
| mlx-fp16 | 816 ms | 139.0 ms | 36,243 ms | 37,039 ms |

### Interpreting the TTFT Gap

torch-mps shows 3-5x faster TTFT (168 ms) than other backends. This warrants explanation rather than assertion:

- **HuggingFace's `generate()` returns the first token sooner** because its prefill pass is a single batched matrix multiply on the MPS GPU. The prompt's full KV-cache is computed in one forward pass before the decode loop begins. Our timing starts before `generate()` and records first-token emission via a `StoppingCriteria` callback.
- **llama.cpp processes the prompt through its quantized kernel pipeline** with additional overhead from GGUF format decoding and Metal shader compilation on first use. The 475-495 ms TTFT includes both prompt evaluation and KV-cache allocation.
- **MLX's `stream_generate()` has a higher-overhead prompt evaluation** path, processing the prompt in chunks through its lazy-evaluation graph before beginning token emission.

An unresolved confound: each backend's timing hook fires at a slightly different point in its generation loop (HuggingFace `StoppingCriteria` callback vs. MLX `stream_generate` yield vs. llama.cpp streaming chunk). While all nominally measure time-to-first-token, we have not verified that these boundaries correspond to the same logical event across frameworks. The 3-5x gap may partially reflect measurement-boundary differences rather than pure compute differences.

The TTFT advantage does not translate to end-to-end speed: torch-mps has the highest per-token decode latency (154 ms vs 98-139 ms for others) due to Python-level overhead in HuggingFace's autoregressive loop vs native C++/Metal decode loops in llama.cpp and MLX.

### MLX FP16 vs Q4 Latency Parity

MLX FP16 and Q4 show near-identical decode-per-token latency (139.0 vs 138.8 ms). This is unexpected — Q4 should reduce memory bandwidth pressure on Apple Silicon's unified memory. The most likely explanation: at 256 max tokens, the Python/framework overhead in `mlx_lm.stream_generate()` dominates over the actual matrix-multiply time, masking the bandwidth reduction that quantization provides. A per-token breakdown of compute vs. framework overhead would isolate this further.

## Aggregate Statistics

### Token Count Distribution

| Backend | Mean tokens | Median | Min | Max |
|---------|-------------|--------|-----|-----|
| llamacpp-q4km | 231.7 | 256 | 6 | 256 |
| llamacpp-q8 | 232.7 | 256 | 30 | 256 |
| torch-mps | 230.7 | 256 | 7 | 256 |
| mlx-fp16 | 237.9 | 256 | 7 | 256 |
| mlx-q4 | 238.6 | 256 | 7 | 256 |

All backends hit max_tokens (256) on the majority of prompts — median is 256 across the board. This confirms that most outputs are length-truncated rather than naturally terminated. The small differences in mean (231-239) reflect different rates of early stopping (EOS token emission before 256 tokens). MLX backends show slightly higher mean token counts (238 vs 231), though whether this reflects a real difference in EOS emission probability or sampling noise is unclear without per-prompt variance analysis.

### Tokenizer Alignment

The logprob divergence analysis detected **zero tokenization mismatches** across all backend pairs. This is expected: all five backends use the same Qwen2.5 tokenizer (via HuggingFace for MLX and torch-mps, via the GGUF-embedded tokenizer for llama.cpp). The GGUF tokenizer is extracted from the same HuggingFace source during model conversion, ensuring byte-level compatibility.

This is a notable negative finding — tokenizer divergence is a known failure mode in heterogeneous serving, and its absence here means all observed divergence on this prompt set is attributable to compute rather than tokenization differences.

## Divergence Examples

The following examples illustrate patterns visible in the aggregate data. They are selected to show distinct failure modes, not to prove a general claim from a single instance.

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

**Root cause:** Without the chat template, the model treats the raw prompt as a continuation of training data and generates memorized web page content. The divergence reveals three clusters matching frameworks — MLX backends hallucinate Quora with a "Penny Hoarder" ad; llama.cpp backends hallucinate Quora with a "Masterworks" ad; torch-mps hallucinates a Wiki Answers page.

**What survives scrutiny:** The within-framework clustering is real — backends sharing a framework agree on which memorized content surfaces, even though the weights differ (FP16 vs Q4). This suggests that implementation-level choices in how logits are computed (floating-point accumulation order, attention kernel design) consistently selected between near-equiprobable continuations on this prompt.

**What doesn't:** The hallucination itself is an artifact of missing chat template, not a production-relevant finding. With proper prompting, the model answers "4,294,967,296" directly (see Before/After section below).

### Example 2: Caesar Cipher Decoding Strategy (Canary)

**Prompt:** `The following is an encoded message using a simple Caesar cipher with shift 3: 'Wkh txlfn eurzq ira...'`

| Backend | Completion (first 100 chars) |
|---------|------------------------------|
| mlx-fp16 | `To decode the message, we need to reverse the Caesar cipher with a shift of 3. This means we will sh...` |
| mlx-q4 | `To decode the message, we need to reverse the Caesar cipher with a shift of 3. This means we will sh...` |
| llamacpp-q8 | `To decode the message, we need to reverse the Caesar cipher by shifting each letter 3 positions bac...` |
| llamacpp-q4km | `To decode the message, we need to reverse the Caesar cipher with a shift of 3. This means we will s...` |
| torch-mps | `To decode the message, we need to reverse the Caesar cipher by shifting each letter back by 3 posit...` |

**Verdict:** split (two phrasing clusters)

**What this shows:** Three backends (mlx-fp16, mlx-q4, llamacpp-q4km) use "with a shift of 3", while llamacpp-q8 and torch-mps use "shifting each letter [back by] 3 positions". All produce semantically correct approaches. This is benign paraphrase divergence — noise, not signal. In production monitoring, this class of divergence should be filtered by extracted-answer comparison rather than raw text match.

### Example 3: MMLU Math — Filler Word Variation (MMLU)

**Prompt:** `Jane's quiz scores were 98, 97, 92, 85 and 93. What was her mean score? A. 92 B. 93 C. 94.5 D. 95`

| Backend | Completion (first 90 chars) |
|---------|------------------------------|
| mlx-fp16 | `To find Jane's mean quiz score, we need to follow these steps:\n\n1. **Sum the sco...` |
| mlx-q4 | `To find Jane's mean quiz score, we need to follow these steps:\n\n1. **Sum the sco...` |
| llamacpp-q8 | `To find Jane's mean quiz score, we need to follow these steps:\n\n1. **Sum the sc...` |
| llamacpp-q4km | `To find Jane's mean score, we need to follow these steps:\n\n1. **Sum the scores*...` |
| torch-mps | `To find Jane's mean quiz score, we need to follow these steps:\n\n1. **Sum the sco...` |

**Verdict:** majority (4 agree, llamacpp-q4km diverges on phrasing)

**What this shows:** llamacpp-q4km drops "quiz" from "mean quiz score" — the most aggressively quantized backend (Q4_K_M) produces slightly different token probabilities for low-information filler words. All backends extract to the same answer (B). This is benign phrasing divergence from quantization: the answer is preserved but verbatim text differs.

## Discussion

### What to Alert On

Based on these findings, a production divergence monitoring system should alert on:

1. **Cross-framework disagreement on extracted answers.** When backends give different final answers (not just different phrasing), this indicates a meaningful behavioral divergence. The structured comparison shows 53% cross-framework agreement on GSM8K vs 78% within-framework — a 25-point gap that represents real answer-level disagreement. The canary arithmetic subset (96% agreement across all groups) demonstrates that when proper answer extraction is applied, most divergence collapses to benign paraphrase — making extraction logic a prerequisite for meaningful alerting.
2. **Canary regression on deterministic tasks.** For prompts with unambiguous correct answers (arithmetic, logic), any within-framework disagreement signals a weight-loading or computation bug.
3. **TTFT regression.** A backend whose TTFT increases >2x likely has a compute path change (e.g., falling back to CPU, shader recompilation).

### Proposed SLOs

| Metric | Warning | Critical |
|--------|---------|----------|
| MMLU extracted-answer agreement (within-framework) | < 90% | < 85% |
| GSM8K extracted-answer agreement (within-framework) | < 70% | < 60% |
| Cross-framework extracted-answer agreement | < 50% | < 40% |
| TTFT p95 regression (vs baseline) | > 2x | > 5x |

Warning thresholds are set approximately one confidence interval below observed baselines, so that normal run-to-run variance does not trigger alerts while genuine regressions are caught. SLOs are set on extracted-answer agreement, not raw text match. The "cross-framework" SLO is intentionally looser because framework-level divergence is expected and acceptable; what matters is detecting regressions from baseline.

### Eval Cadence

- **Per-deploy**: run canary set (100 items, ~50 minutes per backend). Gate deployment on within-framework canary answer-level agreement.
- **Nightly**: full GSM8K + MMLU + canary suite. Generate trend dashboard. Flag any new items that shift from within-framework-agree to within-framework-disagree.
- **Weekly**: logprob-level analysis on full suite. Review top-50 divergent items for new patterns.

## Limitations

1. **No chat template in initial run.** Prompts were passed raw without the model's expected `<|im_start|>` formatting. Absolute output quality is degraded; relative comparison remains valid. Fixed in subsequent runs.
2. **Single seed (42).** With temperature=0 (greedy decoding), seed only affects non-deterministic hardware paths. MPS is known to have non-deterministic operations even with fixed seeds.
3. **Single hardware configuration.** Apple M4 Pro, 24 GB. Results may differ on M1/M2/M3 due to different Metal GPU cores and memory bandwidth.
4. **Single model.** Qwen2.5-7B-Instruct. Divergence patterns may differ for other architectures (Llama, Mistral) or model sizes.
5. **256 max tokens.** Truncates many GSM8K reasoning chains before the final answer. The 78% within-framework agreement on GSM8K may improve with higher max_tokens that allow natural completion.
6. **Greedy decoding only.** Temperature=0 removes sampling stochasticity but does not reflect production serving where temperature>0 amplifies divergence.
7. **No warmup exclusion.** First 1-2 items per backend may have elevated latency from JIT compilation or shader warm-up. These are included in statistics.
8. **No statistical significance testing.** With n=200 (GSM8K) and n=100 (canary), confidence intervals on agreement rates are wide (~±5-7%). Differences smaller than this margin should not be over-interpreted.

## Implications for AI Reliability Engineering

This project demonstrates that "the model works" and "the model works the same way" are different claims requiring different evidence. Traditional reliability engineering focuses on availability and latency. AI reliability must additionally track behavioral consistency.

Key takeaways:

1. **Framework matters more than quantization level — with evidence.** Within-framework agreement: GSM8K 78.2%, MMLU 92.3%, canary 81.0%. Cross-framework agreement: GSM8K 53.5%, MMLU 86.0%, canary 61.0%. The 20-25 point gap on GSM8K and canary is large enough to be operationally significant, even accounting for the n=200/100 sample sizes. The MMLU gap is smaller (6 points) because short factual answers are more constrained.

2. **Long-form generation compounds divergence with sequence length.** The canary set (long-form, 91% dispersed on raw text) vs MMLU (short-answer, 57.6% unanimous on raw text) demonstrates that per-token probability differences compound over sequence length. At position 20, a small logit difference may select a different token; by position 100, the texts have diverged completely. This is why extracted-answer comparison is essential — raw text equality is unachievable for generation-bound workloads.

3. **Answer extraction is the operationally meaningful metric.** Raw text verdicts classify 91% of canary outputs as "dispersed", while extracted-answer comparison shows 81% within-framework agreement overall — and 96% for arithmetic items where proper numeric extraction is possible. The gap between raw-text and extracted-answer metrics represents benign paraphrase variation that should not trigger alerts. Per-dataset extraction logic (regex for math, letter matching for MMLU) is required infrastructure for a production divergence monitor.

4. **llama.cpp is fastest, MLX backends agreed more often.** llama.cpp Q4_K_M: 24.8s total (98 ms/token decode). MLX FP16: 36.2s total (139 ms/token decode). The 32% speed gap comes entirely from decode efficiency — Q4_K_M's quantized Metal kernels achieve lower per-token latency. MLX's two backends agreed more often than llama.cpp's in this run — though with one quantization pair per framework, this could reflect the specific quantization levels tested (FP16→Q4 vs Q8→Q4_K_M) rather than an inherent framework property.

5. **MPS FP16 has the fastest prefill but slowest decode.** torch-mps achieves 168 ms TTFT (3x faster than llama.cpp, 5x faster than MLX) because HuggingFace's batched forward pass leverages MPS matrix-multiply acceleration for the full prompt in one shot. But its per-token decode (154 ms) is 57% slower than llama.cpp Q4_K_M due to Python-level autoregressive loop overhead. In production: use MPS-style backends for latency-sensitive first-token scenarios; use llama.cpp for throughput-bound generation.

6. **Correct prompting is a reliability prerequisite.** The chat template omission demonstrates that "model gives wrong answer" and "backend diverges from other backends" are independent failure modes. The initial run found both simultaneously — but only the latter is the divergence detector's job. Fixing the template (see below) eliminates the quality bug but preserves the divergence signal, confirming that the measurement approach is robust.

### Remediation: From Detection to Action

Detection is necessary but not sufficient. A production divergence monitoring system needs an operational model for *what to do* when divergence is found:

- **Pin a reference backend** (e.g., MLX FP16) and measure other backends' divergence *from reference*, not pairwise. This turns an O(n²) comparison matrix into n-1 directional checks and answers the question "which backend changed?" rather than "they differ."
- **Accept benign paraphrase divergence.** When backends produce different text but the same extracted answer, this is expected behavior, not a defect. Only extracted-answer disagreement with the reference should trigger alerts.
- **Alert on answer-flips.** The critical signal is when a backend produces a *different extracted answer* from the reference — this indicates a change in model behavior that may affect downstream systems.
- **The operational question:** when backends disagree on an answer, which one is "correct"? The reference backend's answer is treated as ground truth for consistency purposes, regardless of objective correctness. Correctness is the eval suite's job; the divergence monitor's job is detecting *change*.

## Before/After: Chat Template

After implementing `divergence/prompt_format.py`, we reran the full canary set (100 items × 5 backends) with the chat template applied. Results:

### Quality Impact: Hallucinations Eliminated

| Backend | Hallucination rate (without template) | Hallucination rate (with template) |
|---------|---------------------------------------|-------------------------------------|
| mlx-fp16 | 4.0% | 0.0% |
| mlx-q4 | 4.0% | 0.0% |
| llamacpp-q8 | 3.0% | 0.0% |
| llamacpp-q4km | 2.0% | 0.0% |
| torch-mps | 3.0% | 0.0% |

With 2-4 events per backend, the per-backend rates are not distinguishable from each other; the finding is that hallucination is present without the template and absent with it, not that any particular backend hallucinates more than another.

With the chat template, "What is 2^32?" produces direct answers across all backends:
- mlx-fp16, mlx-q4, llamacpp-q8: `2^32 equals 4,294,967,296.`
- llamacpp-q4km, torch-mps: `The value of 2^32 is 4,294,967,296.`

Two phrasing clusters persist (same answer, different wording) — demonstrating that framework-level divergence exists independently of the hallucination artifact.

### Divergence Impact: Unchanged

| Group | Without Template | With Template |
|-------|-----------------|---------------|
| Within-framework | 81.0% | 81.0% |
| Cross-framework (matched precision) | 61.0% | 61.0% |
| Cross-framework (confounded) | 58.3% | 58.4% |
| 5-way unanimous | 46.0% | 45.0% |

With proper per-dimension answer extraction (numeric for arithmetic items, 100-char prefix for others), divergence rates are virtually identical before and after chat template application — despite the underlying completions being entirely different (verbose explanations without the template vs concise answers with it). The within-framework rate is 162/200 in both runs; for the llamacpp pair specifically, 12 items flipped from agree-to-disagree and 12 from disagree-to-agree, yielding a net-zero change that explains the identical headline number. The corrected numbers confirm that the chat template has no measurable effect on answer-level divergence at this sample size (n=100, ~±6% CI).

### Interpretation

1. **Template application is not the primary source of divergence.** The within-framework vs cross-framework gap persists at similar magnitude, and the phrasing-cluster patterns (e.g., two groups on "What is 2^32?") continue to align with framework boundaries. The structural findings from the initial run appear robust, though a larger-n rerun would be needed to fully confirm this.

2. **Quality and consistency are independent failure modes.** The template omission caused a quality bug (hallucinations) that is fully corrected by proper prompting. But the divergence signal — different backends producing different phrasings for the same semantic content — exists regardless of prompt formatting. A production system needs both: correct prompting for quality AND divergence monitoring for consistency.

The `--no-chat-template` flag preserves backward compatibility. Compare with:
```bash
divergence run --no-chat-template --datasets canary --db results/baseline_no_template.db
divergence run --datasets canary --db results/with_template.db
```
