# PyTorch MPS Backend Limitations

## Non-Determinism

The MPS backend in PyTorch may exhibit non-deterministic behavior even when seeds
are fixed via `torch.manual_seed()` and `torch.mps.manual_seed()`. This is a known
limitation of the MPS compute framework where certain operations (e.g., atomics in
reduction kernels) do not guarantee bitwise reproducibility.

The determinism test (`test_determinism_with_seed`) is marked `xfail(strict=False)`
because repeated runs with identical seeds may produce different token sequences.
This finding is itself a meaningful result for the divergence analysis — it means
that MPS-based inference cannot be relied upon for exact reproducibility in the way
that CPU or CUDA (with deterministic mode) can.

## MPS Fallback Operations

When `PYTORCH_ENABLE_MPS_FALLBACK=1` is set, operations not natively supported on
MPS silently fall back to CPU execution. This can affect:

- **Performance**: CPU fallback adds host-device data transfer overhead.
- **Determinism**: Mixed device execution may introduce additional non-determinism.

The backend logs a warning at load time if this environment variable is detected.

### Known Fallback Operations (PyTorch 2.x)

- Some complex number operations
- Certain sparse tensor operations
- `torch.cumsum` on some dtypes (prior to PyTorch 2.1)

For Qwen2.5-7B-Instruct in FP16, the standard forward pass and generation loop
run entirely on MPS without fallback on PyTorch >= 2.1. If your PyTorch version
requires fallback, the backend remains functional but results may differ from a
pure-MPS execution.

## Memory Usage

Qwen2.5-7B-Instruct in FP16 requires approximately 14 GB of unified memory.
Ensure your system has sufficient RAM (16 GB minimum, 32 GB recommended for
comfortable operation alongside other processes).

## Recommendations

1. Run without `PYTORCH_ENABLE_MPS_FALLBACK=1` first. If generation fails with
   an unsupported-op error, enable it and note which operations fall back.
2. For reproducibility comparisons across backends, compare logprobs via `score()`
   rather than relying on exact token-id matches from `generate()`.
3. The `torch.mps.synchronize()` call before timing measurements ensures accurate
   latency reporting, since MPS operations are dispatched asynchronously.
