---
name: skill-benchmark
description: Run DLLM evaluations and aggregate rank-1, diagonal, forgetting, and quality metrics into standardized JSON. Use for benchmark execution, result comparison, or failure attribution.
---

# DLLM benchmark operations

Use `scripts/dllm_rank1_probe.py` to measure per-example masked-diffusion gradient geometry on the supplied SMDM checkpoint. It tests the paper's rank-1 claim in a DLLM-specific grid of mask probabilities and reports the diagonal approximation as the direct baseline.

## Workflow

1. Run a small smoke probe first (`170M`, short sequences, few samples). The probe must load the checkpoint, compute finite gradients, and emit one JSON envelope.
2. Run the prespecified mask grid, keeping seed, model, parameter slice, sequence length, sample count, and checkpoint hash fixed.
3. Treat mask probability as the DLLM analogue of noise level only operationally: report the mapping explicitly and do not claim Gaussian-SNR equivalence.
4. Use `scripts/aggregate_results.py` to combine independent envelopes. It computes mean, sample standard deviation, and count for every numeric metric without overwriting raw runs.
5. Interpret rank-1 support only when the result includes `lambda2_over_lambda1`, `rank1_relative_frobenius_error`, `diagonal_relative_frobenius_error`, and cosine-to-mean statistics. A single successful run is a feasibility result, not a replicated conclusion.

The probe can be pointed at any compatible SMDM source tree with `--code-root`. See [references/protocol.md](references/protocol.md).
