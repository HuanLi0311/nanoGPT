# Benchmark protocol

`dllm_rank1_probe.py` loads the supplied masked-diffusion checkpoint and
computes per-example gradients for one explicit parameter slice. It reports
the empirical Fisher's `lambda2_over_lambda1`, mean-gradient cosine,
mean-gradient rank-1 relative Frobenius error, and diagonal relative
Frobenius error. The mask probability is only an operational DLLM analogue of
noise level; it is not Gaussian SNR.

Run a smoke probe before the grid, then preserve one raw JSON envelope per
seed/model configuration. Aggregate only with:

```bash
python3 scripts/aggregate_results.py runs/*/benchmark.json \
  --output runs/aggregate.json
```

The aggregate contains means, sample standard deviations, and counts. Raw
envelopes are never overwritten. A single successful seed is feasibility
evidence, not a replicated conclusion.
