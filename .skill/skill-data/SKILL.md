---
name: skill-data
description: Generate deterministic DLLM language data, task splits, replay manifests, and dataset audits. Use when preparing synthetic smoke tests or reproducible continual-learning inputs.
---

# DLLM data operations

Use `scripts/make_text_tasks.py` for a dependency-free smoke dataset. It writes JSONL records with integer token IDs, disjoint task ranges, a seed, and a manifest-friendly schema; it does not pretend synthetic data is a language-quality result.

## Workflow

1. Fix `seed`, `tasks`, `examples_per_task`, `length`, and `vocab_size` in the run config.
2. Generate task JSONL and retain the command plus SHA-256. Never regenerate a task stream without changing the run ID.
3. Keep train and evaluation examples disjoint. For real text, map records to the same `input_ids` schema and record tokenizer name/version.
4. For masked diffusion, record the mask-probability grid separately. Mask probability is an operational DLLM variable, not identical to Gaussian SNR.
5. Keep replay data/task manifests immutable after training starts.

The generator has an assertion self-check. See [references/protocol.md](references/protocol.md) for the record schema and split rules.
