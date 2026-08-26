# Reusable DLLM skills

The five hyphenated directories hold the implementations; matching underscore
aliases (`skill_env`, `skill_data`, `skill_train`, `skill_benchmark`, and
`skill_paper`) are provided for the requested names. Each directory has
its own `SKILL.md`, a dependency-light script entry point, and a protocol file.

| Module | Entry point | Purpose |
|---|---|---|
| `skill-env` | `scripts/probe.py` | read-only environment and `air-node-03` probe |
| `skill-data` | `scripts/make_text_tasks.py` | deterministic disjoint text-task JSONL |
| `skill-train` | `scripts/checkpoint_info.py` | safetensors header/hash/checkpoint audit |
| `skill-benchmark` | `scripts/dllm_rank1_probe.py`, `aggregate_results.py` | masked-DLLM Fisher geometry and JSON aggregation |
| `skill-paper` | `scripts/extract_evidence.py` | PDF/LaTeX evidence extraction and query index |

All entry points have a `--self-check`. No new runtime dependency was added;
the model probe reuses the existing SMDM source tree and environment.
