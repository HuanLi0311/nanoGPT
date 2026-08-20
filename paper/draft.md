# NanoAgent: A Codex-Compatible Harness and Verifier-Driven Post-Training Pipeline

## Abstract

This draft describes a small agent post-training pipeline that connects Codex-style JSON trajectories to a workspace harness and a verl-compatible GRPO data contract. The harness preserves structured tool calls, executes commands inside a workspace boundary, supports atomic episode checkpoints and retries, and verifies tool outcomes. The current evidence is preliminary: a tokenizer mismatch and empty-label SFT windows were found and fixed; a 10-step AdamW SFT pilot reduced a fixed masked loss from 10.16 to 0.72 on a 500-example pilot, while a one-step GRPO smoke completed but produced reward 0 and an empty response. No agent capability improvement is claimed yet.

## 1. Motivation

Agent training data contains more than assistant text. Tool names, JSON arguments, tool results, exit codes, and the final answer form an execution protocol. Flattening these events into text loses the information needed to train or evaluate an agent in the same environment. NanoAgent therefore keeps a canonical trajectory format and exposes a narrow runtime contract to the trainer.

## 2. Method

The pipeline has four boundaries:

1. Codex JSONL logs are parsed into user, assistant, tool-call, and tool-result events. Developer instructions, session metadata, world state, and reasoning items are excluded from training events.
2. The runtime exposes `exec_command` with a workspace-root check and `apply_patch` supporting both unified diffs and Codex `*** Begin Patch` syntax. State is written through a temporary file and rename, so resume does not observe a partial JSON state.
3. A verifier scores final responses using tool failures and completion evidence. The reward hook for verl scores structured JSON actions and can be replaced by a task-specific verifier.
4. Filtered trajectories are converted to the verl `data_source/prompt/reward_model/extra_info` contract. SFT uses the base checkpoint tokenizer and refuses mismatched vocabularies.

DeepSeek is integrated through its OpenAI-compatible `/chat/completions` endpoint. The key is read only from `DEEPSEEK_API_KEY`; no secret is stored in the repository. A teacher comparison is intentionally not reported until a rotated key is supplied and the same benchmark is run against the local policy.

## 3. Preliminary Experiments

All measurements below were run on `air-node-03` with 8 A100 GPUs unless stated otherwise. The SFT pilot used 500 filtered Codex examples re-encoded with the 32768-token vocabulary matching `best.safetensors`.

| Run | Change | Result |
| --- | --- | --- |
| Muon, 10 steps | existing optimizer, learning rate 5e-5 | loss 9.96 to 14.00 |
| Muon, 10 steps | learning rate 1e-5 | loss 9.96 to 14.19 |
| AdamW, 10 steps | before empty-label guard | NaN at step 8 |
| AdamW, 10 steps | valid-label sampling and NaN guard | loss 9.89 to 1.36; fixed masked loss 0.72 |
| GRPO smoke | AdamW, group 2, one step | checkpoint written; reward 0; response was whitespace |

The optimizer comparison is a pipeline result, not a generalization result. The SFT improvement is likely memorization on a tiny pilot. The full experiment must use a held-out agent benchmark and report task success, tool-call validity, recovery rate, and verifier score.

## 4. Limitations and Next Experiments

The local NanoGPT checkpoint is a custom Transformer and cannot be loaded directly by verl's Hugging Face/vLLM workers. The current verl launcher therefore targets a compatible transformers checkpoint, while the local SFT/GRPO path exercises the custom model. A checkpoint adapter or a matched Hugging Face student is required for a true end-to-end verl run.

The next controlled experiment is: freeze the 500-example pilot split, train AdamW and Muon for the same number of steps, evaluate on held-out Codex tasks, then compare the local policy with a DeepSeek teacher using identical prompts, tools, timeouts, and verifier rules. Results must be written to `logs/` and plotted in `assets/` before making a capability claim.

## 5. Reproducibility

```bash
git submodule update --init --recursive
.venv/bin/python -m model.language_model.scripts.prepare_post_train_sft \
  model/language_model/data/post_train/data/filtered \
  model/language_model/data/encode/sft \
  --tokenizer-dir model/language_model/data/encode/pretrain \
  --limit 500 --shard-tokens 10000000
./model/language_model/scripts/sft.sh
```

The current artifacts are `logs/sft_pilot_comparison.json`, `logs/rl_smoke_summary.json`, and the corresponding PNG curves in `assets/`. These artifacts are preliminary and deliberately do not claim improved agent performance.
