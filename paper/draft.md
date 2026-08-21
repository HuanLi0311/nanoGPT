# NanoAgent: A Codex-Compatible Harness and Verifier-Driven Post-Training Pipeline

## Abstract

This draft describes a small agent post-training pipeline that connects Codex-style JSON trajectories to a workspace harness and a verl-compatible GRPO data contract. The harness preserves structured tool calls, executes commands inside a workspace boundary, supports atomic episode checkpoints and retries, and verifies tool outcomes. The current evidence is preliminary: a tokenizer mismatch and empty-label SFT windows were found and fixed; a 10-step AdamW SFT pilot reduced a fixed masked loss from 10.16 to 0.72 on a 500-example pilot, while a one-step GRPO smoke completed but produced reward 0 and an empty response. No agent capability improvement is claimed yet.

A second harness review fixed four protocol/runtime gaps: run state files are now isolated by run id and reject mismatched or malformed state, relative command working directories are resolved from the workspace root, DeepSeek requests have a bounded timeout, and SFT rendering retains tool results with their call ids. A distributed-training audit also fixed SFT so every DDP rank loads the same pretrained checkpoint. The local self-check covers the harness paths, but no commercial-model episode has been run because the rotated API key has not been supplied.

A data-quality iteration then enforced the filter's previously unused failure and tool-pair checks. The default corpus changed from 4,756 to 4,594 episodes after removing unresolved calls, orphan results, and invalid events; the clean set contains 39,785 calls and 39,785 results. Re-encoding produced 67.46M train tokens and 3.40M validation tokens with the 32,768-token vocabulary. This is a training-input correction, not evidence of improved agent capability.

The next SFT audit found a more serious objective bug: the implementation compared logits at position `t` with the label at the same position, allowing the current token embedding to make the masked loss look nearly perfect without teaching next-token generation. After shifting SFT targets by one token, a 1,000-step five-GPU run went from loss 3.75 to 1.02 and generated non-empty text on train, validation, and held-out prompts. The outputs were still unstructured and produced no JSON tool action; a one-step GRPO smoke from this checkpoint reached reward 0.25 with a malformed natural-language completion. The before/after record and curve are in `logs/sft_alignment_iteration.json` and `assets/sft_clean_aligned_1000_loss.png`.

A focused follow-up retained all trajectory text as context but supervised only the 39,785 assistant tool-call turns (8.45M train labels and 0.52M validation labels). Continuing the aligned checkpoint for 1,000 steps on five A100s produced nonempty raw call-shaped text, but all greedy train, validation, and held-out completions had one extra closing brace. The harness parsed and executed none of the three outputs. This is a format failure, not an agent capability gain; the full record is `logs/sft_toolcalls_iteration.json`.

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
| Tool-call SFT, 1 GPU | 42,330 serialized Codex tool calls, 20 steps | loss 9.60 to 0.35; held-out and training prompts still produced empty output; GRPO reward 0 |
| Tool-call-only SFT, 5 GPUs (air-node-04) | 1,000-step continuation from aligned SFT; 8.45M labeled train tokens | loss 0.48 to 0.60; 0/3 parseable JSON actions, each completion had one trailing `}` |

The optimizer comparison is a pipeline result, not a generalization result. The SFT improvement is likely memorization on a tiny pilot. The full experiment must use a held-out agent benchmark and report task success, tool-call validity, recovery rate, and verifier score.

The tool-call serialization ablation is important: before the fix, assistant `tool_calls` were discarded and the model was trained on empty assistant turns. After the fix, the encoded corpus contains 42,330 tool calls, but a 20-step one-GPU run still produced empty generations. This separates a data-contract bug from the remaining optimization, sampling, and model-capacity problems.

## 4. Limitations and Next Experiments

The local NanoGPT checkpoint is a custom Transformer and cannot be loaded directly by verl's Hugging Face/vLLM workers. The current verl launcher therefore targets a compatible transformers checkpoint, while the local SFT/GRPO path exercises the custom model. A checkpoint adapter or a matched Hugging Face student is required for a true end-to-end verl run.

The verl launcher reached configuration validation and started a local Ray instance on `air-node-03`, but worker initialization failed with a node-health timeout before the first training step. This is recorded in `logs/verl_smoke_air-node-03.json`; it is a protocol/infrastructure result, not a GRPO result. The node's pre-existing eight-process pretraining job was left running.

On `air-node-04`, a second clean-data smoke progressed further through Ray, TransferQueue, prompt filtering, and FSDP actor initialization for Qwen2.5-1.5B. It still stopped before rollout because this verl release requires vLLM >= 0.18 while the node's usable vLLM 0.12 is unsupported; vLLM 0.18's CUDA extension was ABI-incompatible with the installed Torch 2.9. The launcher now defaults to `sdpa`, single-GPU tensor parallelism, and no Ray dashboard for this environment. The complete record is `logs/verl_air-node-04_iteration.json`; it contains no GRPO reward or update.

The tool-call-only continuation changes the failure mode from empty text to almost-valid raw call JSON, but it does not meet the runtime contract. GRPO is deferred because the current reward would conflate a syntax-repair problem with verified task execution. The next experiment should either train directly against the harness's exact action serialization or add a deterministic constrained decoder, then remeasure parser acceptance before any reward update.

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

# Focus only the labels on structured tool calls while retaining all context.
/home/JJ_Group/lih2511/.conda/envs/verl/bin/python -m model.language_model.scripts.prepare_post_train_sft \
  model/language_model/data/post_train/data/filtered \
  model/language_model/data/encode/sft_toolcalls \
  --tokenizer-dir model/language_model/data/encode/pretrain \
  --tool-calls-only
```

The current artifacts are `logs/sft_pilot_comparison.json`, `logs/rl_smoke_summary.json`, and the corresponding PNG curves in `assets/`. These artifacts are preliminary and deliberately do not claim improved agent performance.
