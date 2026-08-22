#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/../../.." && pwd)
verl="$root/third_party/verl"
data="$root/model/language_model/data/post_train/verl"
# Student default: a Hugging Face Qwen coding checkpoint. DeepSeek is used as
# the external teacher/calibration model, not as the policy in this launcher.
model=${MODEL_PATH:-Qwen/Qwen2.5-Coder-7B-Instruct}
gpus=${NGPUS_PER_NODE:-}

if [[ -n "${TASK_MANIFEST:-}" ]]; then
  data_train="$data/tasks_train.jsonl"
  data_val="$data/tasks_val.jsonl"
  # The four-task calibration manifest has three train rows.  A batch of one
  # keeps the smoke run non-empty; larger studies should set TRAIN_BATCH_SIZE.
  train_batch_size=${TRAIN_BATCH_SIZE:-1}
else
  data_train="$data/train.jsonl"
  data_val="$data/val.jsonl"
  train_batch_size=${TRAIN_BATCH_SIZE:-4}
fi

ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-$train_batch_size}
rollout_n=${ROLLOUT_N:-4}
tensor_parallel_size=${TENSOR_MODEL_PARALLEL_SIZE:-1}
attn_implementation=${ATTN_IMPLEMENTATION:-sdpa}
ray_dashboard=${RAY_DASHBOARD:-false}
max_prompt_length=${MAX_PROMPT_LENGTH:-1200}
max_response_length=${MAX_RESPONSE_LENGTH:-256}
max_model_len=${MAX_MODEL_LEN:-$((max_prompt_length + max_response_length))}
max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-$max_model_len}
max_num_seqs=${MAX_NUM_SEQS:-$((train_batch_size * rollout_n))}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.5}
enforce_eager=${VLLM_ENFORCE_EAGER:-false}
logger=${VERL_LOGGER:-console}
tool_config="$root/model/language_model/config/verl_tools.yaml"

if [[ "${REQUIRE_SCORED_DATA:-1}" != "0" ]]; then
  python3 - "$data_train" "$data_val" <<'PY'
import json
import sys
from pathlib import Path


def valid_contract(row):
    truth = row.get("reward_model", {}).get("ground_truth")
    if isinstance(truth, str):
        return bool(truth.strip())
    if not isinstance(truth, dict):
        return False
    if truth.get("kind") == "exact_text":
        return bool(str(truth.get("expected_output", truth.get("value", ""))).strip())
    verifier = truth.get("verifier")
    valid_verifier = bool(verifier.strip()) if isinstance(verifier, str) else (
        isinstance(verifier, dict) and isinstance(verifier.get("command"), str) and bool(verifier["command"].strip())
    )
    return truth.get("kind") == "environment" and bool(truth.get("task_id")) and valid_verifier


for filename in sys.argv[1:]:
    path = Path(filename)
    if not path.is_file():
        raise SystemExit(f"missing verl data file: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    invalid = [i for i, row in enumerate(rows) if not valid_contract(row)]
    if invalid:
        raise SystemExit(
            f"refusing RL: {path} contains {len(invalid)}/{len(rows)} rows without an executable verifier "
            f"(first row {invalid[0]}). Historical Codex replay rows are SFT/diagnostic data, not RL data."
        )
    if not rows:
        raise SystemExit(f"refusing RL: {path} is empty")
print(f"verified RL data: {', '.join(sys.argv[1:])}")
PY
fi

if [[ -z "$gpus" ]]; then
  gpus=$(python3 -c 'import torch; print(max(1, torch.cuda.device_count()))' 2>/dev/null || echo 1)
fi

export PYTHONPATH="$root:$verl${PYTHONPATH:+:$PYTHONPATH}"
cd "$verl"
exec python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="['$data_train']" \
  data.val_files="['$data_val']" \
  data.return_raw_chat=True \
  data.need_tools_kwargs=True \
  data.tool_config_path="$tool_config" \
  data.train_batch_size="$train_batch_size" \
  data.max_prompt_length="$max_prompt_length" \
  data.max_response_length="$max_response_length" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path="$model" \
  actor_rollout_ref.rollout.name="${ROLLOUT_BACKEND:-vllm}" \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.n="$rollout_n" \
  actor_rollout_ref.rollout.temperature="${TEMPERATURE:-0.7}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$tensor_parallel_size" \
  actor_rollout_ref.rollout.max_model_len="$max_model_len" \
  actor_rollout_ref.rollout.max_num_batched_tokens="$max_num_batched_tokens" \
  actor_rollout_ref.rollout.max_num_seqs="$max_num_seqs" \
  actor_rollout_ref.rollout.gpu_memory_utilization="$gpu_memory_utilization" \
  actor_rollout_ref.rollout.enforce_eager="$enforce_eager" \
  actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$root/model/language_model/config/verl_tools.yaml" \
  actor_rollout_ref.rollout.multi_turn.format="${TOOL_FORMAT:-hermes}" \
  +actor_rollout_ref.model.override_config.attn_implementation="$attn_implementation" \
  actor_rollout_ref.actor.ppo_mini_batch_size="$ppo_mini_batch_size" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}" \
  actor_rollout_ref.actor.ppo_epochs=1 \
  reward.custom_reward_function.path="$root/model/language_model/scripts/verl_reward.py" \
  reward.custom_reward_function.name=compute_score \
  reward.reward_manager.source=importlib \
  reward.reward_manager.name=RetryOnIneligibleRewardManager \
  reward.reward_manager.module.path="$root/model/language_model/scripts/verl_reward_manager.py" \
  trainer.v1.sampler.sync_refill_failed_groups=True \
  trainer.project_name=nanoagent \
  trainer.experiment_name="${EXPERIMENT_NAME:-grpo_codex}" \
  trainer.n_gpus_per_node="$gpus" \
  trainer.nnodes=1 \
  trainer.total_epochs="${TOTAL_EPOCHS:-1}" \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS:-null}" \
  trainer.logger="['$logger']" \
  +ray_kwargs.ray_init.include_dashboard="$ray_dashboard" \
  "$@"
