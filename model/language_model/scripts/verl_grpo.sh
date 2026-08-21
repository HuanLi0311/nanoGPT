#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/../../.." && pwd)
verl="$root/third_party/verl"
data="$root/model/language_model/data/post_train/verl"
model=${MODEL_PATH:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}
gpus=${NGPUS_PER_NODE:-$(python3 -c 'import torch; print(max(1, torch.cuda.device_count()))')}
train_batch_size=${TRAIN_BATCH_SIZE:-4}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-$train_batch_size}
tensor_parallel_size=${TENSOR_MODEL_PARALLEL_SIZE:-1}
attn_implementation=${ATTN_IMPLEMENTATION:-sdpa}
ray_dashboard=${RAY_DASHBOARD:-false}
cd "$verl"
exec python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="['$data/train.jsonl']" \
  data.val_files="['$data/val.jsonl']" \
  data.train_batch_size="$train_batch_size" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH:-1200}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH:-256}" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path="$model" \
  actor_rollout_ref.rollout.name="${ROLLOUT_BACKEND:-vllm}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N:-4}" \
  actor_rollout_ref.rollout.temperature="${TEMPERATURE:-0.7}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$tensor_parallel_size" \
  +actor_rollout_ref.model.override_config.attn_implementation="$attn_implementation" \
  actor_rollout_ref.actor.ppo_mini_batch_size="$ppo_mini_batch_size" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}" \
  actor_rollout_ref.actor.ppo_epochs=1 \
  reward.custom_reward_function.path="$root/model/language_model/scripts/verl_reward.py" \
  reward.custom_reward_function.name=compute_score \
  trainer.project_name=nanoagent \
  trainer.experiment_name="${EXPERIMENT_NAME:-grpo_codex}" \
  trainer.n_gpus_per_node="$gpus" \
  trainer.nnodes=1 \
  trainer.total_epochs="${TOTAL_EPOCHS:-1}" \
  +ray_kwargs.ray_init.include_dashboard="$ray_dashboard" \
  "$@"
