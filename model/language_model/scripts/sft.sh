#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/../../.." && pwd)
python="$root/.venv/bin/python"
verl="$root/third_party/verl"
data="$root/model/language_model/data/post_train/data/rendered/sft"
model=${MODEL_PATH:-"$root/model/language_model/checkpoints/qwen/Qwen3-8B-Base"}
save_path=${SAVE_PATH:-"$root/logs/Qwen3-8B-Base-sft"}
workers=${NPROC_PER_NODE:-$("$python" -c 'import torch; print(torch.cuda.device_count())')}
master_addr=${MASTER_ADDR:-127.0.0.1}
master_port=${MASTER_PORT:-29501}
attn_implementation=${ATTN_IMPLEMENTATION:-sdpa}

train_shards=("$data"/train_sft-*.parquet "$data"/codex_train.parquet)
val_shards=("$data"/test_sft-*.parquet "$data"/codex_test.parquet)

for path in "$python" "$verl" "$model" "${train_shards[@]}" "${val_shards[@]}"; do
  [[ -e "$path" ]] || { echo "missing required path: $path" >&2; exit 1; }
done
(( workers > 0 )) || { echo "no CUDA devices found; set NPROC_PER_NODE explicitly" >&2; exit 1; }

hydra_list() {
  local IFS=,
  printf '[%s]' "$*"
}

export PYTHONPATH="$root:$verl${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/nanogpt_pycache}"
cd "$verl"
"$python" -c 'import verl.trainer.sft_trainer'

# ponytail: the no-padding FSDP path truncates rows past 8192 tokens on the right;
# add a dataset preprocessing policy if preserving trailing turns becomes necessary.
exec "$python" -m torch.distributed.run --master_addr="$master_addr" --master_port="$master_port" --nproc_per_node="$workers" \
  -m verl.trainer.sft_trainer \
  data.train_files="$(hydra_list "${train_shards[@]}")" \
  data.val_files="$(hydra_list "${val_shards[@]}")" \
  data.messages_key=messages \
  data.train_batch_size="${TRAIN_BATCH_SIZE:-64}" \
  data.micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU:-1}" \
  data.max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU:-8192}" \
  data.max_length="${MAX_LENGTH:-8192}" \
  data.truncation=right \
  data.ignore_input_ids_mismatch=True \
  optim.lr="${LEARNING_RATE:-1e-5}" \
  optim.weight_decay="${WEIGHT_DECAY:-0.1}" \
  optim.lr_warmup_steps_ratio="${WARMUP_RATIO:-0.03}" \
  optim.lr_scheduler_type=cosine \
  engine=fsdp \
  model.path="$model" \
  +model.override_config.attn_implementation="$attn_implementation" \
  model.use_remove_padding=true \
  trainer.default_local_dir="$save_path" \
  trainer.project_name=nanoagent-sft \
  trainer.experiment_name="${EXPERIMENT_NAME:-qwen3-8b-sft}" \
  trainer.logger='["console"]' \
  trainer.total_epochs="${EPOCHS:-1}" \
  trainer.save_freq=after_each_epoch \
  trainer.test_freq=after_each_epoch \
  trainer.resume_mode=disable \
  trainer.n_gpus_per_node="$workers" \
  "$@"
