#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/../../.." && pwd)
python=$("$root/scripts/nanoagent_python.sh")
verl="$root/third_party/verl"
data="$root/model/language_model/data/post_train/verl"
# Student default matches sft.sh's Qwen3 base and output path. DeepSeek is the
# external teacher/calibration model, not the policy in this launcher.
model=${MODEL_PATH:-"$root/model/language_model/checkpoints/qwen/Qwen3-8B"}
algorithm=${ALGORITHM:-grpo}
gpus=${NGPUS_PER_NODE:-}

if [[ -z "$gpus" ]]; then
  gpus=$("$python" -c 'import torch; print(max(1, torch.cuda.device_count()))' 2>/dev/null || echo 1)
fi
[[ "$gpus" =~ ^[1-9][0-9]*$ ]] || { echo "NGPUS_PER_NODE must be a positive integer: $gpus" >&2; exit 2; }

rollout_n=${ROLLOUT_N:-4}
[[ "$rollout_n" =~ ^[1-9][0-9]*$ ]] || { echo "ROLLOUT_N must be a positive integer: $rollout_n" >&2; exit 2; }
batch_unit=$("$python" - "$gpus" "$rollout_n" <<'PY'
import math
import sys

gpus, rollout_n = map(int, sys.argv[1:])
print(gpus // math.gcd(gpus, rollout_n))
PY
)

if [[ -n "${TENSOR_MODEL_PARALLEL_SIZE:-}" ]]; then
  tensor_parallel_size=$TENSOR_MODEL_PARALLEL_SIZE
elif (( gpus >= 4 )); then
  # One Qwen3-8B replica per four A100s keeps colocated vLLM startup bounded;
  # override for a different model/topology.
  tensor_parallel_size=4
else
  tensor_parallel_size=1
fi
[[ "$tensor_parallel_size" =~ ^[1-9][0-9]*$ ]] || {
  echo "TENSOR_MODEL_PARALLEL_SIZE must be a positive integer: $tensor_parallel_size" >&2
  exit 2
}
(( gpus % tensor_parallel_size == 0 )) || {
  echo "NGPUS_PER_NODE ($gpus) must be divisible by TENSOR_MODEL_PARALLEL_SIZE ($tensor_parallel_size)" >&2
  exit 2
}

worker_slots=$(( gpus < 8 ? gpus : 8 ))

if [[ -z "${TASK_MANIFEST:-}" ]]; then
  # The checked-in parquet is historical Codex replay without a verifier. RL
  # defaults to the small executable manifest so an accidental launch cannot
  # train on censored rows.
  TASK_MANIFEST="$root/agent/tasks/harness_smoke.jsonl"
fi

if [[ -n "${TASK_MANIFEST:-}" ]]; then
  data_train="$data/tasks_train.jsonl"
  data_val="$data/tasks_val.jsonl"
  TASK_MANIFEST="$TASK_MANIFEST" "$root/model/language_model/scripts/prepare_verl_data.sh" >/dev/null
  minimum_train_batch_size=1
else
  data_train="$data/train.parquet"
  data_val="$data/val.parquet"
  minimum_train_batch_size=4
fi

if [[ -n "${TRAIN_BATCH_SIZE:-}" ]]; then
  train_batch_size=$TRAIN_BATCH_SIZE
else
  # Verl's FSDP validator requires train_batch_size * rollout_n to divide the
  # data-parallel world size. Pick the smallest legal batch at or above the
  # smoke/study default instead of failing later inside Hydra validation.
  train_batch_size=$("$python" - "$batch_unit" "$minimum_train_batch_size" <<'PY'
import sys

unit, minimum = map(int, sys.argv[1:])
print(((minimum + unit - 1) // unit) * unit)
PY
  )
fi
[[ "$train_batch_size" =~ ^[1-9][0-9]*$ ]] || {
  echo "TRAIN_BATCH_SIZE must be a positive integer: $train_batch_size" >&2
  exit 2
}
(( (train_batch_size * rollout_n) % gpus == 0 )) || {
  echo "TRAIN_BATCH_SIZE ($train_batch_size) * ROLLOUT_N ($rollout_n) must be divisible by NGPUS_PER_NODE ($gpus)" >&2
  echo "Use a TRAIN_BATCH_SIZE that is a multiple of $batch_unit (or omit it for automatic sizing)." >&2
  exit 2
}

ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-$train_batch_size}
attn_implementation=${ATTN_IMPLEMENTATION:-sdpa}
ray_dashboard=${RAY_DASHBOARD:-false}
transfer_queue_units=${TRANSFER_QUEUE_UNITS:-$worker_slots}
agent_loop_num_workers=${AGENT_LOOP_NUM_WORKERS:-$worker_slots}
reward_num_workers=${REWARD_NUM_WORKERS:-$worker_slots}
dataloader_num_workers=${DATALOADER_NUM_WORKERS:-$worker_slots}
max_prompt_length=${MAX_PROMPT_LENGTH:-1200}
max_response_length=${MAX_RESPONSE_LENGTH:-256}
max_model_len=${MAX_MODEL_LEN:-$((max_prompt_length + max_response_length))}
max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-$max_model_len}
max_num_seqs=${MAX_NUM_SEQS:-$((train_batch_size * rollout_n))}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.4}
enforce_eager=${VLLM_ENFORCE_EAGER:-false}
overlong_buffer_len=${OVERLONG_BUFFER_LEN:-$((max_response_length / 4))}
(( overlong_buffer_len > 0 )) || overlong_buffer_len=1
logger=${VERL_LOGGER:-console}
save_freq=${SAVE_FREQ:-100}
test_freq=${TEST_FREQ:-100}
tool_config="$root/model/language_model/config/verl_tools.yaml"
# Colocated FSDP and vLLM otherwise keep two Qwen copies resident during
# server startup. CPU parameter offload leaves the GPU budget to the rollout;
# callers can disable it for a disaggregated deployment.
actor_param_offload=${ACTOR_PARAM_OFFLOAD:-true}
ref_param_offload=${REF_PARAM_OFFLOAD:-true}
actor_optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD:-true}
actor_optimizer=${ACTOR_OPTIMIZER:-AdamW}
actor_optimizer_impl=${ACTOR_OPTIMIZER_IMPL:-}
actor_optimizer_config=${ACTOR_OPTIMIZER_CONFIG:-}
if [[ -z "$actor_optimizer_impl" ]]; then
  case "$actor_optimizer" in
    Adafactor) actor_optimizer_impl=transformers.optimization ;;
    *) actor_optimizer_impl=torch.optim ;;
  esac
fi
if [[ "$actor_optimizer" == "Adafactor" && -z "$actor_optimizer_config" ]]; then
  actor_optimizer_config='{relative_step:false,scale_parameter:false,warmup_init:false}'
fi
if [[ "$actor_optimizer" == "AdamW" && -z "$actor_optimizer_config" ]]; then
  # ponytail: fused AdamW keeps the optimizer step's temporary peak low and
  # is faster on CUDA; fall back to {foreach:false} if a target torch build
  # lacks the fused kernel.
  actor_optimizer_config='{fused:true}'
fi

[[ -x "$python" ]] || { echo "missing Python environment: $python" >&2; exit 1; }

if [[ "${ALLOW_LEGACY_VERL_TOOLS:-0}" != "1" ]]; then
  "$python" - "$tool_config" <<'PY'
import re
import sys
from pathlib import Path

configured = set(re.findall(r"^\s+name:\s*([A-Za-z_][A-Za-z0-9_]*)\s*$", Path(sys.argv[1]).read_text(encoding="utf-8"), re.MULTILINE))
expected = {"exec_command", "apply_patch"}
missing = sorted(expected - configured)
exposed_verifier = "verify_task" in configured
if missing or exposed_verifier:
    details = []
    if missing:
        details.append(f"missing environment tools: {', '.join(missing)}")
    if exposed_verifier:
        details.append("verify_task is exposed as a model tool")
    raise SystemExit(
        "refusing RL with an invalid NanoAgent tool profile (" + "; ".join(details) + ")."
    )
PY
fi

if [[ "${REQUIRE_SCORED_DATA:-1}" != "0" ]]; then
  "$python" - "$data_train" "$data_val" <<'PY'
import json
import sys
from pathlib import Path
import pyarrow.parquet as pq


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
    rows = pq.read_table(path).to_pylist() if path.suffix == ".parquet" else [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
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

# ponytail: Verl's default TransferQueue reserves nine CPU slots; leave room
# for its storage actors and the per-GPU placement group without using all
# 180 host CPUs. Override for larger experiments with RAY_NUM_CPUS.
ray_cpus=${RAY_NUM_CPUS:-$((gpus * 4 + 8))}

export PYTHONPATH="$root:$verl${PYTHONPATH:+:$PYTHONPATH}"
# vLLM's multiprocessing backend is fork-unsafe inside a Ray actor.  Spawn is
# the compatible default for this colocated Qwen3 launcher; callers can still
# override it for a different backend.
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
# vLLM 0.8.x otherwise auto-selects V0 when create_engine_config runs in the
# Ray server actor's background thread; Verl's async server uses the V1 API.
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
# vLLM 0.8.x's V1 engine otherwise creates a second core process inside the
# Ray server actor.  TP=1 does not need it; disabling it avoids a silent actor
# exit while preserving the same AsyncLLM interface.
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"
# TransferQueue actors import torch concurrently; preload the stdlib modules
# that torch reaches through different import paths in Python 3.11.
export NANOAGENT_RAY_WORKER_SETUP_HOOK="${NANOAGENT_RAY_WORKER_SETUP_HOOK:-agent.verl_adapter.loop_adapter.preload_worker}"
# vLLM 0.8.x inspects model classes in a subprocess; concurrent replica
# startup races in Python's codec imports, so serialize only initialization.
export VERL_SERIALIZE_ROLLOUT_INIT="${VERL_SERIALIZE_ROLLOUT_INIT:-1}"
cd "$verl"
args=(
  "algorithm.adv_estimator=grpo"
  "algorithm.use_kl_in_reward=False"
  "data.train_files=['$data_train']"
  "data.val_files=['$data_val']"
  "data.return_raw_chat=True"
  "+data.need_tools_kwargs=True"
  "data.tool_config_path=$tool_config"
  "data.train_batch_size=$train_batch_size"
  "data.max_prompt_length=$max_prompt_length"
  "data.max_response_length=$max_response_length"
  "data.filter_overlong_prompts=True"
  "data.dataloader_num_workers=$dataloader_num_workers"
  "data.truncation=error"
  "actor_rollout_ref.model.path=$model"
  "actor_rollout_ref.rollout.name=${ROLLOUT_BACKEND:-vllm}"
  "actor_rollout_ref.rollout.mode=async"
  "actor_rollout_ref.rollout.n=$rollout_n"
  "actor_rollout_ref.rollout.temperature=${TEMPERATURE:-0.7}"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_parallel_size"
  "actor_rollout_ref.rollout.max_model_len=$max_model_len"
  "actor_rollout_ref.rollout.max_num_batched_tokens=$max_num_batched_tokens"
  "actor_rollout_ref.rollout.max_num_seqs=$max_num_seqs"
  "actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization"
  "actor_rollout_ref.rollout.enforce_eager=$enforce_eager"
  "actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent"
  "actor_rollout_ref.rollout.multi_turn.enable=True"
  "actor_rollout_ref.rollout.agent.num_workers=$agent_loop_num_workers"
  "actor_rollout_ref.rollout.multi_turn.tool_config_path=$root/model/language_model/config/verl_tools.yaml"
  "actor_rollout_ref.rollout.multi_turn.format=${TOOL_FORMAT:-hermes}"
  "+actor_rollout_ref.model.override_config.attn_implementation=$attn_implementation"
  "actor_rollout_ref.actor.fsdp_config.param_offload=$actor_param_offload"
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=$actor_optimizer_offload"
  "actor_rollout_ref.ref.fsdp_config.param_offload=$ref_param_offload"
  "actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
  "actor_rollout_ref.actor.optim.optimizer=$actor_optimizer"
  "actor_rollout_ref.actor.optim.optimizer_impl=$actor_optimizer_impl"
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
  "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
  "actor_rollout_ref.actor.ppo_epochs=1"
  "reward.custom_reward_function.path=$root/model/language_model/scripts/verl_reward.py"
  "reward.custom_reward_function.name=compute_score"
  "reward.reward_manager.source=importlib"
  "reward.reward_manager.name=RetryOnIneligibleRewardManager"
  "reward.reward_manager.module.path=$root/model/language_model/scripts/verl_reward_manager.py"
  "reward.num_workers=$reward_num_workers"
  "transfer_queue.backend.SimpleStorage.num_data_storage_units=$transfer_queue_units"
  "trainer.v1.sampler.sync_refill_failed_groups=True"
  "trainer.project_name=nanoagent"
  "trainer.experiment_name=${EXPERIMENT_NAME:-${algorithm}_qwen3_8b}"
  "trainer.n_gpus_per_node=$gpus"
  "trainer.nnodes=1"
  "trainer.total_epochs=${TOTAL_EPOCHS:-1}"
  "trainer.total_training_steps=${TOTAL_TRAINING_STEPS:-null}"
  "trainer.save_freq=$save_freq"
  "trainer.test_freq=$test_freq"
  "trainer.logger=['$logger']"
  "ray_kwargs.ray_init.num_cpus=$ray_cpus"
  "+ray_kwargs.ray_init.include_dashboard=$ray_dashboard"
)

case "$algorithm" in
  grpo) ;;
  sapo)
    args+=(
      "actor_rollout_ref.actor.policy_loss.loss_mode=sapo"
      "actor_rollout_ref.actor.tau_pos=${TAU_POS:-1.0}"
      "actor_rollout_ref.actor.tau_neg=${TAU_NEG:-1.05}"
    )
    ;;
  dapo)
    args+=(
      "algorithm.filter_groups.enable=True"
      "algorithm.filter_groups.metric=seq_final_reward"
      "algorithm.norm_adv_by_std_in_grpo=False"
      "actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW:-0.2}"
      "actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH:-0.28}"
      "actor_rollout_ref.actor.clip_ratio_c=${CLIP_RATIO_C:-10.0}"
      "+reward.reward_kwargs.overlong_buffer_cfg.enable=${OVERLONG_BUFFER_ENABLE:-True}"
      "+reward.reward_kwargs.overlong_buffer_cfg.len=$overlong_buffer_len"
      "+reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${OVERLONG_PENALTY_FACTOR:-1.0}"
      "+reward.reward_kwargs.overlong_buffer_cfg.log=False"
      "+reward.reward_kwargs.max_resp_len=$max_response_length"
    )
    ;;
  *)
    echo "unknown ALGORITHM=$algorithm (expected grpo, sapo, or dapo)" >&2
    exit 2
    ;;
esac

if [[ -n "$actor_optimizer_config" ]]; then
  args+=("++actor_rollout_ref.actor.optim.override_optimizer_config=$actor_optimizer_config")
fi

# Preload Python 3.11 modules before torch/Ray's import graph starts.  This
# keeps the entrypoint native (the same main_ppo module) while avoiding the
# intermittent partially-initialized stdlib packages seen on air-node-02.
exec "$python" -c 'import email.feedparser, email.parser, multiprocessing.context, runpy, unittest.mock, unittest.result; runpy.run_module("verl.trainer.main_ppo", run_name="__main__")' "${args[@]}" "$@"
