#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$root"
python=$("$root/scripts/nanoagent_python.sh")
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_ETAG_TIMEOUT=60
source=$("$python" -c 'import yaml; print(yaml.safe_load(open("model/vision_model/config/finetune.yaml"))["model"]["source"])')
target=$("$python" -c 'import yaml; print(yaml.safe_load(open("model/vision_model/config/finetune.yaml"))["model"]["name"])')
hf download "$source" config.json preprocessor_config.json model.safetensors --local-dir "$target"
"$python" -m model.vision_model.scripts.finetune "$@"
