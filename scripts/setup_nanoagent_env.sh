#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
conda=${CONDA_EXE:-/usr/local/miniconda3/bin/conda}
env=${NANOAGENT_ENV:-nanoagent}

[[ -x "$conda" ]] || { echo "missing conda executable: $conda" >&2; exit 1; }

if ! "$conda" run --name "$env" python --version >/dev/null 2>&1; then
  "$conda" create --name "$env" python=3.11 pip -y
fi
"$conda" run --name "$env" python -m pip install --upgrade pip
"$conda" run --name "$env" python -m pip install --upgrade-strategy only-if-needed \
  -r "$root/requirements.txt" -r "$root/third_party/verl/requirements.txt"
"$conda" run --name "$env" python -m pip install --no-deps -e "$root/third_party/verl"
"$conda" run --name "$env" python -m pip check

cd "$root"
"$conda" run --name "$env" env PYTHONPATH="$root:$root/third_party/verl" \
  python model/language_model/scripts/check_sft_environment.py --allow-no-cuda
