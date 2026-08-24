#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$root"
python=$("$root/scripts/nanoagent_python.sh")
"$python" -m model.vision_model.infer "$@"
