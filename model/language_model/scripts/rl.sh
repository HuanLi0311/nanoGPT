#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$root"
python=$("$root/scripts/nanoagent_python.sh")
exec "$python" -m model.language_model.scripts.rl "${1:-model/language_model/config/rl.yaml}"
