#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$root"
exec .venv/bin/python -m model.language_model.scripts.infer "$@"
