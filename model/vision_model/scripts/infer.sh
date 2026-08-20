#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
.venv/bin/python -m model.vision_model.infer "$@"
