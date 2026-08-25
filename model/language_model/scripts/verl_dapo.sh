#!/usr/bin/env bash
set -euo pipefail
ALGORITHM=dapo exec "$(cd "$(dirname "$0")" && pwd)/verl_grpo.sh" "$@"
