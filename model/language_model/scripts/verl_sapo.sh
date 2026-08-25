#!/usr/bin/env bash
set -euo pipefail
ALGORITHM=sapo exec "$(cd "$(dirname "$0")" && pwd)/verl_grpo.sh" "$@"
