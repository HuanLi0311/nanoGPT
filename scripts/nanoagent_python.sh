#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${PYTHON_BIN:-}" ]]; then
  python="$PYTHON_BIN"
elif [[ "${CONDA_DEFAULT_ENV:-}" == "${NANOAGENT_ENV:-nanoagent}" && -n "${CONDA_PREFIX:-}" ]]; then
  python="$CONDA_PREFIX/bin/python"
else
  echo "activate conda env ${NANOAGENT_ENV:-nanoagent} or set PYTHON_BIN" >&2
  exit 1
fi

if [[ "$python" != */* ]]; then
  python="$(command -v "$python" || true)"
fi
[[ -x "$python" ]] || { echo "missing Python executable: $python" >&2; exit 1; }
printf '%s\n' "$python"
