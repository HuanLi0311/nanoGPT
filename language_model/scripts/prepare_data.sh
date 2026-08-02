#!/usr/bin/env bash
set -euo pipefail

stages=${1:?Usage: prepare_data.sh STAGE[,STAGE] CONFIG}
config=${2:?Usage: prepare_data.sh STAGE[,STAGE] CONFIG}
cd "$(dirname "$0")/../.."

.venv/bin/python - "$stages" "$config" <<'PY'
import sys
from pathlib import Path

import yaml

from language_model.scripts import prepare_data

data = yaml.safe_load(Path(sys.argv[2]).read_text(encoding="utf-8"))["data"]
for stage in sys.argv[1].split(","):
    getattr(prepare_data, stage)(data)
PY
