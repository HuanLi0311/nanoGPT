#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/../../.." && pwd)
python="$root/.venv/bin/python"
source_dir=${1:-"$root/model/language_model/data/post_train/data/rendered/sft"}
out=${2:-"$root/model/language_model/data/post_train/verl"}
mkdir -p "$out"
if [[ -n "${TASK_MANIFEST:-}" ]]; then
    "$python" "$root/model/language_model/scripts/prepare_verl_tasks.py" "$TASK_MANIFEST" "$out/tasks.jsonl"
    "$python" - "$out/tasks.jsonl" "$out" <<'PY'
import json, sys
rows = [json.loads(x) for x in open(sys.argv[1], encoding="utf-8") if x.strip()]
if len(rows) < 2:
    raise SystemExit("need at least two verified tasks to create non-empty train and val splits")
cut = min(len(rows) - 1, max(1, int(len(rows) * .75)))
for name, part in [("tasks_train", rows[:cut]), ("tasks_val", rows[cut:])]:
    with open(f"{sys.argv[2]}/{name}.jsonl", "w", encoding="utf-8") as handle:
        for row in part:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"prepared {len(rows)} verified task rows")
PY
    exit 0
fi
"$python" "$root/model/language_model/scripts/prepare_verl_data.py" "$source_dir/codex_train.parquet" "$out/train.parquet"
"$python" "$root/model/language_model/scripts/prepare_verl_data.py" "$source_dir/codex_test.parquet" "$out/val.parquet"
