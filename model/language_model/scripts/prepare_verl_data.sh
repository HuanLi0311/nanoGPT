#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/../../.." && pwd)
source_dir=${1:-"$root/model/language_model/data/post_train/data/filtered"}
out=${2:-"$root/model/language_model/data/post_train/verl"}
mkdir -p "$out"
if [[ -n "${TASK_MANIFEST:-}" ]]; then
    python3 "$root/model/language_model/scripts/prepare_verl_tasks.py" "$TASK_MANIFEST" "$out/tasks.jsonl"
    python3 - "$out/tasks.jsonl" "$out" <<'PY'
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
python3 "$root/model/language_model/scripts/prepare_verl_data.py" "$source_dir" "$out/all.jsonl"
python3 - "$out/all.jsonl" "$out" <<'PY'
import json, sys
rows = [json.loads(x) for x in open(sys.argv[1], encoding="utf-8") if x.strip()]
if len(rows) < 2:
    raise SystemExit("need at least two rows to create non-empty train and val splits")
cut = min(len(rows) - 1, max(1, int(len(rows) * .95)))
for name, part in [("train", rows[:cut]), ("val", rows[cut:])]:
    with open(f"{sys.argv[2]}/{name}.jsonl", "w", encoding="utf-8") as handle:
        for row in part:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"prepared {len(rows)} rows")
PY
