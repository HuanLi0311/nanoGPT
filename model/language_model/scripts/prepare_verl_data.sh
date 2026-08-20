#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/../../.." && pwd)
source_dir=${1:-"$root/model/language_model/data/post_train/data/filtered"}
out=${2:-"$root/model/language_model/data/post_train/verl"}
python3 "$root/model/language_model/scripts/prepare_verl_data.py" "$source_dir" "$out/all.jsonl"
python3 - "$out/all.jsonl" "$out" <<'PY'
import json, sys
rows = [json.loads(x) for x in open(sys.argv[1], encoding="utf-8") if x.strip()]
cut = max(1, int(len(rows) * .95))
for name, part in [("train", rows[:cut]), ("val", rows[cut:])]:
    with open(f"{sys.argv[2]}/{name}.jsonl", "w", encoding="utf-8") as handle:
        for row in part:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"prepared {len(rows)} rows")
PY
