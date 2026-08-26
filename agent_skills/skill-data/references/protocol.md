# Data protocol

`make_text_tasks.py` emits JSONL records with `task_id`, `split`, `example_id`,
and integer `input_ids`. Task token ranges are disjoint, and train/eval records
are generated from different deterministic streams. Keep the JSONL and its
manifest immutable after a run begins; record its SHA-256 in the run manifest.

Example:

```bash
python3 scripts/make_text_tasks.py --output runs/r01/data.jsonl \
  --manifest runs/r01/data_manifest.json --tasks 5 --examples-per-task 64 \
  --eval-examples-per-task 32 --length 64 --vocab-size 32000 --seed 1234
```
