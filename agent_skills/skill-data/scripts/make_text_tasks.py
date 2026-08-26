#!/usr/bin/env python3
"""Create deterministic, disjoint integer-token continual-learning tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import tempfile
from pathlib import Path


def make_records(tasks: int, train_per_task: int, eval_per_task: int, length: int, vocab_size: int, seed: int):
    if tasks < 1 or train_per_task < 1 or eval_per_task < 1 or length < 1:
        raise ValueError("tasks, examples, and length must be positive")
    if vocab_size < tasks * 2:
        raise ValueError("vocab_size must leave each task a non-empty token range")
    width = vocab_size // tasks
    records = []
    for task in range(tasks):
        low = task * width
        high = vocab_size if task == tasks - 1 else (task + 1) * width
        for split, count, split_offset in (("train", train_per_task, 0), ("eval", eval_per_task, 1)):
            for index in range(count):
                rng = random.Random(seed + task * 1_000_003 + split_offset * 10_000_019 + index)
                tokens = [rng.randrange(low, high) for _ in range(length)]
                records.append(
                    {
                        "task_id": task,
                        "split": split,
                        "example_id": "%s-%d-%d" % (split, task, index),
                        "input_ids": tokens,
                        "seed": seed,
                        "vocab_size": vocab_size,
                    }
                )
    return records


def write_tasks(records, output: Path, manifest: Path | None = None):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    result = {
        "schema_version": 1,
        "path": str(output.resolve()),
        "sha256": digest,
        "records": len(records),
        "tasks": sorted({record["task_id"] for record in records}),
        "train_records": sum(record["split"] == "train" for record in records),
        "eval_records": sum(record["split"] == "eval" for record in records),
        "seed": records[0]["seed"] if records else None,
    }
    if manifest:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def self_check():
    records = make_records(3, 4, 3, 8, 30, 7)
    assert len(records) == 21
    for record in records:
        assert len(record["input_ids"]) == 8
        assert 0 <= min(record["input_ids"]) < max(record["input_ids"]) < 30
    task_ranges = []
    for task in range(3):
        values = [token for record in records if record["task_id"] == task for token in record["input_ids"]]
        task_ranges.append((min(values), max(values)))
    assert task_ranges == [(0, 9), (10, 19), (20, 29)]
    with tempfile.TemporaryDirectory(prefix="skill-data-") as directory:
        result = write_tasks(records, Path(directory) / "tasks.jsonl")
        assert result["records"] == 21 and len(result["sha256"]) == 64


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--tasks", type=int, default=5)
    parser.add_argument("--examples-per-task", type=int, default=64)
    parser.add_argument("--eval-examples-per-task", type=int, default=32)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    if args.self_check:
        self_check()
        print(json.dumps({"self_check": "ok"}))
        return 0
    if not args.output:
        parser.error("--output is required unless --self-check is used")
    records = make_records(args.tasks, args.examples_per_task, args.eval_examples_per_task, args.length, args.vocab_size, args.seed)
    result = write_tasks(records, args.output, args.manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
