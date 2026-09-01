#!/usr/bin/env python3
"""Aggregate raw rank-1 probe envelopes without changing them."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _numeric_values(records):
    keys = sorted({key for record in records for key, value in record.items() if isinstance(value, (int, float)) and not isinstance(value, bool)})
    result = {}
    for key in keys:
        values = [float(record[key]) for record in records if isinstance(record.get(key), (int, float)) and not isinstance(record.get(key), bool)]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1) if len(values) > 1 else 0.0
        result[key] = {"mean": mean, "std": math.sqrt(variance), "count": len(values)}
    return result


def aggregate(paths):
    envelopes = []
    failures = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("status") != "ok":
            failures.append({"path": str(Path(path).resolve()), "error": payload.get("error")})
            continue
        envelopes.append(payload)
    groups = {}
    for envelope in envelopes:
        for record in envelope.get("results", []):
            group_key = tuple((key, record.get(key)) for key in ("model_size_m", "mask_condition", "mask_probability", "parameter", "sequence_length", "sample_count", "test_sample_count", "loss_mode", "evaluation"))
            groups.setdefault(group_key, []).append(record)
    aggregate_records = []
    for group_key, records in sorted(groups.items(), key=lambda item: repr(item[0])):
        aggregate_records.append({
            "group": {key: value for key, value in group_key},
            "metrics": _numeric_values(records),
            "raw_record_count": len(records),
            "seeds": sorted({record.get("seed") for record in records}),
        })
    return {
        "schema_version": 1,
        "status": "ok" if envelopes else "failed",
        "experiment": "dllm_rank1_fisher_aggregate",
        "input_files": [str(Path(path).resolve()) for path in paths],
        "successful_envelopes": len(envelopes),
        "failures": failures,
        "groups": aggregate_records,
    }


def self_check():
    result = _numeric_values([{"x": 1.0}, {"x": 3.0}])
    assert result["x"]["mean"] == 2.0 and result["x"]["count"] == 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    if args.self_check:
        self_check()
        print(json.dumps({"self_check": "ok"}))
        return 0
    if not args.inputs:
        parser.error("at least one raw JSON input is required")
    result = aggregate(args.inputs)
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
