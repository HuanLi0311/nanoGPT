"""CLI for the four stages and policy-rollout training-data gate."""

from __future__ import annotations

import argparse
import json
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

if __package__:
    from .graph import build_material_graph
    from .schema import fingerprint, read_json, read_jsonl, write_json, write_jsonl
    from .traj_synth import construct_tasks, validate_oracles
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from graph import build_material_graph
    from schema import fingerprint, read_json, read_jsonl, write_json, write_jsonl
    from traj_synth import construct_tasks, validate_oracles


EXCHANGE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>\s*(?:user\n)?<tool_response>\s*(.*?)\s*</tool_response>\s*(?:assistant\n)?", re.S)


def _policy_messages(row: dict[str, Any], task: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(row.get("messages"), list):
        return deepcopy(row["messages"])
    output = row.get("output")
    if not isinstance(output, str):
        raise ValueError("rollout has neither messages nor serialized model output")
    messages = deepcopy(task["prompt"] if isinstance(task["prompt"], list) else [{"role": "user", "content": task["prompt"]}])
    cursor = 0
    for index, match in enumerate(EXCHANGE.finditer(output)):
        prefix = output[cursor:match.start()].strip().removeprefix("assistant\n").strip()
        call = json.loads(match.group(1))
        if not isinstance(call, dict) or not isinstance(call.get("name"), str) or not isinstance(call.get("arguments"), dict):
            raise ValueError("invalid serialized tool call")
        call_id = f"call_{index:04d}"
        messages.append({"role": "assistant", "content": prefix, "tool_calls": [{"id": call_id, "type": "function",
                         "function": {"name": call["name"], "arguments": json.dumps(call["arguments"], ensure_ascii=False)}}]})
        messages.append({"role": "tool", "tool_call_id": call_id, "content": match.group(2).strip()})
        cursor = match.end()
    tail = output[cursor:].strip().removeprefix("assistant\n").strip()
    if tail:
        messages.append({"role": "assistant", "content": tail})
    return messages


def _valid_messages(messages: list[dict[str, Any]]) -> bool:
    calls, results = [], []
    for message in messages:
        if message.get("role") == "assistant":
            calls.extend(call.get("id") for call in message.get("tool_calls", []) if isinstance(call, dict))
        elif message.get("role") == "tool":
            results.append(message.get("tool_call_id"))
    return bool(calls) and all(calls) and all(results) and len(calls) == len(results) and len(set(calls)) == len(calls) and set(calls) == set(results)


def _success(row: dict[str, Any]) -> bool:
    outcome = row.get("outcome") if isinstance(row.get("outcome"), dict) else row
    try:
        score = float(outcome.get("task_success", outcome.get("score", 0)))
    except (TypeError, ValueError):
        return False
    return score > 0 and outcome.get("harness_status") == "healthy" and outcome.get("eligible", True) is not False


def _rollout_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        for row in read_jsonl(path):
            rows.append({**row, "_source": str(path)})
    return rows


def _sft_row(messages: list[dict[str, Any]], task_id: str, model: str, source: str) -> dict[str, Any]:
    trajectory_id = f"policy:{fingerprint([task_id, model, messages])[:24]}"
    split = "test" if int(fingerprint(trajectory_id)[:8], 16) % 10 == 0 else "train"
    tool_calls = sum(len(message.get("tool_calls", [])) for message in messages)
    return {"messages": messages, "data_source": "policy_verified", "trajectory_id": trajectory_id, "split": split,
            "metadata": {"chars": sum(len(str(message.get("content", ""))) for message in messages),
                         "cli_version": "four-stage-synthesis-v1", "cwd": None, "episode_index": None,
                         "id": trajectory_id, "malformed_json": 0, "model_provider": model,
                         "quality_score": 1.0, "signals": ["policy_rollout", "independent_verifier_passed", "harness_healthy"],
                         "source_file": source, "timestamp": None, "tool_calls": tool_calls}}


def _parquet(rows: list[dict[str, Any]], output: Path) -> dict[str, str]:
    if not rows:
        return {}
    import pyarrow as pa
    import pyarrow.parquet as pq
    from model.language_model.scripts.filter import SFT_SCHEMA

    paths = {}
    for split in ("train", "test"):
        selected = [row for row in rows if row["split"] == split]
        if not selected:
            continue
        path = output / f"{split}_sft-four-stage.parquet"
        temporary = path.with_suffix(".parquet.tmp")
        pq.write_table(pa.Table.from_pylist(selected, schema=SFT_SCHEMA), temporary, compression="zstd")
        temporary.replace(path)
        paths[split] = str(path)
    return paths


def prepare(plan: Path, output: Path, *, profile: str, seed: int, count: int | None,
            path_policy: str, weights: dict[str, float] | None = None) -> dict[str, Any]:
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"refusing to overwrite non-empty run directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    stage1 = build_material_graph(plan, output / "stage1", profile=profile, seed=seed, count=count, weights=weights)
    tasks_path = output / "stage2/tasks.jsonl"
    tasks = construct_tasks(Path(stage1["materials"]), tasks_path)
    stage3 = validate_oracles(tasks_path, output / "stage3", seed=seed, policy=path_policy)
    validated = read_jsonl(Path(stage3["validated_tasks"]))
    rl_path = output / "stage4/rl_tasks.jsonl"
    write_jsonl(rl_path, validated)
    report = {"version": "four-stage-synthesis-v1", "profile": profile, "path_policy": path_policy, "seed": seed,
              "stage1": stage1, "stage2": {"tasks": len(tasks), "manifest": str(tasks_path)}, "stage3": stage3,
              "stage4": {"rl_environments": len(validated), "rl_tasks": str(rl_path),
                         "sft_status": "awaiting verified teacher/current-policy rollouts"}}
    write_json(output / "report.json", report)
    return report


def finalize(prepared: Path, rollout_paths: list[Path], *, policy_kind: str, model: str) -> dict[str, Any]:
    tasks = {task["task_id"]: task for task in read_jsonl(prepared / "stage3/validated_tasks.jsonl")}
    accepted, rejected, seen = [], [], set()
    for row in _rollout_rows(rollout_paths):
        outcome = row.get("outcome") if isinstance(row.get("outcome"), dict) else {}
        policy = row.get("policy") if isinstance(row.get("policy"), dict) else {}
        task_id = str(row.get("task_id") or outcome.get("task_id", ""))
        declared = str(policy.get("kind", row.get("model_provider", policy_kind)))
        try:
            if declared in {"programmatic_oracle", "oracle"} or declared != policy_kind:
                raise ValueError(f"invalid policy provenance: {declared}")
            if task_id not in tasks or not _success(row):
                raise ValueError("unknown task or independent verifier did not pass")
            ground_truth = row.get("gts") if isinstance(row.get("gts"), dict) else {}
            if ground_truth.get("task_id", task_id) != task_id or ground_truth.get("verifier_version", tasks[task_id]["verifier_version"]) != tasks[task_id]["verifier_version"]:
                raise ValueError("rollout verifier contract does not match the prepared task")
            messages = _policy_messages(row, tasks[task_id])
            if not _valid_messages(messages):
                raise ValueError("incomplete tool call/result protocol")
            item = _sft_row(messages, task_id, model, row["_source"])
            if item["trajectory_id"] in seen:
                raise ValueError("duplicate trajectory")
            seen.add(item["trajectory_id"])
            accepted.append(item)
        except Exception as error:
            rejected.append({"task_id": task_id or None, "source": row["_source"], "reason": str(error)})
    output = prepared / "stage4"
    write_jsonl(output / "sft.jsonl", accepted)
    write_jsonl(output / "rejected_rollouts.jsonl", rejected)
    result = {"policy_kind": policy_kind, "model": model, "rollouts": len(accepted) + len(rejected),
              "accepted_sft": len(accepted), "rejected": len(rejected), "sft_jsonl": str(output / "sft.jsonl"),
              "sft_parquet": _parquet(accepted, output)}
    write_json(output / "training_data_report.json", result)
    return result


def diagnose(prepared: Path, rollout_paths: list[Path], output: Path, floor: float = 0.05) -> dict[str, Any]:
    tasks = {task["task_id"]: task for task in read_jsonl(prepared / "stage3/validated_tasks.jsonl")}
    scores: dict[str, list[float]] = {task["task_family"]: [] for task in tasks.values()}
    for row in _rollout_rows(rollout_paths):
        policy = row.get("policy") if isinstance(row.get("policy"), dict) else {}
        if policy.get("kind") in {"programmatic_oracle", "oracle"}:
            continue
        task = tasks.get(str(row.get("task_id", "")))
        outcome = row.get("outcome") if isinstance(row.get("outcome"), dict) else row
        if task and outcome.get("harness_status") == "healthy":
            scores.setdefault(task["task_family"], []).append(float(outcome.get("task_success", outcome.get("score", 0))))
    distribution = {family: max(floor, 1 - sum(values) / len(values)) if values else 1.0
                    for family, values in scores.items()}
    if not distribution:
        raise ValueError("no healthy diagnostic rollouts matched prepared tasks")
    result = {"version": "diagnostic-distribution-v1", "distribution": distribution,
              "counts": {family: len(values) for family, values in scores.items()}}
    write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("prepare")
    build.add_argument("plan", type=Path); build.add_argument("output", type=Path)
    build.add_argument("--profile", default="default"); build.add_argument("--seed", type=int, default=0)
    build.add_argument("--count", type=int); build.add_argument("--path-policy", choices=("goal", "uniform"), default="goal")
    build.add_argument("--weights", type=Path)
    finish = commands.add_parser("finalize")
    finish.add_argument("prepared", type=Path); finish.add_argument("rollouts", nargs="+", type=Path)
    finish.add_argument("--policy-kind", choices=("teacher", "current"), required=True); finish.add_argument("--model", required=True)
    diagnostic = commands.add_parser("diagnose")
    diagnostic.add_argument("prepared", type=Path); diagnostic.add_argument("rollouts", nargs="+", type=Path)
    diagnostic.add_argument("--output", type=Path, required=True); diagnostic.add_argument("--floor", type=float, default=0.05)
    args = parser.parse_args()
    if args.command == "prepare":
        weights = read_json(args.weights).get("distribution") if args.weights else None
        result = prepare(args.plan, args.output, profile=args.profile, seed=args.seed, count=args.count,
                         path_policy=args.path_policy, weights=weights)
    elif args.command == "finalize":
        result = finalize(args.prepared, args.rollouts, policy_kind=args.policy_kind, model=args.model)
    else:
        result = diagnose(args.prepared, args.rollouts, args.output, args.floor)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
