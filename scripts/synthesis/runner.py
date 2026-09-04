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
    from .graph import build_material_graph, expand_knowledge_graph
    from .schema import fingerprint, iter_jsonl, read_json, read_jsonl, write_json, write_jsonl
    from .traj_synth import construct_tasks, validate_oracles
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from graph import build_material_graph, expand_knowledge_graph
    from schema import fingerprint, iter_jsonl, read_json, read_jsonl, write_json, write_jsonl
    from traj_synth import construct_tasks, validate_oracles


EXCHANGE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>\s*(?:user\n)?<tool_response>\s*(.*?)\s*</tool_response>\s*(?:assistant\n)?", re.S)


def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = deepcopy(messages)
    for message in normalized:
        message["content"] = str(message.get("content") or "")
        for call in message.get("tool_calls", []):
            function = call.get("function", {})
            arguments = function.get("arguments", "{}")
            if isinstance(arguments, dict):
                function["arguments"] = json.dumps(arguments, ensure_ascii=False)
            elif not isinstance(arguments, str) or not isinstance(json.loads(arguments), dict):
                raise ValueError("tool arguments must encode a JSON object")
    return normalized


def _policy_messages(row: dict[str, Any], task: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(row.get("messages"), list):
        return _normalize_messages(row["messages"])
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
    return _normalize_messages(messages)


def _valid_messages(messages: list[dict[str, Any]], allowed_tools: list[str]) -> bool:
    calls, results = [], []
    for message in messages:
        if message.get("role") == "assistant":
            for call in message.get("tool_calls", []):
                if not isinstance(call, dict) or call.get("function", {}).get("name") not in allowed_tools:
                    return False
                calls.append(call.get("id"))
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
                         "cli_version": "k3-agentworld-synthesis-v1", "cwd": None, "episode_index": None,
                         "id": trajectory_id, "malformed_json": 0, "model_provider": model,
                         "quality_score": 1.0, "signals": ["policy_rollout", "independent_verifier_passed", "harness_healthy"],
                         "source_file": source, "timestamp": None, "tool_calls": tool_calls}}


def _parquet(rows: list[dict[str, Any]], output: Path) -> dict[str, str]:
    for split in ("train", "test"):
        (output / f"{split}_sft-four-stage.parquet").unlink(missing_ok=True)
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


def _write_run_report(output: Path, report: dict[str, Any]) -> None:
    counts = report["counts_before_verifier"]
    lines = ["# Kimi K3 → Agent-World synthesis report", "",
             "Counts below are generated candidates before verifier filtering.", "",
             "| Stage | Method | Count |", "|---|---|---:|",
             f"| 1 | Kimi K3 domain/material graph | {counts['stage1_unique_materials']:,} |",
             f"| 2 | Agent-World task/environment construction | {counts['stage2_candidate_tasks']:,} |",
             f"| 3 | Agent-World graph trajectories | {counts['stage3_candidate_trajectories']:,} |",
             "", f"Terminal candidate trajectories: **{counts['stage3_candidate_trajectories']:,}**", "",
             "## Stage 1 domains", "", "| Domain | Target | Realized |", "|---|---:|---:|"]
    target = report["stage1"]["target_domain_distribution"]
    realized = report["stage1"]["domain_distribution"]
    lines.extend(f"| {domain} | {target[domain]:,} | {realized.get(domain, 0):,} |" for domain in target)
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare(plan: Path, output: Path, *, profile: str, seed: int, count: int | None,
            path_policy: str, weights: dict[str, float] | None = None,
            tasks_per_material: int | None = None,
            trajectories_per_task: int | None = None) -> dict[str, Any]:
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"refusing to overwrite non-empty run directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    expansion = read_json(plan).get("expansion", {})
    tasks_per_material = int(expansion.get("tasks_per_material", 1)
                             if tasks_per_material is None else tasks_per_material)
    trajectories_per_task = int(expansion.get("trajectories_per_task", 1)
                                if trajectories_per_task is None else trajectories_per_task)
    stage1 = build_material_graph(plan, output / "stage1", profile=profile, seed=seed, count=count, weights=weights)
    tasks_path = output / "stage2/tasks.jsonl"
    stage2 = construct_tasks(Path(stage1["materials"]), tasks_path,
                             variants_per_material=tasks_per_material)
    stage3 = validate_oracles(tasks_path, output / "stage3", seed=seed, policy=path_policy,
                              trajectories_per_task=trajectories_per_task)
    rl_path = output / "stage4/rl_tasks.jsonl"
    rl_environments = write_jsonl(rl_path, iter_jsonl(Path(stage3["validated_tasks"])))
    counts = {"stage1_materials": stage1["material_count"],
              "stage1_unique_materials": stage1["unique_normalized_materials"],
              "stage2_candidate_tasks": stage2["tasks"],
              "stage3_candidate_trajectories": stage3["candidate_trajectories"],
              "gross_artifacts": stage1["material_count"] + stage2["tasks"] + stage3["candidate_trajectories"],
              "verifier_filtered_counts_excluded": True}
    report = {"version": "k3-agentworld-synthesis-v1", "profile": profile,
              "path_policy": path_policy, "seed": seed, "counts_before_verifier": counts,
              "stage1_method": "Kimi-K3 §4.2.2 knowledge-graph-guided synthesis",
              "stage2_3_method": "Agent-World §3.1/§3.1.1 environment and graph-based task synthesis",
              "reproduction_scope": {
                  "implemented": ["recursive web-grounded concept DAG", "controlled material sampling",
                                  "workspace task variants", "weighted strong/weak/independent graph walk"],
                  "not_claimed": ["private prompts or data", "arbitrary generated Verl tool ABI",
                                  "Agent-World database complexification and 2-of-5 ReAct consistency gate"]},
              "stage1": stage1, "stage2": stage2, "stage3": stage3,
              "stage4": {"rl_environments": rl_environments, "rl_tasks": str(rl_path),
                         "sft_status": "awaiting verified teacher/current-policy rollouts"}}
    write_json(output / "report.json", report)
    _write_run_report(output, report)
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
            if not _valid_messages(messages, tasks[task_id]["available_tools"]):
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
    if not 0 <= floor <= 1:
        raise ValueError("floor must be between 0 and 1")
    tasks = {task["task_id"]: task for task in read_jsonl(prepared / "stage3/validated_tasks.jsonl")}
    scores: dict[str, list[float]] = {task["task_family"]: [] for task in tasks.values()}
    for row in _rollout_rows(rollout_paths):
        policy = row.get("policy") if isinstance(row.get("policy"), dict) else {}
        if policy.get("kind") in {"programmatic_oracle", "oracle"}:
            continue
        task = tasks.get(str(row.get("task_id", "")))
        outcome = row.get("outcome") if isinstance(row.get("outcome"), dict) else row
        if task and outcome.get("harness_status") == "healthy":
            score = max(0.0, min(1.0, float(outcome.get("task_success", outcome.get("score", 0)))))
            scores.setdefault(task["task_family"], []).append(score)
    if not any(scores.values()):
        raise ValueError("no healthy diagnostic rollouts matched prepared tasks")
    # ponytail: error-rate allocation is deliberately transparent; replace it
    # with a calibrated learning-progress model only when this baseline fails.
    distribution = {family: max(floor, 1 - sum(values) / len(values)) if values else 1.0
                    for family, values in scores.items()}
    result = {"version": "diagnostic-distribution-v1", "distribution": distribution,
              "counts": {family: len(values) for family, values in scores.items()}}
    write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("prepare")
    build.add_argument("plan", type=Path)
    build.add_argument("output", type=Path)
    build.add_argument("--profile", default="default")
    build.add_argument("--seed", type=int, default=0)
    build.add_argument("--count", type=int)
    build.add_argument("--tasks-per-material", type=int)
    build.add_argument("--trajectories-per-task", type=int)
    build.add_argument("--path-policy", choices=("agentworld", "goal", "uniform"), default="agentworld")
    build.add_argument("--weights", type=Path)
    expand = commands.add_parser("expand-graph")
    expand.add_argument("plan", type=Path)
    expand.add_argument("output", type=Path)
    expand.add_argument("--base-url", required=True)
    expand.add_argument("--model", required=True)
    expand.add_argument("--api-key-env", default="OPENAI_API_KEY")
    expand.add_argument("--max-nodes", type=int, default=2000)
    expand.add_argument("--max-depth", type=int, default=4)
    expand.add_argument("--children-per-node", type=int, default=8)
    expand.add_argument("--searches-per-node", type=int, default=3)
    finish = commands.add_parser("finalize")
    finish.add_argument("prepared", type=Path)
    finish.add_argument("rollouts", nargs="+", type=Path)
    finish.add_argument("--policy-kind", choices=("teacher", "current"), required=True)
    finish.add_argument("--model", required=True)
    diagnostic = commands.add_parser("diagnose")
    diagnostic.add_argument("prepared", type=Path)
    diagnostic.add_argument("rollouts", nargs="+", type=Path)
    diagnostic.add_argument("--output", type=Path, required=True)
    diagnostic.add_argument("--floor", type=float, default=0.05)
    args = parser.parse_args()
    if args.command == "expand-graph":
        import os
        from scripts.synthesis.policy_rollout import openai_client

        client = openai_client(args.base_url, args.model, os.environ.get(args.api_key_env, ""))
        result = expand_knowledge_graph(args.plan, args.output, complete=client,
                                        max_nodes=args.max_nodes, max_depth=args.max_depth,
                                        children_per_node=args.children_per_node,
                                        searches_per_node=args.searches_per_node)
    elif args.command == "prepare":
        weights = read_json(args.weights).get("distribution") if args.weights else None
        result = prepare(args.plan, args.output, profile=args.profile, seed=args.seed, count=args.count,
                         path_policy=args.path_policy, weights=weights,
                         tasks_per_material=args.tasks_per_material,
                         trajectories_per_task=args.trajectories_per_task)
    elif args.command == "finalize":
        result = finalize(args.prepared, args.rollouts, policy_kind=args.policy_kind, model=args.model)
    else:
        result = diagnose(args.prepared, args.rollouts, args.output, args.floor)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
