"""End-to-end programmatic trajectory synthesis.

Usage from the repository root:

    python -m model.language_model.scripts.synthesis.synthesize \
      --tasks agent/tasks/synthesis_seed.jsonl \
      --rollouts 100

The default output locations follow agent/README.md. Existing natural data and
codex_*.parquet files are never touched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .graph import build_graph, compose
from .schema import fingerprint, read_jsonl, stable_json, task_verifier, validate_task, write_jsonl
from .trace_runner import ProgrammaticTraceRunner


ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = ROOT / "model/language_model/data/post_train/data"
DEFAULT_TASKS = ROOT / "agent/tasks/synthesis_seed.jsonl"


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("synth-%Y%m%dT%H%M%SZ")


def _split_for(episode_id: str) -> str:
    value = int.from_bytes(hashlib.sha256(episode_id.encode("utf-8")).digest()[:8], "big")
    return "test" if value % 10 == 0 else "train"


def _sft_row(episode: dict[str, Any], split: str) -> dict[str, Any]:
    messages = episode["messages"]
    tool_calls = sum(len(message.get("tool_calls") or []) for message in messages)
    chars = sum(len(str(message.get("content") or "")) for message in messages)
    return {
        "messages": messages,
        "data_source": "synthetic_verified",
        "trajectory_id": episode["episode_id"],
        "split": split,
        "metadata": {
            "chars": chars,
            "cli_version": "synthesis-v1",
            "cwd": None,
            "episode_index": episode.get("candidate_index"),
            "id": episode["episode_id"],
            "malformed_json": 0,
            "model_provider": "programmatic_oracle",
            "quality_score": 1.0,
            "signals": [
                "task_success",
                "protocol_valid",
                "call_result_linkage_complete",
                "trace_fidelity",
                "independent_verifier_passed",
                "harness_healthy",
                "programmatic_oracle",
            ],
            "source_file": "synthesis",
            "timestamp": None,
            "tool_calls": tool_calls,
        },
    }


def _write_sft_parquet(episodes: list[dict[str, Any]], rendered_dir: Path, run_id: str) -> dict[str, str]:
    if not episodes:
        return {}
    import pyarrow as pa
    import pyarrow.parquet as pq

    from model.language_model.scripts.filter import SFT_SCHEMA

    rows = {"train": [], "test": []}
    for episode in episodes:
        split = _split_for(episode["episode_id"])
        rows[split].append(_sft_row(episode, split))
    rendered_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for split, values in rows.items():
        if not values:
            continue
        filename = "train_sft" if split == "train" else "test_sft"
        path = rendered_dir / f"{filename}-synthesis-{run_id}.parquet"
        if path.exists():
            raise FileExistsError(path)
        pq.write_table(pa.Table.from_pylist(values, schema=SFT_SCHEMA), path, compression="zstd")
        outputs[split] = str(path)
    return outputs


def _rl_task(task: dict[str, Any], report: dict[str, Any], run_id: str) -> dict[str, Any]:
    result = {
        key: task[key]
        for key in ("task_id", "prompt", "files", "verifier", "verifier_version", "tool_schema_version")
        if key in task
    }
    result["extra_info"] = {
        "synthesis_run_id": run_id,
        "environment_id": report["environment_id"],
        "pass_at_100": report["pass_at_100"],
        "pass_count_100": report["pass_count_100"],
        "rollouts": report["rollouts"],
        "generator_version": report["generator_version"],
        "harness_version": report["harness_version"],
        "tool_schema_version": report["tool_schema_version"],
        "verifier_version": report["verifier_version"],
        "reward_contract": {
            "kind": "environment",
            "task_id": task["task_id"],
            "verifier": task_verifier(task),
            "verifier_version": task.get("verifier_version", "manifest-v1"),
        },
    }
    return result


def _seed_graph(task: dict[str, Any]) -> dict[str, Any]:
    """Build a pattern-only graph used to produce the first concrete seed run."""

    return build_graph([task], [])


def run_pipeline(
    task_path: Path,
    *,
    data_root: Path = DATA_ROOT,
    run_id: str | None = None,
    rollouts: int = 100,
    workspace_root: Path | None = None,
    keep_workspaces: bool = False,
    max_timeout: int = 60,
    max_output: int = 12000,
) -> dict[str, Any]:
    """Generate seed traces, compose candidates, execute them, and gate tasks."""

    if rollouts < 1:
        raise ValueError("rollouts must be positive")
    run_id = run_id or _run_id()
    tasks = read_jsonl(task_path)
    if not tasks:
        raise ValueError(f"no tasks in {task_path}")
    for task in tasks:
        validate_task(task)

    raw_dir = data_root / "raw/synthetic"
    jsonl_dir = data_root / "jsonl/synth" / run_id
    rendered_dir = data_root / "rendered/sft"
    raw_path = raw_dir / f"{run_id}.jsonl"
    graph_path = raw_dir / f"{run_id}.graph.json"
    report_path = raw_dir / f"{run_id}.report.json"
    if raw_path.exists() or graph_path.exists() or report_path.exists():
        raise FileExistsError(f"synthesis run already exists: {run_id}")

    work_root = workspace_root or Path(os.environ.get("SYNTHESIS_WORKSPACE_ROOT", "/tmp/nanoagent-synthesis")) / run_id
    runner = ProgrammaticTraceRunner(
        work_root,
        max_timeout=max_timeout,
        max_output=max_output,
        keep_workspaces=keep_workspaces,
    )

    seed_episodes: list[dict[str, Any]] = []
    seed_candidates: list[dict[str, Any]] = []
    for task in tasks:
        seed_graph = _seed_graph(task)
        seed_patterns = list(task.get("seed_patterns", []))
        if not seed_patterns:
            raise ValueError(f"{task['task_id']}: seed_patterns is required")
        seed_candidate = compose(
            task,
            seed_graph,
            candidate_index=0,
            required_patterns=seed_patterns,
            max_steps=int(task.get("max_steps", 8)),
        )
        seed_candidates.append(seed_candidate)
        seed_episode = runner.run(
            task,
            seed_candidate["actions"],
            f"{run_id}:{task['task_id']}:seed",
            candidate_index=None,
        )
        seed_episodes.append(seed_episode)

    graph = build_graph(tasks, seed_episodes)
    all_episodes: list[dict[str, Any]] = [*seed_episodes]
    candidates: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    accepted_sft: list[dict[str, Any]] = []
    accepted_rl: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []

    for task in tasks:
        task_rollouts = int(task.get("rollouts", rollouts))
        if task_rollouts < 1:
            raise ValueError(f"{task['task_id']}: rollouts must be positive")
        reports_for_task: list[dict[str, Any]] = []
        seen_fingerprints: set[str] = set()
        candidate_patterns = list(task.get("candidate_patterns", task.get("seed_patterns", [])))
        for index in range(task_rollouts):
            candidate = compose(
                task,
                graph,
                candidate_index=index,
                required_patterns=candidate_patterns,
                max_steps=int(task.get("max_steps", 8)),
            )
            if candidate["candidate_fingerprint"] in seen_fingerprints:
                raise ValueError(
                    f"{task['task_id']}: candidate {index} duplicates an earlier action/state candidate"
                )
            seen_fingerprints.add(candidate["candidate_fingerprint"])
            candidates.append(candidate)
            episode = runner.run(
                task,
                candidate["actions"],
                f"{run_id}:{task['task_id']}:candidate:{index:04d}",
                candidate_index=index,
            )
            all_episodes.append(episode)
            reports_for_task.append(episode["outcome"])
            if (
                episode["outcome"]["task_success"]
                and episode["outcome"]["protocol_status"] == "valid"
                and episode["outcome"]["call_result_linkage_complete"]
                and episode["outcome"]["trace_fidelity"]
                and episode["outcome"]["independent_verifier_passed"]
                and episode["outcome"]["harness_status"] == "healthy"
            ):
                accepted_sft.append(episode)
            else:
                diagnostics.append(episode)

        passed = sum(bool(outcome["independent_verifier_passed"]) for outcome in reports_for_task)
        gate_rollouts = min(100, len(reports_for_task))
        pass_count_100 = sum(
            bool(outcome["independent_verifier_passed"])
            for outcome in reports_for_task[:gate_rollouts]
        )
        report = {
            "task_id": task["task_id"],
            "environment_id": f"{task['task_id']}:{fingerprint(task['files'])[:12]}",
            "rollouts": task_rollouts,
            "gate_rollouts": gate_rollouts,
            "pass_count_100": pass_count_100,
            "passed_rollouts": passed,
            "pass_at_100": bool(task_rollouts >= 100 and pass_count_100 > 0),
            "generator_version": "graph-synthesis-v1",
            "harness_version": task.get("harness_version", "workspace-host-v2"),
            "tool_schema_version": task.get("tool_schema_version", "workspace-tools-v2"),
            "verifier_version": task.get("verifier_version", "manifest-v1"),
            "unique_candidate_count": len(seen_fingerprints),
        }
        reports.append(report)
        if report["pass_at_100"]:
            accepted_rl.append(_rl_task(task, report, run_id))

    # Candidate executions are evidence too.  Persist the final graph with all
    # live transitions, not only the seed episodes used to build the composer.
    graph = build_graph(tasks, all_episodes)
    raw_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(raw_path, all_episodes)
    write_jsonl(jsonl_dir / "seed_candidates.jsonl", seed_candidates)
    write_jsonl(jsonl_dir / "candidates.jsonl", candidates)
    write_jsonl(jsonl_dir / "episodes.jsonl", all_episodes)
    write_jsonl(
        jsonl_dir / "sft.jsonl",
        [_sft_row(item, _split_for(item["episode_id"])) for item in accepted_sft],
    )
    write_jsonl(jsonl_dir / "rl_tasks.jsonl", accepted_rl)
    write_jsonl(jsonl_dir / "diagnostics.jsonl", diagnostics)
    graph_path.write_text(stable_json(graph) + "\n", encoding="utf-8")
    report_payload = {
        "run_id": run_id,
        "task_path": str(task_path),
        "rollouts": rollouts,
        "tasks": reports,
        "accepted_sft_episodes": len(accepted_sft),
        "accepted_rl_environments": len(accepted_rl),
        "diagnostic_episodes": len(diagnostics),
        "outputs": {
            "raw_episodes": str(raw_path),
            "graph": str(graph_path),
            "jsonl": str(jsonl_dir),
        },
    }
    report_path.write_text(stable_json(report_payload) + "\n", encoding="utf-8")
    parquet = _write_sft_parquet(accepted_sft, rendered_dir, run_id)
    report_payload["outputs"]["sft_parquet"] = parquet
    report_path.write_text(stable_json(report_payload) + "\n", encoding="utf-8")
    return report_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--run-id")
    parser.add_argument("--rollouts", type=int, default=100)
    parser.add_argument("--workspace-root", type=Path)
    parser.add_argument("--keep-workspaces", action="store_true")
    parser.add_argument("--max-timeout", type=int, default=60)
    parser.add_argument("--max-output", type=int, default=12000)
    args = parser.parse_args()
    print(json.dumps(run_pipeline(
        args.tasks,
        data_root=args.data_root,
        run_id=args.run_id,
        rollouts=args.rollouts,
        workspace_root=args.workspace_root,
        keep_workspaces=args.keep_workspaces,
        max_timeout=args.max_timeout,
        max_output=args.max_output,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
