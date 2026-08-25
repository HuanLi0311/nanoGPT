"""Run the 80k task variants in resumable per-task batches."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory

from .schema import read_jsonl, stable_json
from .synthesize import run_pipeline


def safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def existing_report(data_root: Path, run_id: str) -> dict | None:
    path = data_root / "raw/synthetic" / f"{run_id}.report.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run(tasks_path: Path, data_root: Path, run_id: str, start: int, stop: int | None) -> dict:
    tasks = read_jsonl(tasks_path)
    selected = tasks[start:stop]
    if not selected:
        raise ValueError("selected task range is empty")
    results = []
    for index, task in ((start + offset, task) for offset, task in enumerate(selected)):
        task_run_id = f"{run_id}-{index:03d}-{safe(task['task_id'])}"
        expected = int(task["rollouts"])
        previous = existing_report(data_root, task_run_id)
        if previous is not None:
            if previous.get("accepted_sft_episodes") != expected or previous.get("diagnostic_episodes") != 0:
                raise RuntimeError(f"existing incomplete report: {task_run_id}")
            report = previous
            print(json.dumps({"status": "skip", "task": task["task_id"], "accepted": expected}), flush=True)
        else:
            with TemporaryDirectory(prefix=f"{task_run_id}-manifest-") as temporary:
                one_task = Path(temporary) / "task.jsonl"
                one_task.write_text(json.dumps(task, ensure_ascii=False) + "\n", encoding="utf-8")
                quiet = io.StringIO()
                with contextlib.redirect_stdout(quiet):
                    report = run_pipeline(
                        one_task,
                        data_root=data_root,
                        run_id=task_run_id,
                        rollouts=100,
                        workspace_root=Path("/tmp/nanoagent-synthesis") / task_run_id,
                    )
            if report["accepted_sft_episodes"] != expected or report["diagnostic_episodes"] != 0:
                raise RuntimeError(f"quality gate failed: {task_run_id}: {report}")
            print(json.dumps({
                "status": "done",
                "task": task["task_id"],
                "accepted": report["accepted_sft_episodes"],
                "pass_at_100": all(item["pass_at_100"] for item in report["tasks"]),
            }), flush=True)
        results.append({
            "task_id": task["task_id"],
            "domain": task["domain"],
            "trajectory_profile": task["trajectory_profile"],
            "rollouts": expected,
            "accepted_sft_episodes": report["accepted_sft_episodes"],
            "diagnostic_episodes": report["diagnostic_episodes"],
            "pass_at_100": all(item["pass_at_100"] for item in report["tasks"]),
            "run_id": task_run_id,
            "report": report["outputs"],
        })

    summary = {
        "run_id": run_id,
        "task_manifest": str(tasks_path),
        "task_count": len(tasks),
        "processed_task_count": len(results),
        "accepted_sft_episodes": sum(item["accepted_sft_episodes"] for item in results),
        "diagnostic_episodes": sum(item["diagnostic_episodes"] for item in results),
        "all_pass_at_100": all(item["pass_at_100"] for item in results),
        "tasks": results,
    }
    output = data_root / "raw/synthetic" / f"{run_id}.summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(stable_json(summary) + "\n", encoding="utf-8")
    print(json.dumps({
        "summary": str(output),
        "processed": len(results),
        "accepted": summary["accepted_sft_episodes"],
        "all_pass_at_100": summary["all_pass_at_100"],
    }), flush=True)
    return summary


def main() -> None:
    root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, default=root / "agent/tasks/synthesis_80k.jsonl")
    parser.add_argument("--data-root", type=Path, default=root / "model/language_model/data/post_train/data")
    parser.add_argument("--run-id", default="synth-80k-20260825")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    args = parser.parse_args()
    run(args.tasks, args.data_root, args.run_id, args.start, args.stop)


if __name__ == "__main__":
    main()
