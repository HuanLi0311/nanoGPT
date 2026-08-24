"""Self-checks for the programmatic synthesis pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from .graph import build_graph, compose
from .schema import read_jsonl
from .synthesize import run_pipeline
from .trace_runner import ProgrammaticTraceRunner


ROOT = Path(__file__).resolve().parents[4]
TASKS = ROOT / "agent/tasks/synthesis_seed.jsonl"


def main() -> None:
    task = read_jsonl(TASKS)[0]
    pattern_graph = build_graph([task], [])
    first = compose(task, pattern_graph, candidate_index=0, required_patterns=["write_answer", "readback"])
    second = compose(task, pattern_graph, candidate_index=1, required_patterns=["write_answer", "readback"])
    assert first["action_pattern_ids"] == ["write_answer", "readback"]
    assert first["candidate_fingerprint"] != second["candidate_fingerprint"]
    assert first["actions"][1]["arguments"]["command"] != second["actions"][1]["arguments"]["command"]

    with TemporaryDirectory(prefix="synthesis-test-data-") as temporary:
        temporary_root = Path(temporary)
        runner = ProgrammaticTraceRunner(temporary_root / "work")
        episode = runner.run(task, first["actions"], "test-episode", candidate_index=0)
        outcome = episode["outcome"]
        assert outcome["task_success"] is True
        assert outcome["independent_verifier_passed"] is True
        assert outcome["harness_status"] == "healthy"
        assert outcome["call_result_linkage_complete"] is True
        assert outcome["trace_fidelity"] is True
        assert episode["events"][0]["state_delta"]["added"] == ["answer.txt"]
        assert episode["events"][1]["state_delta"] == {"added": [], "removed": [], "changed": []}

        report = run_pipeline(
            TASKS,
            data_root=temporary_root / "data",
            run_id="self-check",
            rollouts=3,
            workspace_root=temporary_root / "pipeline-work",
        )
        assert report["accepted_sft_episodes"] == 3
        assert report["accepted_rl_environments"] == 0
        saved = json.loads(Path(report["outputs"]["graph"]).read_text(encoding="utf-8"))
        assert saved["graph_version"] == "execution-graph-v1"
        assert any(edge["relation"] == "invokes" for edge in saved["transition_edges"])
        assert len(saved["event_nodes"]) == 11  # seed: 2 events + 3 candidates: 3 each
        episodes = [json.loads(line) for line in Path(report["outputs"]["raw_episodes"]).read_text().splitlines()]
        assert len(episodes) == 4  # one seed episode plus three candidates
        assert all(item["outcome"]["harness_status"] == "healthy" for item in episodes)

        failed_task = {**task, "task_id": "synthesis_negative_verifier", "verifier": {"command": "false"}}
        failed_tasks = temporary_root / "failed_tasks.jsonl"
        failed_tasks.write_text(json.dumps(failed_task, ensure_ascii=False) + "\n", encoding="utf-8")
        failed_report = run_pipeline(
            failed_tasks,
            data_root=temporary_root / "failed-data",
            run_id="negative-check",
            rollouts=1,
            workspace_root=temporary_root / "failed-work",
        )
        assert failed_report["accepted_sft_episodes"] == 0
        assert failed_report["accepted_rl_environments"] == 0
        assert failed_report["diagnostic_episodes"] == 1

    print("synthesis self-check passed")


if __name__ == "__main__":
    main()
