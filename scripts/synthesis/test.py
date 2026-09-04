"""One dependency-light end-to-end check for the four-stage pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

if __package__:
    from .runner import diagnose, finalize, prepare
    from .schema import read_json, read_jsonl, write_json, write_jsonl
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from runner import diagnose, finalize, prepare
    from schema import read_json, read_jsonl, write_json, write_jsonl


def _plan() -> dict:
    return {
        "profiles": {
            "narrow": {"count": 2, "distribution": {"coding.transform": 1, "artifact.create": 0}},
            "diverse": {"count": 2, "distribution": {"coding.transform": 1, "artifact.create": 1}},
        },
        "domains": [
            {"name": "coding", "subdomains": [{"name": "transform", "concepts": [{
                "name": "uppercase", "source": {"kind": "inline", "content": "alpha\n", "license": "test-only"},
                "task": {
                    "id": "uppercase", "prompt": "Read input.txt and save its uppercase value in answer.txt.",
                    "files": {"input.txt": "{{material}}"},
                    "available_tools": ["exec_command"],
                    "verifier": {"command": "test \"$(cat answer.txt)\" = ALPHA"},
                    "initial_facts": ["workspace:ready"], "target_facts": ["file:answer.txt:exists"],
                    "actions": [
                        {"id": "read_upper", "tool": "exec_command", "arguments": {"cmd": "tr '[:lower:]' '[:upper:]' < input.txt"},
                         "preconditions": ["file:input.txt:exists"], "effects": ["value:upper"]},
                        {"id": "write_answer", "tool": "exec_command",
                         "arguments_template": {"cmd": "printf '%s\\n' {{output:read_upper}} > answer.txt"},
                         "depends_on": ["read_upper"], "preconditions": ["value:upper"], "effects": ["file:answer.txt:exists"]},
                    ],
                },
            }]}]},
            {"name": "artifact", "subdomains": [{"name": "create", "concepts": [{
                "name": "report", "source": {"kind": "inline", "content": "beta\n", "license": "test-only"},
                "task": {
                    "id": "report", "prompt": "Create report.txt containing BETA.", "files": {},
                    "available_tools": ["apply_patch"], "verifier": "test \"$(cat report.txt)\" = BETA",
                    "target_facts": ["file:report.txt:exists"],
                    "actions": [{"id": "write_report", "tool": "apply_patch",
                                 "arguments": {"patch": "*** Begin Patch\n*** Add File: report.txt\n+BETA\n*** End Patch"},
                                 "preconditions": ["workspace:ready"], "effects": ["file:report.txt:exists"]}],
                },
            }]}]},
        ],
    }


def main() -> None:
    with TemporaryDirectory(prefix="four-stage-synthesis-") as temporary:
        root = Path(temporary)
        plan = root / "plan.json"
        write_json(plan, _plan())
        report = prepare(plan, root / "run", profile="diverse", seed=7, count=None, path_policy="goal")
        assert report["stage1"]["realized_distribution"] == {"artifact.create": 1, "coding.transform": 1}
        assert report["stage3"]["validated"] == 2 and report["stage3"]["rejected"] == 0
        assert report["stage4"]["sft_status"].startswith("awaiting")

        tasks = read_jsonl(root / "run/stage3/validated_tasks.jsonl")
        episodes = read_jsonl(root / "run/stage3/oracle_episodes.jsonl")
        from model.language_model.scripts.prepare_verl_tasks import task_rows
        verl_rows = list(task_rows(root / "run/stage4/rl_tasks.jsonl"))
        assert all(set(row["extra_info"]["tools_kwargs"]) == set(task["available_tools"])
                   for row, task in zip(verl_rows, tasks, strict=True))
        assert all(set(row["extra_info"]["tool_selection"]) == set(task["available_tools"])
                   for row, task in zip(verl_rows, tasks, strict=True))
        by_task = {episode["task_id"]: episode for episode in episodes}
        oracle_result = finalize(root / "run", [root / "run/stage3/oracle_episodes.jsonl"],
                                 policy_kind="teacher", model="test-teacher")
        assert oracle_result["accepted_sft"] == 0 and oracle_result["rejected"] == 2
        serialized_action = by_task[tasks[0]["task_id"]]["actions"][0]
        serialized = ("<tool_call>\n" + json.dumps({"name": serialized_action["tool"],
                      "arguments": serialized_action["arguments"]}) +
                      "\n</tool_call>\nuser\n<tool_response>\nexit_code=0\nok\n</tool_response>\nassistant\nDONE")
        rollouts = root / "rollouts.jsonl"
        write_jsonl(rollouts, [
            {"task_id": tasks[0]["task_id"], "policy": {"kind": "teacher"}, "messages": by_task[tasks[0]["task_id"]]["messages"],
             "outcome": {"task_success": 1, "harness_status": "healthy", "eligible": True}},
            {"task_id": tasks[1]["task_id"], "policy": {"kind": "teacher"}, "messages": by_task[tasks[1]["task_id"]]["messages"],
             "outcome": {"task_success": 0, "harness_status": "healthy", "eligible": True}},
            {"task_id": tasks[0]["task_id"], "policy": {"kind": "programmatic_oracle"},
             "messages": by_task[tasks[0]["task_id"]]["messages"], "outcome": {"task_success": 1, "harness_status": "healthy"}},
            {"task_id": tasks[0]["task_id"], "policy": {"kind": "teacher"}, "output": serialized,
             "gts": {"task_id": tasks[0]["task_id"], "verifier_version": tasks[0]["verifier_version"]},
             "task_success": 1, "harness_status": "healthy", "eligible": True},
        ])
        result = finalize(root / "run", [rollouts], policy_kind="teacher", model="test-teacher")
        assert result["accepted_sft"] == 2 and result["rejected"] == 2
        assert read_jsonl(root / "run/stage4/sft.jsonl")[0]["metadata"]["model_provider"] == "test-teacher"

        weights = root / "weights.json"
        diagnosed = diagnose(root / "run", [rollouts], weights)
        failed_family = next(task["task_family"] for task in tasks if task["task_id"] == tasks[1]["task_id"])
        passed_family = next(task["task_family"] for task in tasks if task["task_id"] == tasks[0]["task_id"])
        assert diagnosed["distribution"][failed_family] > diagnosed["distribution"][passed_family]
        assert read_json(weights)["version"] == "diagnostic-distribution-v1"
    print("four-stage synthesis self-check passed")


if __name__ == "__main__":
    main()
