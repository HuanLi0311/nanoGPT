"""Contracts for programmatic Agent trajectory synthesis.

The graph is an offline artifact.  The episode event log is the source of truth:
every abstract edge must point back to a concrete tool call and two workspace
snapshots.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


SUPPORTED_ACTION_TOOLS = {"exec_command", "apply_patch"}
REQUIRED_TASK_FIELDS = {"task_id", "prompt", "files", "verifier"}


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fingerprint(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(stable_json(row) + "\n")
            count += 1
    temporary.replace(path)
    return count


def validate_task(task: dict[str, Any]) -> None:
    missing = REQUIRED_TASK_FIELDS - task.keys()
    if missing:
        raise ValueError(f"task {task.get('task_id', '<unknown>')} is missing {sorted(missing)}")
    if not isinstance(task["task_id"], str) or not task["task_id"].strip():
        raise ValueError("task_id must be a non-empty string")
    if not isinstance(task["prompt"], (str, list)):
        raise ValueError(f"{task['task_id']}: prompt must be a string or chat messages")
    if not isinstance(task["files"], dict):
        raise ValueError(f"{task['task_id']}: files must be an object")
    verifier = task["verifier"]
    if not isinstance(verifier, str) and not (
        isinstance(verifier, dict) and isinstance(verifier.get("command"), str) and verifier["command"].strip()
    ):
        raise ValueError(f"{task['task_id']}: verifier must be a non-empty command")
    patterns = task.get("action_patterns", [])
    if not isinstance(patterns, list) or not patterns:
        raise ValueError(f"{task['task_id']}: action_patterns must be a non-empty list")
    for pattern in patterns:
        validate_action(pattern, allow_template=True)


def validate_action(action: dict[str, Any], *, allow_template: bool = False) -> None:
    tool = action.get("tool")
    if tool not in SUPPORTED_ACTION_TOOLS:
        raise ValueError(f"unsupported programmatic action tool: {tool}")
    arguments = action.get("arguments", action.get("arguments_template"))
    if not isinstance(arguments, dict):
        raise ValueError(f"{action.get('id', '<action>')}: arguments must be an object")
    for name in ("preconditions", "effects"):
        values = action.get(name, [])
        if not isinstance(values, list) or not all(isinstance(item, str) and item for item in values):
            raise ValueError(f"{action.get('id', '<action>')}: {name} must be a list of strings")
    if allow_template and action.get("variant_count") is not None:
        count = action["variant_count"]
        if not isinstance(count, int) or count < 1:
            raise ValueError(f"{action.get('id', '<action>')}: variant_count must be positive")


def validate_episode(episode: dict[str, Any]) -> None:
    for field in ("episode_id", "task_id", "events", "outcome"):
        if field not in episode:
            raise ValueError(f"episode missing {field}")
    if not isinstance(episode["events"], list):
        raise ValueError("episode events must be a list")
    outcome = episode["outcome"]
    if not isinstance(outcome, dict):
        raise ValueError("episode outcome must be an object")
    for field in ("protocol_status", "call_result_linkage_complete", "trace_fidelity", "harness_status"):
        if field not in outcome:
            raise ValueError(f"episode outcome missing {field}")


def task_verifier(task: dict[str, Any]) -> str:
    verifier = task["verifier"]
    return verifier["command"] if isinstance(verifier, dict) else verifier


def task_prompt(task: dict[str, Any]) -> list[dict[str, str]]:
    prompt = task["prompt"]
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    return prompt
