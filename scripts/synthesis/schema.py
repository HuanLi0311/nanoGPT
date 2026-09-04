"""Small, shared contracts for the four-stage synthesis pipeline."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator


TOOLS = {"exec_command", "apply_patch"}
TOKEN = re.compile(r"\{\{([a-zA-Z0-9_.:-]+)\}\}")


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fingerprint(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected one JSON object")
    return value


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number}: expected a JSON object")
            yield value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


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


def render(value: Any, values: dict[str, Any]) -> Any:
    if isinstance(value, str):
        return TOKEN.sub(lambda match: str(values.get(match.group(1), match.group(0))), value)
    if isinstance(value, list):
        return [render(item, values) for item in value]
    if isinstance(value, dict):
        return {key: render(item, values) for key, item in value.items()}
    return value


def relative_path(value: Any) -> str:
    path = PurePosixPath(str(value))
    if not str(value).strip() or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"workspace path must be relative: {value!r}")
    return str(path)


def validate_plan(plan: dict[str, Any]) -> None:
    domains = plan.get("domains")
    if not isinstance(domains, list) or not domains:
        raise ValueError("plan.domains must be a non-empty list")
    seen: set[str] = set()
    for domain in domains:
        if not isinstance(domain.get("name"), str) or not isinstance(domain.get("subdomains"), list):
            raise ValueError("each domain needs name and subdomains")
        for subdomain in domain["subdomains"]:
            if not isinstance(subdomain.get("name"), str) or not isinstance(subdomain.get("concepts"), list):
                raise ValueError("each subdomain needs name and concepts")
            for concept in subdomain["concepts"]:
                key = f"{domain['name']}.{subdomain['name']}.{concept.get('name', '')}"
                tasks = concept.get("tasks", [concept.get("task")])
                if (key in seen or not concept.get("source") or not isinstance(tasks, list)
                        or not tasks or any(not isinstance(task, dict) for task in tasks)):
                    raise ValueError(f"duplicate or incomplete concept: {key}")
                for field in ("parent_ids", "related_ids"):
                    if not isinstance(concept.get(field, []), list) or not all(
                            isinstance(item, str) for item in concept.get(field, [])):
                        raise ValueError(f"{key}.{field} must be a list of node ids")
                seen.add(key)


def validate_task(task: dict[str, Any]) -> None:
    required = {"task_id", "prompt", "files", "verifier", "available_tools", "action_patterns", "sandbox_backend"}
    missing = required - task.keys()
    if missing:
        raise ValueError(f"{task.get('task_id', '<task>')}: missing {sorted(missing)}")
    if not isinstance(task["files"], dict) or any(relative_path(path) != path for path in task["files"]):
        raise ValueError(f"{task['task_id']}: invalid initial files")
    verifier = task["verifier"]
    command = verifier.get("command") if isinstance(verifier, dict) else verifier
    if not isinstance(command, str) or not command.strip():
        raise ValueError(f"{task['task_id']}: independent verifier command is required")
    if task["sandbox_backend"] not in {"bwrap", "workspace_host"}:
        raise ValueError(f"{task['task_id']}: sandbox_backend must be bwrap or workspace_host")
    tools = task["available_tools"]
    if not isinstance(tools, list) or not set(tools).issubset(TOOLS):
        raise ValueError(f"{task['task_id']}: unsupported available_tools")
    actions = task["action_patterns"]
    ids = [action.get("id") for action in actions]
    if not actions or len(ids) != len(set(ids)) or any(not value for value in ids):
        raise ValueError(f"{task['task_id']}: action ids must be non-empty and unique")
    for action in actions:
        if action.get("tool") not in tools or not isinstance(action.get("arguments", action.get("arguments_template")), dict):
            raise ValueError(f"{task['task_id']}: invalid action {action.get('id')}")
        for field in ("preconditions", "effects", "depends_on"):
            if not all(isinstance(item, str) for item in action.get(field, [])):
                raise ValueError(f"{task['task_id']}: {action.get('id')}.{field} must be strings")
    if not task.get("target_facts") and not task.get("required_actions"):
        raise ValueError(f"{task['task_id']}: target_facts or required_actions is required")
