"""Convert a small verified task manifest to verl tool-agent rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Iterator


# Verification is an environment/harness transition, not a policy action.
TOOL_NAMES = ("exec_command", "apply_patch")


def _valid_verifier(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    return isinstance(value, dict) and isinstance(value.get("command"), str) and bool(value["command"].strip())


def _valid_relative_path(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    path = PurePosixPath(str(value))
    return not path.is_absolute() and ".." not in path.parts


def task_rows(source: Path, limit: int | None = None) -> Iterator[dict[str, Any]]:
    count = 0
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{source}:{line_number}: invalid JSON") from error
        task_id = str(item.get("task_id", "")).strip()
        prompt = item.get("prompt")
        verifier = item.get("verifier")
        verifier_version = str(item.get("verifier_version", "manifest-v1"))
        harness_version = str(item.get("harness_version", "workspace-host-v2"))
        tool_schema_version = str(item.get("tool_schema_version", "workspace-tools-v2"))
        files = item.get("files", {})
        if not task_id or not isinstance(prompt, (str, list)) or not _valid_verifier(verifier):
            raise ValueError(f"{source}:{line_number}: task_id, prompt, and verifier are required")
        if not isinstance(files, dict):
            raise ValueError(f"{source}:{line_number}: files must be an object of relative paths to contents")
        if any(not _valid_relative_path(path) for path in files):
            raise ValueError(f"{source}:{line_number}: files may not escape the workspace")
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]
        contract = {
            "kind": "environment",
            "task_id": task_id,
            "verifier": verifier,
            "verifier_version": verifier_version,
            "harness_version": harness_version,
            "tool_schema_version": tool_schema_version,
        }
        create_kwargs = {
            "task_id": task_id,
            "files": files,
            "verifier": verifier,
            "verifier_version": verifier_version,
            "harness_version": harness_version,
            "tool_schema_version": tool_schema_version,
        }
        extra_info = dict(item.get("extra_info", {})) if isinstance(item.get("extra_info", {}), dict) else {}
        tool_names = item.get("available_tools", list(TOOL_NAMES))
        if not isinstance(tool_names, list) or not tool_names or not set(tool_names).issubset(TOOL_NAMES):
            raise ValueError(f"{source}:{line_number}: available_tools must be a non-empty subset of {TOOL_NAMES}")
        extra_info.update({
            "task_id": task_id,
            "reward_contract": contract,
            "harness_version": harness_version,
            "tool_schema_version": tool_schema_version,
            "verifier_version": verifier_version,
            "tool_selection": list(tool_names),
            "need_tools_kwargs": True,
            "tools_kwargs": {name: {"create_kwargs": create_kwargs} for name in tool_names},
        })
        yield {
            "data_source": "harness_tasks",
            "prompt": prompt,
            "agent_name": "tool_agent",
            "ability": "agentic_coding",
            "reward_model": {"style": "environment", "ground_truth": contract},
            "extra_info": extra_info,
        }
        count += 1
        if limit is not None and count >= limit:
            return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in task_rows(args.source, args.limit):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
