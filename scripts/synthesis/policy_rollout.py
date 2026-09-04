"""Run a teacher/current policy against validated tasks through real tools."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from tempfile import mkdtemp
from types import SimpleNamespace
from typing import Any, Callable
from urllib.parse import urlparse
from urllib.request import Request, urlopen

if __package__:
    from .schema import read_jsonl, write_jsonl
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from schema import read_jsonl, write_jsonl


Complete = Callable[[list[dict[str, Any]], list[dict[str, Any]]], Any]


def openai_client(base_url: str, model: str, api_key: str = "", timeout: int = 120,
                  temperature: float = 0.0, max_tokens: int = 2048) -> Complete:
    if urlparse(base_url).scheme not in {"http", "https"}:
        raise ValueError("base_url must use http or https")
    if timeout < 1 or max_tokens < 1:
        raise ValueError("timeout and max_tokens must be positive")
    endpoint = base_url.rstrip("/")
    if not endpoint.endswith("chat/completions"):
        endpoint += "/chat/completions"

    def complete(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        body = json.dumps({"model": model, "messages": messages, "tools": tools,
                           "tool_choice": "auto", "temperature": temperature,
                           "max_tokens": max_tokens}).encode()
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        with urlopen(Request(endpoint, data=body, headers=headers), timeout=timeout) as response:
            value = json.loads(response.read())
        return value["choices"][0]["message"]

    return complete


def _schemas(names: list[str], root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    from model.language_model.scripts.synthesis.trace_runner import _tool_schema
    from model.language_model.scripts.verl_workspace_tool import WorkspaceTool

    tools = {name: WorkspaceTool({"operation": name, "workspace_root": str(root)}, _tool_schema(name))
             for name in [*names, "verify_task"]}
    schemas = {name: _tool_schema(name).model_dump(exclude_none=True) for name in names}
    return tools, schemas


def _message(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("policy response must be an object")
    if isinstance(value.get("choices"), list):
        value = value["choices"][0]["message"]
    content = value.get("content")
    if isinstance(content, list):
        content = "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
    return {"role": "assistant", "content": str(content or ""), "tool_calls": value.get("tool_calls") or []}


async def _one(task: dict[str, Any], complete: Complete, policy_kind: str, model: str,
               index: int, max_turns: int, workspace_root: Path) -> dict[str, Any]:
    tools, schemas = _schemas(task["available_tools"], workspace_root)
    episode_id = f"policy-{policy_kind}-{index:06d}"
    create = {key: task[key] for key in ("task_id", "files", "verifier", "verifier_version",
                                         "harness_version", "tool_schema_version")}
    agent_data = SimpleNamespace(request_id=episode_id, extra_fields={})
    for tool in tools.values():
        await tool.create(episode_id, create_kwargs=create)
    messages = deepcopy(task["prompt"] if isinstance(task["prompt"], list)
                        else [{"role": "user", "content": task["prompt"]}])
    termination = "turn_limit"
    try:
        for turn in range(max_turns):
            response = complete(deepcopy(messages), list(schemas.values()))
            response = await response if inspect.isawaitable(response) else response
            assistant = _message(response)
            calls = assistant.pop("tool_calls")
            if calls:
                normalized, pending = [], []
                for call_index, call in enumerate(calls):
                    function = call.get("function", {}) if isinstance(call, dict) else {}
                    name, arguments = function.get("name"), function.get("arguments", "{}")
                    arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
                    if name not in tools or name == "verify_task" or not isinstance(arguments, dict):
                        raise ValueError(f"invalid policy tool call: {name}")
                    call_id = str(call.get("id") or f"call_{turn:03d}_{call_index:03d}")
                    normalized.append({"id": call_id, "type": "function",
                                       "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}})
                    pending.append((call_id, name, arguments))
                assistant["tool_calls"] = normalized
                messages.append(assistant)
                for call_id, name, arguments in pending:
                    agent_data.tool_calls = [SimpleNamespace(name=name, arguments=json.dumps(arguments), tool_call_id=call_id)]
                    tool_response, _, _ = await tools[name].execute(episode_id, arguments, agent_data=agent_data)
                    messages.append({"role": "tool", "tool_call_id": call_id,
                                     "content": str(getattr(tool_response, "text", ""))})
            else:
                messages.append(assistant)
                termination = "final"
                break
        await tools["verify_task"].execute(episode_id, {}, agent_data=agent_data)
        outcome = agent_data.extra_fields.get("task_outcome") or {
            "task_id": task["task_id"], "task_success": 0, "harness_status": "fault",
            "eligible": False, "reason": "missing verifier outcome",
        }
        return {"task_id": task["task_id"], "policy": {"kind": policy_kind, "model": model},
                "messages": messages, "outcome": outcome,
                "agent_events": agent_data.extra_fields.get("agent_events", []),
                "termination_reason": termination,
                "gts": {"kind": "environment", "task_id": task["task_id"],
                        "verifier_version": task["verifier_version"]}}
    finally:
        for tool in tools.values():
            await tool.release(episode_id)


def run_policy_tasks(tasks_path: Path, output: Path, *, complete: Complete, policy_kind: str,
                     model: str, max_turns: int = 20, limit: int | None = None,
                     workspace_root: Path | None = None, keep_workspaces: bool = False) -> dict[str, Any]:
    if policy_kind not in {"teacher", "current"} or max_turns < 1:
        raise ValueError("policy_kind must be teacher/current and max_turns must be positive")
    tasks = read_jsonl(tasks_path)[:limit]
    if any(not isinstance(task.get("oracle_proof"), dict) for task in tasks):
        raise ValueError("policy rollout only accepts oracle-validated tasks")
    if not tasks:
        raise ValueError("no oracle-validated tasks to roll out")
    workspace_base = workspace_root or output.parent / "policy_workspaces"
    workspace_base.mkdir(parents=True, exist_ok=True)
    root = Path(mkdtemp(prefix="run-", dir=workspace_base))
    rows = []
    # ponytail: run sequentially so one workspace maps to one trace; batch
    # endpoint requests only when policy-rollout throughput is the bottleneck.
    for index, task in enumerate(tasks):
        try:
            rows.append(asyncio.run(_one(task, complete, policy_kind, model, index, max_turns, root)))
        except Exception as error:
            rows.append({"task_id": task.get("task_id"), "policy": {"kind": policy_kind, "model": model},
                         "messages": task.get("prompt", []), "outcome": {"task_success": 0,
                         "harness_status": "fault", "failure_class": "model_transport",
                         "eligible": False, "reason": str(error)}})
    write_jsonl(output, rows)
    if not keep_workspaces:
        shutil.rmtree(root, ignore_errors=True)
    return {"tasks": len(rows), "passed": sum(bool(row["outcome"].get("task_success")) for row in rows),
            "output": str(output), "policy_kind": policy_kind, "model": model,
            "workspace": str(root) if keep_workspaces else None}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tasks", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--policy-kind", choices=("teacher", "current"), required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workspace-root", type=Path)
    parser.add_argument("--keep-workspaces", action="store_true")
    args = parser.parse_args()
    client = openai_client(args.base_url, args.model, os.environ.get(args.api_key_env, ""),
                           args.timeout, args.temperature, args.max_tokens)
    print(json.dumps(run_policy_tasks(args.tasks, args.output, complete=client, policy_kind=args.policy_kind,
          model=args.model, max_turns=args.max_turns, limit=args.limit, workspace_root=args.workspace_root,
          keep_workspaces=args.keep_workspaces), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
