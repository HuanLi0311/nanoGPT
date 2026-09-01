"""Programmatic trace runner backed by the same WorkspaceTool as VERL.

The runner receives an action sequence; it never asks a model to produce the
next action.  The verifier is invoked out-of-band after the sequence and is
recorded separately from the model-facing messages.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from verl.tools.schemas import OpenAIFunctionToolSchema

from model.language_model.scripts.verl_workspace_tool import WorkspaceTool
from agent.workspace.snapshot import snapshot

from .schema import fingerprint, task_prompt, task_verifier, validate_action, validate_task


_SAFE_PART = re.compile(r"[^A-Za-z0-9._-]+")
_TOOL_DESCRIPTIONS = {
    "exec_command": "Run a shell command inside the episode workspace.",
    "apply_patch": "Apply a patch inside the episode workspace.",
    "verify_task": "Run the independent task verifier.",
}


def safe_part(value: Any, default: str) -> str:
    result = _SAFE_PART.sub("_", str(value or default)).strip("._")
    return result or default


def _tool_schema(name: str) -> OpenAIFunctionToolSchema:
    properties: dict[str, dict[str, Any]]
    required: list[str]
    if name == "exec_command":
        properties = {
            "cmd": {"type": "string", "description": "Command to run."},
            "workdir": {"type": "string", "description": "Relative workspace directory."},
            "yield_time_ms": {"type": "integer", "description": "Polling delay in milliseconds."},
            "max_output_tokens": {"type": "integer", "description": "Maximum returned output tokens."},
            "tty": {"type": "boolean", "description": "Terminal hint."},
            "shell": {"type": "string", "description": "Shell executable."},
            "login": {"type": "boolean", "description": "Login-shell hint."},
            "justification": {"type": "string", "description": "Permission-audit justification."},
            "prefix_rule": {"type": "array", "items": {"type": "string"}, "description": "Permission-audit prefix rule."},
            "sandbox_permissions": {"type": "string", "enum": ["use_default", "require_escalated"], "description": "Permission request; escalation is unavailable in workspace_host."},
        }
        required = ["cmd"]
    elif name == "apply_patch":
        properties = {"patch": {"type": "string", "description": "Unified or Codex patch."}}
        required = ["patch"]
    else:
        properties = {}
        required = []
    return OpenAIFunctionToolSchema.model_validate({
        "type": "function",
        "function": {
            "name": name,
            "description": _TOOL_DESCRIPTIONS[name],
            "parameters": {"type": "object", "properties": properties, "required": required, "additionalProperties": False},
        },
    })


def _verifier_output(response: Any) -> str:
    return str(getattr(response, "text", None) or "")


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _canonical_action(action: dict[str, Any]) -> dict[str, Any]:
    """Keep legacy synthetic actions readable while emitting the Codex ABI."""

    if action.get("tool") != "exec_command" or not isinstance(action.get("arguments"), dict):
        return action
    arguments = dict(action["arguments"])
    if "cmd" not in arguments and "command" in arguments:
        arguments["cmd"] = arguments.pop("command")
    else:
        arguments.pop("command", None)
    if "workdir" not in arguments and "cwd" in arguments:
        arguments["workdir"] = arguments.pop("cwd")
    else:
        arguments.pop("cwd", None)
    return {**action, "arguments": arguments}


def snapshot_environment(root: Path) -> dict[str, Any]:
    state = snapshot(root)
    files = {item["path"]: {"sha256": item["sha256"], "size": item["size"]} for item in state["files"]}
    environment = {"root_kind": "workspace", "files": files}
    return {"state_hash": fingerprint(environment), **environment}


def _state_view(
    root: Path,
    task: dict[str, Any],
    observations: list[str],
    call_ids: list[str],
    verifier_status: str,
) -> dict[str, Any]:
    environment = snapshot_environment(root)
    state = {
        "environment_state": environment,
        "agent_context_state": {
            "observation_count": len(observations),
            "observation_hashes": [fingerprint(item) for item in observations],
        },
        "harness_state": {
            "toolset": ["exec_command", "apply_patch"],
            "tool_call_ids": list(call_ids),
            "last_tool_call_id": call_ids[-1] if call_ids else None,
            "protocol_status": "valid",
        },
        "goal_verifier_state": {
            "task_id": task["task_id"],
            "verifier_version": task.get("verifier_version", "manifest-v1"),
            "status": verifier_status,
        },
    }
    return {"state_hash": fingerprint(state), **state}


def _file_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, list[str]]:
    before_files = before["environment_state"]["files"]
    after_files = after["environment_state"]["files"]
    added = sorted(set(after_files) - set(before_files))
    removed = sorted(set(before_files) - set(after_files))
    changed = sorted(
        name for name in set(before_files) & set(after_files)
        if before_files[name] != after_files[name]
    )
    return {"added": added, "removed": removed, "changed": changed}


def _action_message(call_id: str, action: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {
                "name": action["tool"],
                "arguments": json.dumps(action["arguments"], ensure_ascii=False, separators=(",", ":")),
            },
        }],
    }


class ProgrammaticTraceRunner:
    """Execute fixed actions and record auditable state transitions."""

    def __init__(
        self,
        workspace_root: Path,
        *,
        max_timeout: int = 60,
        max_output: int = 12000,
        keep_workspaces: bool = False,
    ):
        self.workspace_root = workspace_root
        self.max_timeout = max(1, int(max_timeout))
        self.max_output = max(256, int(max_output))
        self.keep_workspaces = keep_workspaces
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.tools = {
            name: WorkspaceTool(
                {
                    "operation": name,
                    "workspace_root": str(workspace_root),
                    "max_timeout": self.max_timeout,
                    "max_output": self.max_output,
                },
                _tool_schema(name),
            )
            for name in ("exec_command", "apply_patch", "verify_task")
        }

    async def _run(
        self,
        task: dict[str, Any],
        actions: list[dict[str, Any]],
        episode_id: str,
        *,
        candidate_index: int | None = None,
    ) -> dict[str, Any]:
        validate_task(task)
        for action in actions:
            validate_action(action)
        actions = [_canonical_action(action) for action in actions]

        task_id = safe_part(task["task_id"], "task")
        episode_key = safe_part(episode_id, "episode")
        workspace = self.workspace_root / task_id / episode_key
        workspace.mkdir(parents=True, exist_ok=True)
        WorkspaceTool._initialize_files(workspace, task["files"])

        create_kwargs = {
            "task_id": task["task_id"],
            "files": task["files"],
            "verifier": task["verifier"],
            "verifier_version": task.get("verifier_version", "manifest-v1"),
            "harness_version": task.get("harness_version", "workspace-host-v2"),
            "tool_schema_version": task.get("tool_schema_version", "workspace-tools-v2"),
        }
        agent_data = SimpleNamespace(request_id=episode_id, extra_fields={})
        for tool in self.tools.values():
            await tool.create(episode_id, create_kwargs=create_kwargs)

        observations: list[str] = []
        call_ids: list[str] = []
        events: list[dict[str, Any]] = []
        messages = deepcopy(task_prompt(task))
        initial_state = _state_view(workspace, task, observations, call_ids, "pending")
        harness_fault: str | None = None

        for index, action in enumerate(actions):
            call_id = f"call_{index:04d}"
            call_ids.append(call_id)
            before = _state_view(workspace, task, observations, call_ids, "pending")
            result_text = ""
            metrics: dict[str, Any] = {}
            reward = 0.0
            tool_ok = False
            try:
                response, reward, raw_metrics = await self.tools[action["tool"]].execute(
                    episode_id,
                    deepcopy(action["arguments"]),
                    agent_data=agent_data,
                )
                result_text = _verifier_output(response)
                metrics = _json_safe(raw_metrics if isinstance(raw_metrics, dict) else {"value": raw_metrics})
                exit_code = metrics.get("exit_code") if isinstance(metrics, dict) else None
                tool_ok = not result_text.startswith("ERROR:") and exit_code in (None, 0)
                if isinstance(metrics, dict) and metrics.get("harness_status") == "fault":
                    harness_fault = str(metrics.get("failure_class", "tool_runtime"))
            except Exception as error:  # Tool adapter failures are harness faults.
                result_text = f"ERROR: {error}"
                metrics = {"harness_status": "fault", "failure_class": "tool_runtime"}
                harness_fault = "tool_runtime"

            observations.append(result_text)
            after = _state_view(workspace, task, observations, call_ids, "pending")
            events.append({
                "event_id": f"{episode_id}:event:{index:04d}",
                "kind": "tool_call",
                "tool_call_id": call_id,
                "action": deepcopy(action),
                "state_before": before,
                "state_after": after,
                "state_delta": _file_delta(before, after),
                "tool_result": {
                    "content": result_text,
                    "metrics": metrics,
                    "reward": float(reward),
                    "ok": tool_ok,
                },
                "provenance": {
                    "episode_id": episode_id,
                    "candidate_index": candidate_index,
                    "execution_mode": "workspace_host",
                },
            })
            messages.append(_action_message(call_id, action))
            messages.append({"role": "tool", "tool_call_id": call_id, "content": result_text})

        verifier_before = _state_view(workspace, task, observations, call_ids, "pending")
        verifier_text = ""
        verifier_metrics: dict[str, Any] = {}
        verifier_reward = 0.0
        try:
            response, verifier_reward, raw_metrics = await self.tools["verify_task"].execute(
                episode_id,
                {},
                agent_data=agent_data,
            )
            verifier_text = _verifier_output(response)
            verifier_metrics = _json_safe(raw_metrics if isinstance(raw_metrics, dict) else {"value": raw_metrics})
        except Exception as error:
            verifier_text = f"ERROR: {error}"
            verifier_metrics = {"harness_status": "fault", "failure_class": "verifier_runtime"}
            harness_fault = harness_fault or "verifier_runtime"
        verifier_outcome = agent_data.extra_fields.get("task_outcome")
        if not isinstance(verifier_outcome, dict):
            verifier_outcome = {
                "task_id": task["task_id"],
                "task_success": 0.0,
                "verifier_score": 0.0,
                "harness_status": "fault",
                "failure_class": "missing_verifier_outcome",
            }
        if harness_fault:
            verifier_outcome = {**verifier_outcome, "harness_status": "fault", "failure_class": harness_fault}

        verifier_after = _state_view(
            workspace,
            task,
            observations + [verifier_text],
            call_ids,
            "passed" if verifier_outcome.get("task_success") else "failed",
        )
        events.append({
            "event_id": f"{episode_id}:verifier",
            "kind": "independent_verifier",
            "state_before": verifier_before,
            "state_after": verifier_after,
            "state_delta": _file_delta(verifier_before, verifier_after),
            "verifier": {
                "command": task_verifier(task),
                "content": verifier_text,
                "metrics": verifier_metrics,
                "reward": float(verifier_reward),
                "outcome": verifier_outcome,
            },
            "provenance": {"episode_id": episode_id, "execution_mode": "workspace_host"},
        })

        passed = bool(verifier_outcome.get("task_success")) and verifier_outcome.get("harness_status") == "healthy"
        messages.append({"role": "assistant", "content": "DONE" if passed else "STOPPED"})
        protocol_valid = all(
            event.get("tool_call_id") and event.get("tool_result", {}).get("content") is not None
            for event in events if event["kind"] == "tool_call"
        )
        trace_fidelity = all(
            event.get("state_before", {}).get("state_hash")
            and event.get("state_after", {}).get("state_hash")
            and isinstance(event.get("state_delta"), dict)
            for event in events
        )
        outcome = {
            "task_success": bool(verifier_outcome.get("task_success")),
            "protocol_status": "valid" if protocol_valid else "invalid",
            "call_result_linkage_complete": protocol_valid,
            "trace_fidelity": trace_fidelity,
            "independent_verifier_passed": passed,
            "harness_status": "fault" if harness_fault else str(verifier_outcome.get("harness_status", "unknown")),
            "failure_class": None if passed else verifier_outcome.get("failure_class", "task_assertion"),
            "had_recoverable_tool_failure": any(
                not event["tool_result"]["ok"] for event in events if event["kind"] == "tool_call"
            ),
            "verifier_version": task.get("verifier_version", "manifest-v1"),
        }
        episode = {
            "episode_id": episode_id,
            "task_id": task["task_id"],
            "environment_id": f"{task['task_id']}:{fingerprint(task['files'])[:12]}",
            "initial_state_hash": initial_state["state_hash"],
            "candidate_index": candidate_index,
            "execution_mode": "workspace_host",
            "harness_version": task.get("harness_version", "workspace-host-v2"),
            "tool_schema_version": task.get("tool_schema_version", "workspace-tools-v2"),
            "verifier_version": task.get("verifier_version", "manifest-v1"),
            "actions": deepcopy(actions),
            "events": events,
            "messages": messages,
            "outcome": outcome,
            "workspace": str(workspace),
        }
        if not self.keep_workspaces:
            shutil.rmtree(workspace, ignore_errors=True)
        return episode

    def run(
        self,
        task: dict[str, Any],
        actions: list[dict[str, Any]],
        episode_id: str,
        *,
        candidate_index: int | None = None,
    ) -> dict[str, Any]:
        return asyncio.run(self._run(task, actions, episode_id, candidate_index=candidate_index))
