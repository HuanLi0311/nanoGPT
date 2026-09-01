"""Small stateful workspace adapter for verl's ``tool_agent`` loop.

The training engine owns generation and token masks.  This module only owns the
episode workspace, model-facing tool ABI, and the independent verifier call.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

try:
    from agent.workspace.boundary import workspace_path
    from agent.workspace.snapshot import snapshot
    from agent.verifier.verifier import run_verifier
    from agent.verl_adapter.loop_adapter import record_tool_event
except ModuleNotFoundError:  # direct `python path/to/verl_workspace_tool.py`
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from agent.workspace.boundary import workspace_path
    from agent.workspace.snapshot import snapshot
    from agent.verifier.verifier import run_verifier
    from agent.verl_adapter.loop_adapter import record_tool_event


_SAFE_PART = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_part(value: Any, default: str) -> str:
    result = _SAFE_PART.sub("_", str(value or default)).strip("._")
    return result or default


class WorkspaceTool(BaseTool):
    """Execute one configured workspace operation for an agent episode."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.operation = str(config["operation"])
        self.workspace_root = Path(config.get("workspace_root") or os.environ.get("VERL_WORKSPACE_ROOT", "/tmp/verl-workspaces"))
        self.max_timeout = max(1, int(config.get("max_timeout", 60)))
        self.max_output = max(256, int(config.get("max_output", 12000)))
        self._configs: dict[str, dict[str, Any]] = {}
        self._roots: dict[str, Path] = {}

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        instance_id = instance_id or uuid.uuid4().hex
        create_kwargs = kwargs.get("create_kwargs", {})
        self._configs[instance_id] = create_kwargs if isinstance(create_kwargs, dict) else {}
        return instance_id, ToolResponse(text="workspace ready")

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        config = self._configs.get(instance_id, {})
        agent_data = kwargs.get("agent_data")
        parameters = self._canonical_parameters(parameters)
        root: Optional[Path] = None
        try:
            root = self._episode_root(instance_id, config, agent_data)
            if self.operation == "verify_task":
                return self._verify(root, config, agent_data)
            if self.operation not in {"exec_command", "apply_patch"}:
                return self._fault(config, agent_data, "unknown_operation", f"unknown workspace operation: {self.operation}")

            # A later action invalidates the previous result. The verifier below
            # then runs out-of-band after the action, never as a model tool.
            self._invalidate_outcome(agent_data)
            before = snapshot(root)
            response, reward, result = self._exec(root, parameters) if self.operation == "exec_command" else self._patch(root, parameters)
            after = snapshot(root)
            result = dict(result or {})
            result.setdefault("exit_code", 0)
            result.setdefault("tool_status", "completed" if result["exit_code"] == 0 else "failed")
            record_tool_event(
                agent_data,
                operation=self.operation,
                arguments=parameters,
                result=result,
                state_before=before,
                state_after=after,
            )
            self._auto_verify(root, config, agent_data, result)
            return response, reward, result
        except subprocess.TimeoutExpired:
            return self._runtime_fault(root, config, agent_data, parameters, "timeout", f"command timed out after {self.max_timeout}s")
        except OSError as error:
            return self._runtime_fault(root, config, agent_data, parameters, "tool_runtime", str(error))
        except ValueError as error:
            # Invalid arguments or a non-matching patch are policy/tool
            # failures, not a broken harness.
            if root is not None:
                before = snapshot(root)
                record_tool_event(
                    agent_data,
                    operation=self.operation,
                    arguments=parameters,
                    result={"exit_code": 1, "tool_status": "failed", "failure_class": "tool_failure", "output": str(error)},
                    state_before=before,
                    state_after=before,
                )
                self._auto_verify(
                    root,
                    config,
                    agent_data,
                    {"exit_code": 1, "tool_status": "failed", "failure_class": "tool_failure"},
                )
            return self._error(str(error))
        except Exception as error:
            return self._runtime_fault(root, config, agent_data, parameters, "tool_runtime", str(error))

    async def release(self, instance_id: str, **kwargs) -> None:
        self._configs.pop(instance_id, None)

    def _episode_root(self, instance_id: str, config: dict[str, Any], agent_data: Any) -> Path:
        if instance_id in self._roots:
            return self._roots[instance_id]
        request_id = getattr(agent_data, "request_id", None) or instance_id
        task_id = _safe_part(config.get("task_id"), "task")
        episode_id = _safe_part(request_id, "episode")
        root = self.workspace_root / task_id / episode_id
        root.mkdir(parents=True, exist_ok=True)
        self._initialize_files(root, config.get("files", {}))
        if agent_data is not None and hasattr(agent_data, "extra_fields"):
            initial = snapshot(root)
            agent_data.extra_fields.setdefault("task_id", task_id)
            agent_data.extra_fields.setdefault("environment_id", f"workspace:{task_id}")
            agent_data.extra_fields.setdefault("initial_state_hash", initial["state_hash"])
            agent_data.extra_fields.setdefault("harness_version", "nanoagent-verl-v1")
            agent_data.extra_fields.setdefault("execution_mode", "workspace_host")
            agent_data.extra_fields.setdefault("tool_schema_version", str(config.get("tool_schema_version", "workspace-tools-v2")))
            agent_data.extra_fields.setdefault("verifier_version", str(config.get("verifier_version", "manifest-v1")))
        self._roots[instance_id] = root
        return root

    @staticmethod
    def _initialize_files(root: Path, files: Any) -> None:
        if not isinstance(files, dict):
            return
        for relative, content in files.items():
            path = WorkspaceTool._safe_path(root, relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_text(str(content), encoding="utf-8")

    @staticmethod
    def _safe_path(root: Path, relative: Any) -> Path:
        return workspace_path(root, relative)

    @staticmethod
    def _canonical_parameters(parameters: Any) -> Any:
        """Accept old aliases at the adapter edge; record only the Codex ABI."""

        if not isinstance(parameters, dict):
            return parameters
        normalized = dict(parameters)
        if "cmd" not in normalized and "command" in normalized:
            normalized["cmd"] = normalized.pop("command")
        if "workdir" not in normalized and "cwd" in normalized:
            normalized["workdir"] = normalized.pop("cwd")
        return normalized

    def _exec(self, root: Path, parameters: dict[str, Any]) -> tuple[ToolResponse, float, dict]:
        command = str(parameters.get("cmd") or "").strip()
        if not command:
            return self._error("cmd is required")
        # ponytail: shell execution is workspace-scoped only; use an OS/container
        # sandbox before running untrusted workloads at scale.
        cwd_value = parameters.get("workdir", ".")
        cwd = self._safe_path(root, cwd_value)
        cwd.mkdir(parents=True, exist_ok=True)
        timeout_value = parameters.get("timeout")
        if timeout_value is None and parameters.get("timeout_ms") is not None:
            timeout_value = max(1, int(parameters["timeout_ms"]) / 1000)
        timeout = min(self.max_timeout, max(1, int(timeout_value or self.max_timeout)))
        result = subprocess.run(
            command,
            cwd=cwd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        output = (result.stdout + result.stderr)[-self.max_output :]
        return ToolResponse(text=f"exit_code={result.returncode}\n{output}"), 0.0, {
            "exit_code": result.returncode,
            "output": output,
        }

    def _patch(self, root: Path, parameters: dict[str, Any] | str) -> tuple[ToolResponse, float, dict]:
        patch = parameters if isinstance(parameters, str) else str(parameters.get("patch", ""))
        if not patch.strip():
            return self._error("patch is required")
        if patch.lstrip().startswith("*** Begin Patch"):
            return self._codex_patch(root, patch)
        paths = re.findall(r"^(?:---|\+\+\+) (?:[ab]/)?([^\t\n]+)$", patch, re.MULTILINE)
        if not paths and "diff --git a/" not in patch:
            return self._error("patch must contain file paths")
        for path in paths:
            if path != "/dev/null":
                self._safe_path(root, path)
        result = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", "-"],
            cwd=root,
            input=patch,
            capture_output=True,
            text=True,
            timeout=self.max_timeout,
            check=False,
        )
        output = (result.stdout + result.stderr)[-self.max_output :]
        if result.returncode:
            return self._error(f"git apply failed ({result.returncode}): {output}")
        return ToolResponse(text=output or "patch applied"), 0.0, {"exit_code": 0, "output": output}

    def _codex_patch(self, root: Path, patch: str) -> tuple[ToolResponse, float, dict]:
        blocks: list[tuple[str, str, list[str]]] = []
        operation, path, body = None, None, []
        for line in patch.splitlines():
            header = re.match(r"\*\*\* (Add|Update|Delete) File: (.+)$", line)
            if header:
                if operation is not None:
                    blocks.append((operation, path, body))
                operation, path, body = header.group(1), header.group(2), []
            elif line == "*** End Patch":
                break
            elif operation is not None:
                body.append(line)
        if operation is not None:
            blocks.append((operation, path, body))
        if not blocks:
            return self._error("Codex patch has no file operations")
        files = [(operation, self._safe_path(root, path)) for operation, path, _ in blocks]
        for (operation, path), (_, _, body) in zip(files, blocks, strict=True):
            if operation == "Add":
                if path.exists():
                    return self._error(f"file exists: {path.name}")
                path.parent.mkdir(parents=True, exist_ok=True)
                content = [line[1:] for line in body if line.startswith("+")]
                path.write_text("\n".join(content) + "\n", encoding="utf-8")
            elif operation == "Delete":
                path.unlink()
            else:
                self._apply_codex_update(path, body)
        return ToolResponse(text=f"applied {len(blocks)} file(s)"), 0.0, {"exit_code": 0}

    @staticmethod
    def _apply_codex_update(path: Path, body: list[str]) -> None:
        current = path.read_text(encoding="utf-8").splitlines()
        cursor = 0
        hunks: list[list[str]] = []
        hunk: list[str] = []
        for line in body:
            if line.startswith("@@"):
                if hunk:
                    hunks.append(hunk)
                hunk = []
            elif line and line != "*** End of File":
                hunk.append(line)
        if hunk:
            hunks.append(hunk)
        for change in hunks:
            old_lines = [line[1:] for line in change if line[0] in " -"]
            new_lines = [line[1:] for line in change if line[0] in " +"]
            at = WorkspaceTool._find_lines(current, old_lines, cursor)
            if at < 0:
                raise ValueError(f"hunk does not match: {path.name}")
            current[at : at + len(old_lines)] = new_lines
            cursor = at + len(new_lines)
        path.write_text("\n".join(current) + "\n", encoding="utf-8")

    @staticmethod
    def _find_lines(lines: list[str], wanted: list[str], start: int) -> int:
        for index in range(start, len(lines) - len(wanted) + 1):
            if lines[index : index + len(wanted)] == wanted:
                return index
        return -1

    def _verify(self, root: Path, config: dict[str, Any], agent_data: Any) -> tuple[ToolResponse, float, dict]:
        outcome = run_verifier(
            str(root),
            config.get("verifier"),
            task_id=str(config.get("task_id")) if config.get("task_id") is not None else None,
            verifier_version=str(config.get("verifier_version", "manifest-v1")),
            timeout=self.max_timeout,
            max_output=self.max_output,
        )
        self._publish_outcome(agent_data, outcome)
        return ToolResponse(text=json.dumps(outcome)), float(outcome.get("score", 0.0)), {
            "exit_code": 0 if outcome.get("harness_status") == "healthy" else 1,
            "harness_status": outcome.get("harness_status"),
        }

    def _auto_verify(self, root: Path, config: dict[str, Any], agent_data: Any, tool_result: dict[str, Any]) -> None:
        # Standalone tool users may intentionally omit a verifier; the RL data
        # validator rejects that case before training, while local tool smoke
        # tests keep the outcome unset until an explicit verifier is supplied.
        if not config.get("verifier"):
            return
        outcome = run_verifier(
            str(root),
            config.get("verifier"),
            task_id=str(config.get("task_id")) if config.get("task_id") is not None else None,
            verifier_version=str(config.get("verifier_version", "manifest-v1")),
            timeout=self.max_timeout,
            max_output=self.max_output,
        )
        outcome["tool_status"] = tool_result.get("tool_status", "completed")
        outcome["tool_result"] = tool_result
        self._publish_outcome(agent_data, outcome)

    def _runtime_fault(
        self,
        root: Optional[Path],
        config: dict[str, Any],
        agent_data: Any,
        parameters: dict[str, Any],
        failure_class: str,
        message: str,
    ):
        if root is not None:
            state = snapshot(root)
            record_tool_event(
                agent_data,
                operation=self.operation,
                arguments=parameters,
                result={"exit_code": 1, "tool_status": "failed", "failure_class": failure_class, "output": message},
                state_before=state,
                state_after=state,
            )
        return self._fault(config, agent_data, failure_class, message)

    @staticmethod
    def _publish_outcome(agent_data: Any, outcome: dict[str, Any]) -> None:
        if agent_data is not None and hasattr(agent_data, "extra_fields"):
            agent_data.extra_fields["task_outcome"] = outcome

    @staticmethod
    def _invalidate_outcome(agent_data: Any) -> None:
        if agent_data is not None and hasattr(agent_data, "extra_fields"):
            agent_data.extra_fields.pop("task_outcome", None)

    def _fault(self, config: dict[str, Any], agent_data: Any, failure_class: str, message: str):
        outcome = {
            "task_id": config.get("task_id"),
            "task_success": 0.0,
            "verifier_score": 0.0,
            "harness_status": "fault",
            "protocol_status": "valid",
            "tool_status": "failed",
            "failure_class": failure_class,
            "verifier_version": config.get("verifier_version", "manifest-v1"),
            "verifier_output": message,
            "reward_source": "unscored",
            "eligible": False,
            "protocol_status": "valid",
        }
        self._publish_outcome(agent_data, outcome)
        return ToolResponse(text=f"ERROR: {message}"), 0.0, {"harness_status": "fault"}

    @staticmethod
    def _error(message: str) -> tuple[ToolResponse, float, dict]:
        return ToolResponse(text=f"ERROR: {message}"), 0.0, {
            "exit_code": 1,
            "harness_status": "healthy",
            "tool_status": "failed",
            "failure_class": "tool_failure",
            "output": message,
        }


def _schema(name: str, description: str, properties: dict, required: list[str]) -> OpenAIFunctionToolSchema:
    return OpenAIFunctionToolSchema.model_validate({
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    })


if __name__ == "__main__":
    async def check() -> None:
        root = Path(tempfile.mkdtemp(prefix="verl-workspace-check-"))
        try:
            tool = WorkspaceTool(
                {"operation": "exec_command", "workspace_root": str(root)},
                _schema("exec_command", "run a command", {"cmd": {"type": "string"}}, ["cmd"]),
            )
            instance, _ = await tool.create(create_kwargs={"task_id": "check", "files": {"x.txt": "ok\n"}})
            response, _, _ = await tool.execute(instance, {"cmd": "test -f x.txt"})
            assert "exit_code=0" in (response.text or "")
            response, _, _ = await tool.execute(instance, {"cmd": "test -f x.txt", "workdir": ".", "timeout_ms": 1000})
            assert "exit_code=0" in (response.text or "")
            response, _, _ = await tool.execute(
                instance, {"cmd": "test -f x.txt", "workdir": str(tool._roots[instance])}
            )
            assert "exit_code=0" in (response.text or "")
            response, _, _ = await tool.execute(instance, {"cmd": "test -f x.txt", "workdir": "/workspace"})
            assert "exit_code=0" in (response.text or "")
            response, _, _ = await tool.execute(instance, {"cmd": "true", "workdir": "/outside/workspace"})
            assert "path escapes workspace" in (response.text or "")
            patch_tool = WorkspaceTool(
                {"operation": "apply_patch", "workspace_root": str(root)},
                _schema("apply_patch", "apply a patch", {"patch": {"type": "string"}}, ["patch"]),
            )
            patch_instance, _ = await patch_tool.create(create_kwargs={"task_id": "patch"})
            await patch_tool.execute(
                patch_instance,
                {"patch": "*** Begin Patch\n*** Add File: value.txt\n+OLD\n*** End Patch"},
            )
            await patch_tool.execute(
                patch_instance,
                {"patch": "*** Begin Patch\n*** Update File: value.txt\n@@\n-OLD\n+NEW\n*** End Patch"},
            )
            await patch_tool.execute(
                patch_instance,
                "*** Begin Patch\n*** Add File: raw.txt\n+RAW\n*** End Patch",
            )
            values = list((root / "patch").rglob("value.txt"))
            raw_values = list((root / "patch").rglob("raw.txt"))
            assert values and values[0].read_text(encoding="utf-8") == "NEW\n"
            assert raw_values and raw_values[0].read_text(encoding="utf-8") == "RAW\n"
            verify_tool = WorkspaceTool(
                {"operation": "verify_task", "workspace_root": str(root)},
                _schema("verify_task", "verify", {}, []),
            )
            verify_instance, _ = await verify_tool.create(
                create_kwargs={"task_id": "verify", "files": {"ok.txt": "ok\n"}, "verifier": "test -f ok.txt"}
            )
            agent_data = SimpleNamespace(request_id="verify-episode", extra_fields={})
            response, reward, _ = await verify_tool.execute(verify_instance, {}, agent_data=agent_data)
            assert json.loads(response.text or "{}")["passed"] is True and reward == 1.0
            assert agent_data.extra_fields["task_outcome"]["task_success"] == 1.0
            exec_instance, _ = await tool.create(create_kwargs={"task_id": "verify"})
            await tool.execute(exec_instance, {"command": "true"}, agent_data=agent_data)
            assert "task_outcome" not in agent_data.extra_fields
            fault_instance, _ = await verify_tool.create(
                create_kwargs={"task_id": "fault", "verifier": "command_that_does_not_exist_123"}
            )
            await verify_tool.execute(fault_instance, {}, agent_data=agent_data)
            assert agent_data.extra_fields["task_outcome"]["harness_status"] == "fault"
        finally:
            shutil.rmtree(root, ignore_errors=True)

    asyncio.run(check())
