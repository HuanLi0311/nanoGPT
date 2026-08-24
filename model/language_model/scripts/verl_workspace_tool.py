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
        try:
            root = self._episode_root(instance_id, config, agent_data)
            if self.operation != "verify_task":
                # A later edit or command changes the state that was verified;
                # require a fresh postcondition before awarding the episode.
                self._invalidate_outcome(agent_data)
            if self.operation == "exec_command":
                return self._exec(root, parameters)
            if self.operation == "apply_patch":
                return self._patch(root, parameters)
            if self.operation == "verify_task":
                return self._verify(root, config, agent_data)
            return self._fault(config, agent_data, "unknown_operation", f"unknown workspace operation: {self.operation}")
        except subprocess.TimeoutExpired:
            return self._fault(config, agent_data, "timeout", f"command timed out after {self.max_timeout}s")
        except OSError as error:
            return self._fault(config, agent_data, "tool_runtime", str(error))
        except Exception as error:
            return self._error(str(error))

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
        candidate = (root / str(relative or ".")).resolve()
        if candidate != root and root not in candidate.parents:
            raise ValueError(f"path escapes workspace: {relative}")
        return candidate

    def _exec(self, root: Path, parameters: dict[str, Any]) -> tuple[ToolResponse, float, dict]:
        command = str(parameters.get("command", "")).strip()
        if not command:
            return self._error("command is required")
        # ponytail: shell execution is workspace-scoped only; use an OS/container
        # sandbox before running untrusted workloads at scale.
        cwd = self._safe_path(root, parameters.get("cwd", "."))
        cwd.mkdir(parents=True, exist_ok=True)
        timeout = min(self.max_timeout, max(1, int(parameters.get("timeout", self.max_timeout))))
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
        return ToolResponse(text=f"exit_code={result.returncode}\n{output}"), 0.0, {"exit_code": result.returncode}

    def _patch(self, root: Path, parameters: dict[str, Any]) -> tuple[ToolResponse, float, dict]:
        patch = str(parameters.get("patch", ""))
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
        return ToolResponse(text=output or "patch applied"), 0.0, {}

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
        return ToolResponse(text=f"applied {len(blocks)} file(s)"), 0.0, {}

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
        verifier = config.get("verifier")
        command = verifier.get("command") if isinstance(verifier, dict) else verifier
        if not isinstance(command, str) or not command.strip():
            outcome = {
                "task_id": config.get("task_id"),
                "task_success": 0.0,
                "verifier_score": 0.0,
                "harness_status": "fault",
                "protocol_status": "valid",
                "tool_status": "failed",
                "failure_class": "missing_verifier",
                "verifier_version": config.get("verifier_version", "manifest-v1"),
            }
            self._publish_outcome(agent_data, outcome)
            return ToolResponse(text=json.dumps(outcome)), 0.0, {}
        result = subprocess.run(
            command,
            cwd=root,
            shell=True,
            capture_output=True,
            text=True,
            timeout=self.max_timeout,
            check=False,
        )
        passed = result.returncode == 0
        verifier_fault = result.returncode in {2, 126, 127} or result.returncode < 0
        output = (result.stdout + result.stderr)[-self.max_output :]
        outcome = {
            "task_id": config.get("task_id"),
            "task_success": float(passed),
            "verifier_score": float(passed),
            "harness_status": "fault" if verifier_fault else "healthy",
            "protocol_status": "valid",
            "tool_status": "completed",
            "failure_class": None if passed else ("verifier_runtime" if verifier_fault else "task_assertion"),
            "verifier_version": config.get("verifier_version", "manifest-v1"),
            "verifier_returncode": result.returncode,
            "verifier_output": output,
        }
        self._publish_outcome(agent_data, outcome)
        return ToolResponse(text=json.dumps({"passed": passed, "output": output})), float(passed), {}

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
        }
        self._publish_outcome(agent_data, outcome)
        return ToolResponse(text=f"ERROR: {message}"), 0.0, {"harness_status": "fault"}

    @staticmethod
    def _error(message: str) -> tuple[ToolResponse, float, dict]:
        return ToolResponse(text=f"ERROR: {message}"), 0.0, {"harness_status": "healthy"}


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
                _schema("exec_command", "run a command", {"command": {"type": "string"}}, ["command"]),
            )
            instance, _ = await tool.create(create_kwargs={"task_id": "check", "files": {"x.txt": "ok\n"}})
            response, _, _ = await tool.execute(instance, {"command": "test -f x.txt"})
            assert "exit_code=0" in (response.text or "")
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
            values = list((root / "patch").rglob("value.txt"))
            assert values and values[0].read_text(encoding="utf-8") == "NEW\n"
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
