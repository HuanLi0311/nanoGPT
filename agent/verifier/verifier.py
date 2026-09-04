"""Out-of-band command verifier with explicit task/system failure classes."""

from __future__ import annotations

if __name__ == "__main__" and __package__ is None:
    # Keep the package's types.py from shadowing the stdlib module named
    # types during direct self-check execution.
    import os
    import sys

    script_dir = os.path.abspath(os.path.dirname(__file__))
    sys.path[:] = [item for item in sys.path if os.path.abspath(item or os.curdir) != script_dir]
    sys.path.insert(0, os.getcwd())

import subprocess
from pathlib import Path
from typing import Any

try:
    from .types import VerificationResult
except ImportError:  # direct script execution
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from agent.verifier.types import VerificationResult


def _command(verifier: Any) -> str | None:
    if isinstance(verifier, str) and verifier.strip():
        return verifier.strip()
    if isinstance(verifier, dict) and isinstance(verifier.get("command"), str) and verifier["command"].strip():
        return verifier["command"].strip()
    return None


def run_verifier(
    root: str,
    verifier: Any,
    *,
    task_id: str | None = None,
    verifier_version: str = "manifest-v1",
    timeout: int = 60,
    max_output: int = 12_000,
    sandbox_backend: str = "workspace_host",
) -> dict[str, Any]:
    """Run a trusted manifest verifier and return a traceable result.

    New synthesis tasks use Bubblewrap; legacy manifests retain the host-shell
    backend for compatibility.
    """

    command = _command(verifier)
    if command is None:
        return VerificationResult(
            0.0,
            False,
            "missing verifier command",
            reward_source="unscored",
            harness_status="fault",
            failure_class="missing_verifier",
            eligible=False,
            task_id=task_id,
            verifier_version=verifier_version,
        ).as_dict()
    if sandbox_backend not in {"bwrap", "workspace_host"}:
        return VerificationResult(
            0.0,
            False,
            f"unknown sandbox backend: {sandbox_backend}",
            reward_source="unscored",
            harness_status="fault",
            failure_class="verifier_runtime",
            eligible=False,
            task_id=task_id,
            verifier_version=verifier_version,
        ).as_dict()
    try:
        if sandbox_backend == "bwrap":
            from agent.workspace.sandbox import bubblewrap_command

            invocation, shell, cwd = bubblewrap_command(Path(root), command), False, None
        else:
            invocation, shell, cwd = command, True, root
        result = subprocess.run(
            invocation,
            cwd=cwd,
            shell=shell,
            env={} if sandbox_backend == "bwrap" else None,
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout)),
            check=False,
        )
        if sandbox_backend == "bwrap" and result.returncode and result.stderr.startswith("bwrap:"):
            raise RuntimeError(result.stderr.strip())
    except subprocess.TimeoutExpired as error:
        return VerificationResult(
            0.0,
            False,
            f"verifier timed out after {timeout}s",
            reward_source="unscored",
            harness_status="fault",
            failure_class="verifier_timeout",
            eligible=False,
            task_id=task_id,
            verifier_version=verifier_version,
            output=str(error),
        ).as_dict()
    except (OSError, RuntimeError, ValueError) as error:
        return VerificationResult(
            0.0,
            False,
            f"verifier runtime failure: {error}",
            reward_source="unscored",
            harness_status="fault",
            failure_class="verifier_runtime",
            eligible=False,
            task_id=task_id,
            verifier_version=verifier_version,
            output=str(error),
        ).as_dict()

    output = (result.stdout + result.stderr)[-max(256, int(max_output)) :]
    if result.returncode == 0:
        return VerificationResult(
            1.0,
            True,
            "external task verifier passed",
            task_id=task_id,
            verifier_version=verifier_version,
            returncode=result.returncode,
            output=output,
        ).as_dict()

    # Convention: 2 and shell command-not-found/permission codes indicate a
    # broken verifier; ordinary non-zero assertions are task failures.
    runtime_fault = result.returncode in {2, 126, 127} or result.returncode < 0
    return VerificationResult(
        0.0,
        False,
        "verifier runtime failure" if runtime_fault else "task postcondition failed",
        reward_source="unscored" if runtime_fault else "task_verifier",
        harness_status="fault" if runtime_fault else "healthy",
        failure_class="verifier_runtime" if runtime_fault else "task_assertion",
        eligible=not runtime_fault,
        task_id=task_id,
        verifier_version=verifier_version,
        returncode=result.returncode,
        output=output,
    ).as_dict()


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        passed = run_verifier(directory, "true")
        failed = run_verifier(directory, "false")
        missing = run_verifier(directory, None)
        assert passed["task_success"] == 1.0 and passed["harness_status"] == "healthy"
        assert failed["failure_class"] == "task_assertion" and failed["eligible"] is True
        assert missing["failure_class"] == "missing_verifier" and missing["eligible"] is False
    print("verifier self-check passed")
