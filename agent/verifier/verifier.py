"""Out-of-band command verifier with explicit task/system failure classes."""

from __future__ import annotations

import subprocess
from typing import Any

try:
    from .types import VerificationResult
except ImportError:  # direct script execution
    import sys
    from pathlib import Path

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
) -> dict[str, Any]:
    """Run a trusted manifest verifier and return a traceable result.

    ponytail: manifest commands intentionally use the platform shell because
    existing task files already specify shell snippets; put untrusted tasks in
    a container before scaling this runner.
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
    try:
        result = subprocess.run(
            command,
            cwd=root,
            shell=True,
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout)),
            check=False,
        )
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
    except OSError as error:
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
