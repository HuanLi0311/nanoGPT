"""Structured verifier results shared by local and Verl execution."""

from __future__ import annotations

if __name__ == "__main__" and __package__ is None:
    # This file is intentionally named types.py for the package API. Remove
    # its directory first so direct self-checks do not shadow stdlib types.
    import os
    import sys

    script_dir = os.path.abspath(os.path.dirname(__file__))
    sys.path[:] = [item for item in sys.path if os.path.abspath(item or os.curdir) != script_dir]
    sys.path.insert(0, os.getcwd())

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class VerificationResult:
    score: float
    passed: bool
    reason: str
    reward_source: str = "task_verifier"
    harness_status: str = "healthy"
    failure_class: str | None = None
    eligible: bool = True
    task_id: str | None = None
    verifier_version: str = "manifest-v1"
    returncode: int | None = None
    output: str = ""

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        # Keep the names consumed by existing Verl reward code as aliases.
        result.update(
            task_success=float(self.score if self.passed else 0.0),
            verifier_score=float(self.score),
            verifier_returncode=self.returncode,
            verifier_output=self.output,
            protocol_status="valid",
            tool_status="completed",
        )
        return result


if __name__ == "__main__":
    assert VerificationResult(1.0, True, "ok").as_dict()["task_success"] == 1.0
    print("verifier types self-check passed")
