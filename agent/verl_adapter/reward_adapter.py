"""One reward contract for inference diagnostics and Verl rollouts."""

from __future__ import annotations

import json
import math
from typing import Any


FAULT_STATUSES = {"fault", "failed", "unknown", "unscored", "censored", "error", "timeout", "harness_fault"}


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, number)) if math.isfinite(number) else None


def _spec(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"kind": "exact_text", "value": value}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _result(score: float, eligible: bool, reason: str, source: str, status: str) -> dict[str, Any]:
    return {
        "score": max(0.0, min(1.0, score)),
        "eligible": eligible,
        "reason": reason,
        "reward_source": source,
        "harness_status": status,
    }


def compute_reward(data_source: Any, solution_str: Any, ground_truth: Any, extra_info: Any = None, **_: Any) -> dict[str, Any]:
    """Return a reward plus its cause; never turn a harness fault into a label."""

    info = extra_info if isinstance(extra_info, dict) else {}
    outcome = info.get("task_outcome") or info.get("outcome") or {}
    outcome = outcome if isinstance(outcome, dict) else {}
    status = str(outcome.get("harness_status", info.get("harness_status", "missing"))).strip().lower()

    if status in FAULT_STATUSES:
        replay = _number(outcome.get("reference_task_success", outcome.get("replay_task_success")))
        if replay is not None:
            return _result(replay, True, "candidate harness fault; reference replay used", "reference_replay", status)
        return _result(0.0, False, "harness fault without reference replay", "unscored", status)

    task_score = _number(outcome.get("task_success", outcome.get("verifier_score")))
    if task_score is not None:
        return _result(task_score, True, str(outcome.get("reason", "external task verifier")), "task_verifier", status)

    if "tool_rewards" in info:
        return _result(0.0, True, "agent produced no verifier outcome", "protocol", "healthy")

    spec = _spec(ground_truth)
    if spec.get("kind") == "exact_text" or "expected_output" in spec:
        expected = str(spec.get("expected_output", spec.get("value", ""))).strip()
        actual = str(solution_str or "").strip()
        return _result(float(actual == expected), True, "exact text verifier", "exact_text", status)

    return _result(0.0, False, "missing task outcome/verifier", "unscored", status)


if __name__ == "__main__":
    assert compute_reward("task", "done", "", {"task_outcome": {"task_success": 1}})["score"] == 1.0
    assert compute_reward("task", "done", "", {"task_outcome": {"harness_status": "fault"}})["eligible"] is False
    assert compute_reward("task", "OK", "OK")["score"] == 1.0
    print("reward adapter self-check passed")
