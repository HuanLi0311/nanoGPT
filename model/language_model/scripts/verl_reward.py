"""Task-outcome reward for agent rollouts.

Historical Codex rows do not contain a workspace verifier, so they are reported
as unscored instead of receiving a format-shaped reward.
"""

from __future__ import annotations

import json
import math
from typing import Any


FAULT_STATUSES = {"fault", "failed", "unknown", "unscored", "censored", "error", "timeout", "harness_fault"}


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    try:
        value = float(value)
        return max(0.0, min(1.0, value)) if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


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


def _result(score: float, eligible: bool, reason: str, source: str, status: str = "missing") -> dict[str, Any]:
    return {
        "score": max(0.0, min(1.0, score)),
        "eligible": eligible,
        "reason": reason,
        "reward_source": source,
        "harness_status": status,
    }


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Return task outcome, with reference replay as a harness-fault fallback.

    The agent loop should put a dict in ``extra_info['task_outcome']`` with
    ``task_success`` and ``harness_status``. ``ground_truth`` is only used for
    simple exact-text tasks; it is not a substitute for an environment verifier.
    """
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
        return _result(task_score, True, "external task verifier", "task_verifier", status)

    # A tool-agent trajectory that reaches a final answer without calling the
    # verifier is a policy/protocol failure, not a reason to train on an
    # invented harness failure.  The tool loop supplies this field even when
    # the list is empty; a completely missing field remains censored.
    if "tool_rewards" in info:
        return _result(0.0, True, "agent produced no verifier outcome", "protocol", "healthy")

    spec = _spec(ground_truth)
    if spec.get("kind") == "exact_text" or "expected_output" in spec:
        expected = str(spec.get("expected_output", spec.get("value", ""))).strip()
        actual = str(solution_str or "").strip()
        return _result(float(actual == expected), True, "exact text verifier", "exact_text", status)

    return _result(0.0, False, "missing task outcome/verifier", "unscored", status)


if __name__ == "__main__":
    assert compute_score("task", "done", "", {"task_outcome": {"harness_status": "healthy", "task_success": 1}})["score"] == 1.0
    assert compute_score("task", "done", "", {"task_outcome": {"harness_status": "fault", "reference_task_success": 1}})["reward_source"] == "reference_replay"
    assert compute_score("task", "done", "", {})["eligible"] is False
    assert compute_score("task", "done", "", {"task_outcome": {"harness_status": "timeout"}})["eligible"] is False
    assert compute_score("task", "done", "", {"tool_rewards": []})["reward_source"] == "protocol"
    assert compute_score("task", "OK", "OK")["score"] == 1.0
