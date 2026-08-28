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


def _tool_metrics(info: dict[str, Any]) -> dict[str, float]:
    events = info.get("agent_events", [])
    if not isinstance(events, (list, tuple)):
        events = []
    calls = [event for event in events if isinstance(event, dict) and event.get("kind") == "tool_call"]
    results = [event for event in events if isinstance(event, dict) and event.get("kind") == "tool_result"]
    successful = sum(event.get("exit_code") == 0 for event in results)
    denominator = len(results) or len(calls)
    hit_rate = successful / denominator if denominator else 0.0
    return {
        "tool_calls": float(len(calls)),
        "tool_successes": float(successful),
        "tool_call_hit_rate": float(hit_rate),
        "tool_success_rate": float(hit_rate),
    }


def _result(
    score: float,
    eligible: bool,
    reason: str,
    source: str,
    status: str,
    *,
    task_id: Any = None,
    pass_rate: float | None = None,
    info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    score = max(0.0, min(1.0, score))
    result = {
        "score": score,
        # DAPO's group filter reads this from reward_extra_info. Keep it equal
        # to the unpenalized verifier score; the reward manager owns penalties.
        "seq_final_reward": score,
        "eligible": eligible,
        "reason": reason,
        "reward_source": source,
        "harness_status": status,
        "task_id": str(task_id) if task_id is not None else "unknown",
        "task_success": float(score),
        "pass_rate": float(score if pass_rate is None else pass_rate),
        "reward_mean": float(score),
    }
    result.update(_tool_metrics(info or {}))
    return result


def compute_reward(data_source: Any, solution_str: Any, ground_truth: Any, extra_info: Any = None, **_: Any) -> dict[str, Any]:
    """Return a reward plus its cause; never turn a harness fault into a label."""

    info = extra_info if isinstance(extra_info, dict) else {}
    outcome = info.get("task_outcome") or info.get("outcome") or {}
    outcome = outcome if isinstance(outcome, dict) else {}
    status = str(outcome.get("harness_status", info.get("harness_status", "missing"))).strip().lower()
    spec = _spec(ground_truth)
    task_id = outcome.get("task_id") or info.get("task_id") or spec.get("task_id") or data_source

    if status in FAULT_STATUSES:
        replay = _number(outcome.get("reference_task_success", outcome.get("replay_task_success")))
        if replay is not None:
            return _result(
                replay,
                True,
                "candidate harness fault; reference replay used",
                "reference_replay",
                status,
                task_id=task_id,
                pass_rate=replay,
                info=info,
            )
        return _result(
            0.0,
            False,
            "harness fault without reference replay",
            "unscored",
            status,
            task_id=task_id,
            pass_rate=0.0,
            info=info,
        )

    task_score = _number(outcome.get("task_success", outcome.get("verifier_score")))
    if task_score is not None:
        verifier_pass = _number(outcome.get("passed"))
        return _result(
            task_score,
            True,
            str(outcome.get("reason", "external task verifier")),
            "task_verifier",
            status,
            task_id=task_id,
            pass_rate=task_score if verifier_pass is None else verifier_pass,
            info=info,
        )

    if "tool_rewards" in info:
        return _result(
            0.0,
            True,
            "agent produced no verifier outcome",
            "protocol",
            "healthy",
            task_id=task_id,
            pass_rate=0.0,
            info=info,
        )

    if spec.get("kind") == "exact_text" or "expected_output" in spec:
        expected = str(spec.get("expected_output", spec.get("value", ""))).strip()
        actual = str(solution_str or "").strip()
        score = float(actual == expected)
        return _result(score, True, "exact text verifier", "exact_text", status, task_id=task_id, info=info)

    return _result(
        0.0,
        False,
        "missing task outcome/verifier",
        "unscored",
        status,
        task_id=task_id,
        pass_rate=0.0,
        info=info,
    )


if __name__ == "__main__":
    result = compute_reward("task", "done", "", {"task_outcome": {"task_success": 1}})
    assert result["score"] == result["seq_final_reward"] == 1.0
    assert compute_reward("task", "done", "", {"task_outcome": {"harness_status": "fault"}})["eligible"] is False
    assert compute_reward("task", "OK", "OK")["score"] == 1.0
    result = compute_reward(
        "task",
        "done",
        "",
        {
            "task_id": "task",
            "task_outcome": {"task_success": 1, "passed": True},
            "agent_events": [
                {"kind": "tool_call"},
                {"kind": "tool_result", "exit_code": 0},
                {"kind": "tool_call"},
                {"kind": "tool_result", "exit_code": 1},
            ],
        },
    )
    assert result["task_id"] == "task" and result["pass_rate"] == 1.0
    assert result["tool_call_hit_rate"] == 0.5 and result["tool_calls"] == 2.0
    print("reward adapter self-check passed")
