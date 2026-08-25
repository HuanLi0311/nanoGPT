"""Verl entry point for the shared, traceable agent reward contract."""

from __future__ import annotations

try:
    from agent.verl_adapter.reward_adapter import compute_reward
except ModuleNotFoundError:  # direct script execution
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from agent.verl_adapter.reward_adapter import compute_reward


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    return compute_reward(data_source, solution_str, ground_truth, extra_info, **kwargs)


if __name__ == "__main__":
    assert compute_score("task", "done", "", {"task_outcome": {"harness_status": "healthy", "task_success": 1}})["score"] == 1.0
    assert compute_score("task", "done", "", {})["eligible"] is False
    assert compute_score("task", "OK", "OK")["score"] == 1.0
