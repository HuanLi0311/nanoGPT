"""Small deterministic reward hook for verl's custom_reward_function."""

from __future__ import annotations

import json


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    text = str(solution_str or "").strip()
    if not text:
        return 0.0
    score = 0.25
    try:
        value = json.loads(text)
        if isinstance(value, dict) and ("message" in value or "tool_call" in value or "name" in value):
            score += 0.5
    except json.JSONDecodeError:
        score += 0.25
    if "```" in text:
        score -= 0.1
    if len(text) <= 4000:
        score += 0.25
    return max(0.0, min(1.0, score))


if __name__ == "__main__":
    assert compute_score("codex", '{"message":"done"}', "") == 1.0
