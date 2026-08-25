"""Retry manager for reward samples whose harness outcome is censored."""

from __future__ import annotations

import os

import numpy as np

from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager


class RetryOnIneligibleRewardManager(NaiveRewardManager):
    """Do not let a harness-fault sample reach GRPO; ask the replay buffer to refill it."""

    async def run_single(self, data):
        # NaiveRewardManager merges tool fields into extra_info in place.  The
        # same dataset dict can be reused by rollout.n sibling trajectories, so
        # isolate that mutable envelope before computing this sample's reward.
        original = data.non_tensor_batch.get("extra_info")
        if isinstance(original, np.ndarray):
            isolated = np.empty(len(original), dtype=object)
            isolated[:] = [dict(value) if isinstance(value, dict) else value for value in original]
            data.non_tensor_batch["extra_info"] = isolated
        try:
            result = await super().run_single(data)
        finally:
            if original is not None:
                data.non_tensor_batch["extra_info"] = original
        extra = result.get("reward_extra_info", {}) if isinstance(result, dict) else {}
        reward_kwargs = self.config.get("reward", {}).get("reward_kwargs", {})
        overlong = reward_kwargs.get("overlong_buffer_cfg", {})
        if overlong.get("enable", False):
            response_ids = data[-1].batch["responses"]
            attention = data[-1].batch["attention_mask"]
            response_length = response_ids.shape[-1]
            valid_length = int(attention[-response_length:].sum())
            buffer_len = int(overlong.get("len", 0))
            max_response = int(reward_kwargs.get("max_resp_len", 0))
            if buffer_len <= 0 or max_response < buffer_len:
                raise RuntimeError("invalid DAPO overlong buffer configuration")
            exceed = valid_length - (max_response - buffer_len)
            penalty = min(-exceed / buffer_len * float(overlong.get("penalty_factor", 1.0)), 0.0)
            result["reward_score"] += penalty
            if overlong.get("log", False):
                extra["overlong_reward"] = penalty
                extra["overlong"] = penalty < 0
        if extra.get("eligible") is False and os.environ.get("VERL_RETRY_CENSORED", "1") != "0":
            reason = extra.get("reason", "censored reward")
            raise RuntimeError(f"retryable reward sample: {reason}")
        return result


if __name__ == "__main__":
    assert issubclass(RetryOnIneligibleRewardManager, NaiveRewardManager)
