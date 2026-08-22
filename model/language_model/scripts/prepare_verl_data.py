"""Convert filtered Codex rollouts to the verl GRPO row contract.

``ground_truth`` is a verifier contract, not a gold assistant response.
Historical rows without an independent verifier remain explicitly censored.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _contract(item: dict, path: Path, metadata: dict, episode_index: int) -> tuple[str, dict]:
    task_id = f"codex:{path.stem}:{metadata.get('episode_index', episode_index)}"
    existing = item.get("reward_contract") or metadata.get("reward_contract")
    if isinstance(existing, dict) and existing.get("kind") in {"environment", "exact_text"}:
        return task_id, existing
    return task_id, {
        "kind": "unscored_codex_replay",
        "task_id": task_id,
        "reason": "historical trace has no workspace snapshot or independent verifier",
    }


def rows(source: Path, limit: int | None = None, require_verifier: bool = False):
    count = 0
    for path in sorted(source.rglob("*.jsonl")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            messages = item.get("messages", [])
            user = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
            if not user:
                continue
            metadata = item.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            task_id, contract = _contract(item, path, metadata, count)
            if require_verifier and contract.get("kind") not in {"environment", "exact_text"}:
                raise ValueError(
                    f"{path}: row {count} has no executable reward contract; "
                    "supply a workspace snapshot plus independent verifier before RL"
                )
            yield {"data_source": "codex", "prompt": [{"role": "user", "content": user}],
                   "ability": "agentic_coding", "reward_model": {"style": "environment", "ground_truth": contract},
                   "extra_info": {"source": path.name, "task_id": task_id, "reward_contract": contract,
                                  "messages": messages, "metadata": metadata}}
            count += 1
            if limit and count >= limit:
                return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--require-verifier",
        action="store_true",
        help="fail instead of emitting historical rows without an executable task contract",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows(args.source, args.limit, args.require_verifier):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
