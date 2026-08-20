"""Convert filtered Codex rollouts to the verl GRPO row contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def rows(source: Path, limit: int | None = None):
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
            yield {"data_source": "codex", "prompt": [{"role": "user", "content": user}],
                   "ability": "agentic_coding", "reward_model": {"style": "rule", "ground_truth": ""},
                   "extra_info": {"source": path.name, "messages": messages, "metadata": item.get("metadata", {})}}
            count += 1
            if limit and count >= limit:
                return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows(args.source, args.limit):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
