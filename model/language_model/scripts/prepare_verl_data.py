"""Convert filtered Codex Parquet to the verl GRPO row contract.

``ground_truth`` is a verifier contract, not a gold assistant response.
Historical rows without an independent verifier remain explicitly censored.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _contract(path: Path, metadata: dict, episode_index: int) -> tuple[str, dict]:
    task_id = f"codex:{path.stem}:{metadata.get('episode_index', episode_index)}"
    return task_id, {
        "kind": "unscored_codex_replay",
        "task_id": task_id,
        "reason": "historical trace has no workspace snapshot or independent verifier",
    }


def rows(source: Path, limit: int | None = None, require_verifier: bool = False):
    count = 0
    paths = [source] if source.suffix == ".parquet" else [source / "codex_train.parquet", source / "codex_test.parquet"]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        for batch in pq.ParquetFile(path).iter_batches(columns=["messages", "metadata"], batch_size=256):
            values = batch.to_pydict()
            for messages, metadata in zip(values["messages"], values["metadata"], strict=True):
                user = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
                if not user:
                    continue
                if not isinstance(metadata, dict):
                    metadata = {}
                task_id, contract = _contract(path, metadata, count)
                if require_verifier and contract.get("kind") not in {"environment", "exact_text"}:
                    raise ValueError(
                        f"{path}: row {count} has no executable reward contract; "
                        "supply a workspace snapshot plus independent verifier before RL"
                    )
                yield {"data_source": "codex", "prompt": [{"role": "user", "content": user}],
                       "ability": "agentic_coding", "reward_model": {"style": "environment", "ground_truth": contract},
                       "extra_info": {"source": metadata.get("source_file", path.name), "task_id": task_id, "reward_contract": contract,
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
    if args.output.suffix != ".parquet":
        raise ValueError("verl datasets must be written as .parquet")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pylist(list(rows(args.source, args.limit, args.require_verifier))), temporary, compression="zstd")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
