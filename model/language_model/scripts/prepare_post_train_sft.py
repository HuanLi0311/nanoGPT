"""Encode filtered Codex JSONL with the tokenizer used by the base checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..model.tokenizer import load_tokenizer


def encode(source: Path, output: Path, tokenizer_dir: Path, prefix: str, shard_tokens: int, limit: int | None = None, batch_size: int = 32) -> None:
    tokenizer = load_tokenizer(tokenizer_dir, prefix)
    for path in output.glob("input_*.bin"):
        path.unlink()
    for path in output.glob("labels_*.bin"):
        path.unlink()
    (output / "metadata.json").unlink(missing_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    buffers = {"train": ([], []), "validation": ([], [])}
    counts = {"train": 0, "validation": 0}
    shards = {"train": 0, "validation": 0}
    seen = 0
    pending: list[tuple[str, list[tuple[int, int]], str]] = []

    def flush(split: str, final: bool = False) -> None:
        ids, labels = buffers[split]
        while len(ids) >= shard_tokens or (final and ids):
            size = min(shard_tokens, len(ids))
            np.asarray(ids[:size], dtype=np.uint32).tofile(output / f"input_{split}_{shards[split]:05d}.bin")
            np.asarray(labels[:size], dtype=np.int32).tofile(output / f"labels_{split}_{shards[split]:05d}.bin")
            del ids[:size], labels[:size]
            counts[split] += size
            shards[split] += 1

    def process(items: list[tuple[str, list[tuple[int, int]], str]]) -> None:
        encodings = tokenizer.encode_batch([item[0] for item in items])
        for (text, ranges, split), encoded in zip(items, encodings, strict=True):
            labels = [token if any(offset[1] > start and offset[0] < end for start, end in ranges) else -100
                      for token, offset in zip(encoded.ids, encoded.offsets)]
            buffers[split][0].extend(encoded.ids)
            buffers[split][1].extend(labels)
            flush(split)

    for path in sorted(source.rglob("*.jsonl")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if limit and seen >= limit:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            messages = row.get("messages", [])
            if not messages:
                continue
            text, ranges = "", []
            for message in messages:
                role, content = message.get("role"), str(message.get("content", ""))
                text += f"<|im_start|>{role}\n"
                start = len(text)
                text += f"{content}<|im_end|>\n"
                if role == "assistant":
                    ranges.append((start, len(text)))
            split = "validation" if seen % 20 == 0 else "train"
            pending.append((text, ranges, split))
            seen += 1
            if len(pending) >= batch_size:
                process(pending)
                pending.clear()
        if limit and seen >= limit:
            break
    process(pending)
    flush("train", True)
    flush("validation", True)
    (output / "metadata.json").write_text(json.dumps({
        "input_dtype": "uint32", "label_dtype": "int32", "ignore_index": -100,
        "vocabulary_size": tokenizer.get_vocab_size(), "tokens": counts, "shards": shards,
        "examples": seen, "chat_template": "<|im_start|>{role}\\n{content}<|im_end|>\\n",
    }, indent=2), encoding="utf-8")
    print(json.dumps({"examples": seen, "vocabulary_size": tokenizer.get_vocab_size(), "tokens": counts, "shards": shards}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-prefix", default="byte_bpe")
    parser.add_argument("--shard-tokens", type=int, default=100_000_000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    encode(args.source, args.output, args.tokenizer_dir, args.tokenizer_prefix, args.shard_tokens, args.limit, args.batch_size)


if __name__ == "__main__":
    main()
