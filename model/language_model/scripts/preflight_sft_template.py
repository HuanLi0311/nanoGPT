"""Preflight SFT chat-template concatenation, allowing Qwen3's empty think block."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from verl.utils import hf_tokenizer
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.tokenizer.chat_template import apply_chat_template


def input_paths(root: Path) -> list[Path]:
    paths = {*root.glob("train_sft-*.parquet"), *root.glob("test_sft-*.parquet")}
    paths.update({root / "codex_train.parquet", root / "codex_test.parquet"})
    return sorted(path for path in paths if path.is_file())


def sample_indices(size: int, limit: int | None) -> list[int]:
    if limit is None or limit >= size:
        return list(range(size))
    if limit <= 1:
        return [0]
    values = {0, size - 1}
    values.update(round(index * (size - 1) / (limit - 1)) for index in range(limit))
    return sorted(values)


def remove_subsequence(values: list[int], pattern: list[int]) -> tuple[list[int], int]:
    if not pattern:
        return values, 0
    kept: list[int] = []
    removed = 0
    index = 0
    while index < len(values):
        if values[index : index + len(pattern)] == pattern:
            removed += 1
            index += len(pattern)
        else:
            kept.append(values[index])
            index += 1
    return kept, removed


def per_turn_ids(dataset: MultiTurnSFTDataset, index: int) -> tuple[list[int], list]:
    row = dataset.dataframe.iloc[index].to_dict()
    messages = dataset._build_messages(row)
    tools = dataset.tools[index] if dataset.tools is not None else None
    enable_thinking = (
        dataset.enable_thinking[index]
        if dataset.enable_thinking is not None
        else dataset.enable_thinking_default
    )
    if enable_thinking is not None:
        enable_thinking = bool(enable_thinking)

    pieces = []
    for message_index, message in enumerate(messages):
        input_ids, loss_mask, attention_mask, _ = dataset._process_single_message(
            index=message_index,
            message=message,
            full_message=messages,
            tools=tools if message_index == 0 else None,
            enable_thinking=enable_thinking,
        )
        if not (len(input_ids) == len(loss_mask) == len(attention_mask)):
            raise AssertionError(f"mask/input length mismatch at row {index}")
        pieces.append(input_ids)
    return torch.cat(pieces).tolist(), messages


def check_file(path: Path, tokenizer, samples_per_file: int | None, think_pattern: list[int]) -> dict:
    config = {
        "messages_key": "messages",
        "pad_mode": "no_padding",
        "max_length": 6144,
        "truncation": "left",
        "ignore_input_ids_mismatch": True,
        # Match sft_trainer_engine.yaml: YAML's `none` is a string here.
        "enable_thinking_default": "none",
        "validate_input_ids": False,
    }
    dataset = MultiTurnSFTDataset([str(path)], tokenizer=tokenizer, processor=None, config=config)
    indices = sample_indices(len(dataset), samples_per_file)
    counts: Counter = Counter()
    unexpected = []
    for index in indices:
        concat_ids, messages = per_turn_ids(dataset, index)
        full = apply_chat_template(
            tokenizer,
            messages=messages,
            tools=None,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        full_ids = full["input_ids"].squeeze(0).tolist()
        if concat_ids == full_ids:
            counts["exact"] += 1
            continue
        normalized, removed = remove_subsequence(full_ids, think_pattern)
        if removed and normalized == concat_ids:
            counts["expected_empty_think"] += 1
            counts[f"expected_empty_think_blocks_{removed}"] += 1
            continue
        counts["unexpected_mismatch"] += 1
        if len(unexpected) < 3:
            unexpected.append({
                "row": index,
                "messages": len(messages),
                "concat_tokens": len(concat_ids),
                "full_tokens": len(full_ids),
                "first_diff": next(
                    (position for position, (left, right) in enumerate(zip(concat_ids, full_ids)) if left != right),
                    min(len(concat_ids), len(full_ids)),
                ),
            })
    return {
        "rows": len(dataset),
        "checked": len(indices),
        "counts": dict(counts),
        "unexpected": unexpected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[3]
    parser.add_argument("--data", type=Path, default=root / "model/language_model/data/post_train/data/rendered/sft")
    parser.add_argument("--model", type=Path, default=root / "model/language_model/checkpoints/qwen/Qwen3-8B-Base")
    parser.add_argument("--samples-per-file", type=int, default=8)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    tokenizer = hf_tokenizer(str(args.model), local_files_only=True)
    think_pattern = tokenizer.encode("<think>\n\n</think>\n\n", add_special_tokens=False)
    limit = None if args.full else args.samples_per_file
    report = {
        "policy": "allow_only_qwen3_empty_think_insertion_v1",
        "data": str(args.data),
        "model": str(args.model),
        "samples_per_file": limit,
        "think_pattern": think_pattern,
        "files": {},
        "totals": Counter(),
    }
    for path in input_paths(args.data):
        result = check_file(path, tokenizer, limit, think_pattern)
        report["files"][path.name] = result
        report["totals"].update(result["counts"])
        print(json.dumps({"file": path.name, **result["counts"]}, ensure_ascii=False), flush=True)
    report["totals"] = dict(report["totals"])
    report["status"] = "pass" if report["totals"].get("unexpected_mismatch", 0) == 0 else "fail"
    output = args.output or args.data / "sft_template_preflight_20260825.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "totals": report["totals"], "output": str(output)}, ensure_ascii=False, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
