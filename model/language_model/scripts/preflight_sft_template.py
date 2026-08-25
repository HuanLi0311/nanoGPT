"""Preflight SFT chat-template concatenation with explicit known-difference rules."""

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


def per_turn_ids(
    dataset: MultiTurnSFTDataset, index: int
) -> tuple[list[list[int]], list, list[dict] | None, bool | None]:
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
        pieces.append(input_ids.tolist())
    return pieces, messages, tools, enable_thinking


def remove_consecutive_tool_boundaries(
    pieces: list[list[int]], messages: list[dict], tokenizer
) -> tuple[list[int], int]:
    """Remove delimiters added when consecutive tool turns are templated alone.

    Qwen3's full-conversation template groups adjacent ``tool`` messages inside
    one ``<|im_start|>user`` block. MultiTurnSFTDataset templates each turn
    separately, so it adds ``<|im_end|>\n<|im_start|>user`` between them.
    Keep this normalization tied to an actual tool-to-tool boundary.
    """
    im_end_newline = tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
    user_start = tokenizer.encode("<|im_start|>user", add_special_tokens=False)
    normalized = [list(piece) for piece in pieces]
    removed = 0
    for index in range(len(normalized) - 1):
        if messages[index].get("role") != "tool" or messages[index + 1].get("role") != "tool":
            continue
        if normalized[index][-len(im_end_newline) :] != im_end_newline:
            raise AssertionError(f"unexpected end delimiter at tool boundary {index}")
        if normalized[index + 1][: len(user_start)] != user_start:
            raise AssertionError(f"unexpected user delimiter at tool boundary {index + 1}")
        normalized[index] = normalized[index][: -len(im_end_newline)]
        normalized[index + 1] = normalized[index + 1][len(user_start) :]
        removed += 1
    return [token for piece in normalized for token in piece], removed


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
        pieces, messages, tools, enable_thinking = per_turn_ids(dataset, index)
        concat_ids = [token for piece in pieces for token in piece]
        full_kwargs = {}
        if enable_thinking is not None:
            full_kwargs["enable_thinking"] = enable_thinking
        full = apply_chat_template(
            tokenizer,
            messages=messages,
            tools=tools,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **full_kwargs,
        )
        full_ids = full["input_ids"].squeeze(0).tolist()
        if concat_ids == full_ids:
            counts["exact"] += 1
            continue
        normalized_concat, tool_boundaries = remove_consecutive_tool_boundaries(pieces, messages, tokenizer)
        normalized_full, think_blocks = remove_subsequence(full_ids, think_pattern)
        if (tool_boundaries or think_blocks) and normalized_concat == normalized_full:
            counts["expected_known_template_difference"] += 1
            if think_blocks:
                counts["expected_empty_think"] += 1
                counts[f"expected_empty_think_blocks_{think_blocks}"] += 1
            if tool_boundaries:
                counts["expected_consecutive_tool_boundaries"] += 1
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
        "policy": "allow_qwen3_empty_think_and_consecutive_tool_boundaries_v2",
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
