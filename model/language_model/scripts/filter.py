"""Split Codex sessions into strict task-level SFT Parquet rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as pq

SECRET = re.compile(r"-----BEGIN .* PRIVATE KEY-----|AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9]{20,}")
ENV_SECRET = re.compile(r"(?i)\b(?:api[_-]?key|token|secret|password|cookie)\s*=[^\s]+")
BAD = {"turn_aborted", "turn_failed", "task_failed", "session_error", "fatal_error"}
PERMISSION = re.compile(r"(?i)permission denied|operation not permitted|access denied")
TEST_OK = re.compile(r"(?i)(?:pytest|unittest|tests?).{0,80}(?:passed|ok|success)|\b(?:passed|ok)\b")
EXIT_OK = re.compile(r"(?i)(?:exit(?:ed|\s+code)?|return code)\D{0,10}0\b")
PATCH_OK = re.compile(r"(?i)(?:patch|apply_patch).{0,80}(?:done|success|applied|succeeded)")
MAX_TOOLS, MAX_CHARS = 128, 200_000

MESSAGE_SCHEMA = pa.list_(pa.struct([
    pa.field("content", pa.string()),
    pa.field("role", pa.string()),
    pa.field("tool_call_id", pa.string()),
    pa.field("tool_calls", pa.list_(pa.struct([
        pa.field("function", pa.struct([pa.field("arguments", pa.string()), pa.field("name", pa.string())])),
        pa.field("id", pa.string()), pa.field("type", pa.string()),
    ]))),
]))
SFT_SCHEMA = pa.schema([
    pa.field("messages", MESSAGE_SCHEMA), pa.field("data_source", pa.string()),
    pa.field("trajectory_id", pa.string()), pa.field("split", pa.string()),
    pa.field("metadata", pa.struct([
        pa.field("chars", pa.int64()), pa.field("cli_version", pa.string()), pa.field("cwd", pa.string()),
        pa.field("episode_index", pa.int64()), pa.field("id", pa.string()), pa.field("malformed_json", pa.int64()),
        pa.field("model_provider", pa.string()), pa.field("quality_score", pa.float64()),
        pa.field("signals", pa.list_(pa.string())), pa.field("source_file", pa.string()),
        pa.field("timestamp", pa.string()), pa.field("tool_calls", pa.int64()),
    ])),
])


def text_content(value) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    return "".join(x.get("text", "") for x in value if isinstance(x, dict) and x.get("type") in {"input_text", "output_text", "text"})


def event_message(payload: dict) -> dict | None:
    kind = payload.get("type")
    if payload.get("role") in {"user", "assistant"} and "content" in payload:
        blocks = payload.get("content") if isinstance(payload.get("content"), list) else []
        text = text_content(payload.get("content"))
        calls = [{"id": b.get("id"), "type": "function", "function": {"name": b.get("name"), "arguments": b.get("input", {})}}
                 for b in blocks if isinstance(b, dict) and b.get("type") in {"tool_use", "function_call"}]
        if text.strip() or calls:
            message = {"role": payload["role"], "content": text}
            if calls:
                message["tool_calls"] = calls
            return message
    if payload.get("role") == "tool" or kind in {"tool_result", "function_call_output", "custom_tool_call_output"}:
        content = payload.get("content", payload.get("output", ""))
        return {"role": "tool", "content": text_content(content), "tool_call_id": payload.get("tool_use_id", payload.get("call_id"))}
    if kind in {"function_call", "custom_tool_call"}:
        return {"role": "assistant", "content": "", "tool_calls": [{
            "id": payload.get("call_id"), "type": "function", "function": {
                "name": payload.get("name"), "arguments": payload.get("arguments", payload.get("input", ""))
            }
        }]}
    if kind in {"function_call_output", "custom_tool_call_output"}:
        return {"role": "tool", "content": text_content(payload.get("output")), "tool_call_id": payload.get("call_id")}
    return None


def load(path: Path) -> tuple[dict, list[tuple[dict, dict]]]:
    meta, events, malformed = {}, [], 0
    for line in path.open(encoding="utf-8"):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        payload = event.get("payload", event)
        if isinstance(event.get("message"), dict):
            payload = {**event["message"], "type": event.get("type")}
        if not isinstance(payload, dict):
            continue
        if event.get("type") == "session_meta":
            meta = {k: payload.get(k) for k in ("id", "timestamp", "cwd", "model_provider", "cli_version")}
        message = event_message(payload)
        if message:
            events.append((message, payload))
        if malformed:
            meta["malformed_json"] = malformed
    return meta, events


def iter_episodes(path: Path):
    """Yield one task at a time so large sessions never accumulate in memory."""
    meta, current, pending, malformed = {}, [], set(), 0
    for line in path.open(encoding="utf-8"):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        payload = event.get("payload", event)
        if isinstance(event.get("message"), dict):
            payload = {**event["message"], "type": event.get("type")}
        if not isinstance(payload, dict):
            continue
        if event.get("type") == "session_meta":
            meta = {k: payload.get(k) for k in ("id", "timestamp", "cwd", "model_provider", "cli_version")}
        message = event_message(payload)
        if message:
            if message["role"] == "user" and current and not pending:
                yield {"meta": {**meta, "malformed_json": malformed} if malformed else meta,
                       "events": current, "source": path.name, "index": 0}
                current, pending = [], set()
            current.append((message, payload))
            for call in message.get("tool_calls", []):
                call_id = call.get("id") or call.get("call_id")
                if call_id:
                    pending.add(call_id)
            if message["role"] == "tool":
                pending.discard(message.get("tool_call_id") or payload.get("tool_call_id") or payload.get("call_id"))
    if current:
        yield {"meta": {**meta, "malformed_json": malformed} if malformed else meta,
               "events": current, "source": path.name, "index": 0}


def make_episodes(meta: dict, events: list[tuple[dict, dict]], source: str) -> list[dict]:
    episodes, current, pending = [], [], set()
    for message, payload in events:
        if message["role"] == "user" and current and not pending:
            episodes.append(current)
            current, pending = [], set()
        current.append((message, payload))
        for call in message.get("tool_calls", []):
            call_id = call.get("id") or call.get("call_id")
            if call_id:
                pending.add(call_id)
        if message["role"] == "tool":
            pending.discard(message.get("tool_call_id") or payload.get("tool_call_id") or payload.get("call_id"))
    if current:
        episodes.append(current)
    return [{"meta": meta, "events": episode, "source": source, "index": i} for i, episode in enumerate(episodes)]


def inspect(item: dict) -> tuple[dict | None, str | None, str]:
    events = item["events"]
    messages = [m for m, _ in events]
    user = next((m["content"] for m in messages if m["role"] == "user"), "").strip()
    if not user or user.startswith("<environment_context>"):
        return None, "no_clear_task", user
    if not any(m["role"] == "assistant" for m in messages):
        return None, "no_task_result", user
    text = "\n".join(m.get("content", "") for m in messages)
    if SECRET.search(text) or ENV_SECRET.search(text):
        return None, "possible_secret", user
    flags = set()
    tool_calls = 0
    pending = set()
    orphan_results = 0
    duplicate_ids = set()
    missing_ids = 0
    call_ids, result_ids = set(), set()
    signals = set()
    for message, payload in events:
        kind = payload.get("type")
        if kind in BAD:
            flags.add(kind)
        if PERMISSION.search(text_content(payload.get("output"))):
            flags.add("permission_error")
        for call in message.get("tool_calls", []):
            tool_calls += 1
            call_id = call.get("id") or call.get("call_id")
            if not call_id:
                missing_ids += 1
            elif call_id in call_ids:
                duplicate_ids.add(call_id)
            else:
                call_ids.add(call_id)
                pending.add(call_id)
        if message.get("role") == "tool" or kind in {"function_call_output", "custom_tool_call_output", "tool_result"}:
            call_id = message.get("tool_call_id") or payload.get("tool_call_id") or payload.get("call_id")
            if not call_id:
                missing_ids += 1
            elif call_id in result_ids:
                duplicate_ids.add(call_id)
            elif call_id in pending:
                result_ids.add(call_id)
                pending.discard(call_id)
            else:
                result_ids.add(call_id)
                orphan_results += 1
        output = text_content(payload.get("output"))
        raw = json.dumps(payload, ensure_ascii=False)
        if payload.get("exit_code") == 0 or EXIT_OK.search(output + raw):
            signals.add("exit_code_0")
        if TEST_OK.search(output + raw):
            signals.add("tests_passed")
        if PATCH_OK.search(output + raw):
            signals.add("patch_success")
    chars = sum(len(m.get("content", "")) for m in messages)
    if tool_calls > MAX_TOOLS:
        return None, "too_many_tools", user
    if flags:
        return None, "invalid_event", user
    if missing_ids:
        return None, "missing_tool_call_id", user
    if duplicate_ids:
        return None, "duplicate_tool_call_id", user
    if pending:
        return None, "unresolved_tool_calls", user
    if orphan_results:
        return None, "orphan_tool_results", user
    if chars > MAX_CHARS:
        return None, "too_long", user
    score = min(1.0, 0.5 + 0.15 * bool(signals) + 0.1 * ("tests_passed" in signals) + 0.1 * ("exit_code_0" in signals) + 0.05 * ("patch_success" in signals))
    record = {"messages": messages, "data_source": "codex", "metadata": {
        **item["meta"], "source_file": item["source"], "episode_index": item["index"],
        "quality_score": round(score, 3), "signals": sorted(signals), "tool_calls": tool_calls, "chars": chars,
    }}
    return record, None, user


def text(value) -> str | None:
    return value if isinstance(value, str) else None if value is None else str(value)


def row(record: dict, split: str) -> dict:
    metadata = record["metadata"]
    messages = []
    for message in record["messages"]:
        calls = []
        for call in message.get("tool_calls", []):
            function = call.get("function", call)
            arguments = function.get("arguments", {})
            calls.append({"function": {"name": text(function.get("name")), "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))},
                          "id": text(call.get("id", call.get("call_id"))), "type": text(call.get("type", "function"))})
        messages.append({"content": text(message.get("content")) or "", "role": text(message.get("role")),
                         "tool_call_id": text(message.get("tool_call_id")), "tool_calls": calls or None})
    return {"messages": messages, "data_source": record["data_source"], "trajectory_id": text(metadata.get("id")), "split": split,
            "metadata": {"chars": metadata.get("chars"), "cli_version": text(metadata.get("cli_version")), "cwd": text(metadata.get("cwd")),
                         "episode_index": metadata.get("episode_index"), "id": text(metadata.get("id")),
                         "malformed_json": metadata.get("malformed_json"), "model_provider": text(metadata.get("model_provider")),
                         "quality_score": metadata.get("quality_score"), "signals": metadata.get("signals", []),
                         "source_file": text(metadata.get("source_file")), "timestamp": text(metadata.get("timestamp")),
                         "tool_calls": metadata.get("tool_calls")}}


def split_for(record: dict) -> str:
    metadata = record["metadata"]
    identity = f"{metadata['source_file']}:{metadata['episode_index']}".encode()
    return "test" if int.from_bytes(hashlib.sha256(identity).digest()[:8], "big") % 10 == 0 else "train"


def write_parquet(input_dir: Path, output_dir: Path, batch_size: int = 128) -> dict:
    """Filter raw sessions once and atomically replace the two Codex SFT shards."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {"input_sessions": 0, "episodes": 0, "kept": 0, "dropped": 0, "drop_reasons": {}, "splits": {"train": 0, "test": 0}}
    seen, buffers = set(), {"train": [], "test": []}
    with TemporaryDirectory(prefix=".codex-filter-", dir=output_dir) as temporary:
        temporary = Path(temporary)
        paths = {split: temporary / f"codex_{split}.parquet" for split in buffers}
        writers = {split: pq.ParquetWriter(path, SFT_SCHEMA, compression="zstd") for split, path in paths.items()}

        def flush(split: str) -> None:
            if buffers[split]:
                writers[split].write_table(pa.Table.from_pylist(buffers[split], schema=SFT_SCHEMA))
                buffers[split].clear()

        try:
            for source in sorted(input_dir.glob("*.jsonl")):
                stats["input_sessions"] += 1
                for episode_index, item in enumerate(iter_episodes(source)):
                    stats["episodes"] += 1
                    item["index"] = episode_index
                    record, reason, task = inspect(item)
                    if record is None:
                        stats["dropped"] += 1; stats["drop_reasons"][reason] = stats["drop_reasons"].get(reason, 0) + 1; continue
                    key = hashlib.sha256(re.sub(r"\s+", " ", task.lower()).strip().encode()).hexdigest()
                    if key in seen:
                        stats["dropped"] += 1; stats["drop_reasons"]["duplicate_task"] = stats["drop_reasons"].get("duplicate_task", 0) + 1; continue
                    seen.add(key)
                    split = split_for(record)
                    buffers[split].append(row(record, split)); stats["kept"] += 1; stats["splits"][split] += 1
                    if len(buffers[split]) >= batch_size:
                        flush(split)
            for split in buffers:
                flush(split); writers[split].close()
            assert stats["kept"] == sum(stats["splits"].values())
            for split, path in paths.items():
                assert pq.ParquetFile(path).metadata.num_rows == stats["splits"][split]
                path.replace(output_dir / path.name)
        finally:
            for writer in writers.values():
                writer.close()
    stats["keep_ratio"] = stats["kept"] / stats["episodes"] if stats["episodes"] else 0
    temporary = output_dir / ".codex_filter_stats.json.tmp"
    temporary.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_dir / "codex_filter_stats.json")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    data = Path(__file__).resolve().parents[1] / "data/post_train/data"
    parser.add_argument("--input", type=Path, default=data / "raw")
    parser.add_argument("--output", type=Path, default=data / "rendered/sft")
    args = parser.parse_args()
    stats = write_parquet(args.input, args.output)
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
