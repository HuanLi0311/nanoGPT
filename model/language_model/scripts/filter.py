"""Split Codex sessions into lenient, task-level veRL messages."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

SECRET = re.compile(r"-----BEGIN .* PRIVATE KEY-----|AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9]{20,}")
ENV_SECRET = re.compile(r"(?i)\b(?:api[_-]?key|token|secret|password|cookie)\s*=[^\s]+")
BAD = {"turn_aborted", "turn_failed", "task_failed", "session_error", "fatal_error"}
PERMISSION = re.compile(r"(?i)permission denied|operation not permitted|access denied")
TEST_OK = re.compile(r"(?i)(?:pytest|unittest|tests?).{0,80}(?:passed|ok|success)|\b(?:passed|ok)\b")
EXIT_OK = re.compile(r"(?i)(?:exit(?:ed|\s+code)?|return code)\D{0,10}0\b")
PATCH_OK = re.compile(r"(?i)(?:patch|apply_patch).{0,80}(?:done|success|applied|succeeded)")
MAX_TOOLS, MAX_CHARS = 128, 200_000


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
    meta, current, malformed = {}, [], 0
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
            if message["role"] == "user" and current:
                yield {"meta": {**meta, "malformed_json": malformed} if malformed else meta,
                       "events": current, "source": path.name, "index": 0}
                current = []
            current.append((message, payload))
    if current:
        yield {"meta": {**meta, "malformed_json": malformed} if malformed else meta,
               "events": current, "source": path.name, "index": 0}


def make_episodes(meta: dict, events: list[tuple[dict, dict]], source: str) -> list[dict]:
    episodes, current = [], []
    for message, payload in events:
        if message["role"] == "user" and current:
            episodes.append(current)
            current = []
        current.append((message, payload))
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
    signals = set()
    for message, payload in events:
        kind = payload.get("type")
        if kind in BAD:
            flags.add(kind)
        if PERMISSION.search(text_content(payload.get("output"))):
            flags.add("permission_error")
        if kind in {"function_call", "custom_tool_call"}:
            tool_calls += 1
            if payload.get("call_id"):
                pending.add(payload["call_id"])
        elif kind in {"function_call_output", "custom_tool_call_output"}:
            pending.discard(payload.get("call_id"))
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
    if chars > MAX_CHARS:
        return None, "too_long", user
    score = min(1.0, 0.5 + 0.15 * bool(signals) + 0.1 * ("tests_passed" in signals) + 0.1 * ("exit_code_0" in signals) + 0.05 * ("patch_success" in signals))
    record = {"messages": messages, "data_source": "codex", "metadata": {
        **item["meta"], "source_file": item["source"], "episode_index": item["index"],
        "quality_score": round(score, 3), "signals": sorted(signals), "tool_calls": tool_calls, "chars": chars,
    }}
    return record, None, user


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path(__file__).parents[1] / "data/raw/rl/raw")
    parser.add_argument("--output", type=Path, default=Path(__file__).parents[1] / "data/raw/rl/filtered")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    kept, dropped, seen, existing, reasons = 0, 0, set(), set(), {}
    output_index = len(list(args.output.glob("*.jsonl")))
    for old in args.output.glob("*.jsonl"):
        try:
            record = json.loads(old.read_text(encoding="utf-8"))
            metadata = record.get("metadata", {})
            existing.add((metadata.get("source_file"), metadata.get("episode_index")))
            task = next(m.get("content", "") for m in record.get("messages", []) if m.get("role") == "user")
            seen.add(hashlib.sha256(re.sub(r"\s+", " ", task.lower()).strip().encode()).hexdigest())
        except (OSError, json.JSONDecodeError, StopIteration):
            continue
    for source in sorted(args.input.glob("*.jsonl")):
        for episode_index, item in enumerate(iter_episodes(source)):
            item["index"] = episode_index
            record, reason, task = inspect(item)
            identity = (source.name, episode_index)
            if identity in existing:
                continue
            if record is None:
                dropped += 1; reasons[reason] = reasons.get(reason, 0) + 1; continue
            key = hashlib.sha256(re.sub(r"\s+", " ", task.lower()).strip().encode()).hexdigest()
            if key in seen:
                dropped += 1; reasons["duplicate_task"] = reasons.get("duplicate_task", 0) + 1; continue
            seen.add(key)
            (args.output / f"{source.stem}-episode-{item['index']:04d}-{output_index:06d}.jsonl").write_text(
                json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            output_index += 1
            kept += 1
    total = kept + dropped
    stats = {"mode": "append_lenient", "input_sessions": len(list(args.input.glob("*.jsonl"))), "new_episodes": total, "new_kept": kept, "new_dropped": dropped, "keep_ratio": kept / total if total else 0, "drop_reasons": reasons, "limits": {"max_tool_calls": MAX_TOOLS, "max_chars": MAX_CHARS}}
    (args.output / "filter_stats_append.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"sessions={stats['input_sessions']} new_episodes={total} new_kept={kept} new_dropped={dropped} keep_ratio={stats['keep_ratio']:.4f}")
    print(json.dumps(reasons, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
