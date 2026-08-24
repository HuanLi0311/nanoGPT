"""Regenerate trajectory artifacts and readable ChatML templates from raw logs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

try:
    from .filter import event_message, inspect, iter_episodes
    from .tool_message import message_content
except ImportError:  # Supports direct execution from this scripts directory.
    from filter import event_message, inspect, iter_episodes
    from tool_message import message_content


def payload_for(event: dict) -> dict:
    payload = event.get("payload", event)
    if isinstance(event.get("message"), dict):
        payload = {**event["message"], "type": event.get("type")}
    return payload if isinstance(payload, dict) else {}


def raw_episodes(source: Path):
    current: list[dict] | None = None
    index = 0
    with source.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            message = event_message(payload_for(event))
            if message and message.get("role") == "user":
                if current:
                    yield index, current
                    index += 1
                current = [event]
            elif current is not None:
                current.append(event)
    if current:
        yield index, current


def results(events: list[dict]) -> list[dict]:
    output = []
    for event in events:
        payload = payload_for(event)
        code = payload.get("exit_code")
        if isinstance(code, int) and not isinstance(code, bool):
            output.append({
                "timestamp": event.get("timestamp"), "type": event.get("type"),
                "payload_type": payload.get("type"), "call_id": payload.get("call_id"),
                "tool_use_id": payload.get("tool_use_id"), "exit_code": code,
                "status": payload.get("status"),
                "output": payload.get("output", payload.get("aggregated_output", payload.get("formatted_output", ""))),
            })
    return output


def reset(path: Path, raw: Path) -> None:
    path = path.resolve()
    if path == raw or path in raw.parents:
        raise ValueError(f"refusing to replace unsafe output path: {path}")
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)


def write(path: Path, value: str | dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def template(messages: list[dict]) -> str:
    return "".join(f"<|im_start|>{message.get('role', '')}\n{message_content(message)}<|im_end|>\n" for message in messages)


def build(raw: Path, artifacts: Path, templates: Path) -> dict:
    raw = raw.resolve()
    if not raw.is_dir():
        raise ValueError(f"raw input directory does not exist: {raw}")
    reset(artifacts, raw)
    reset(templates, raw)
    counts = {"episodes": 0, "success": 0, "failure": 0, "skipped_no_structured_exit_code": 0,
              "templates": 0, "template_dropped": 0}

    for source in sorted(raw.glob("*.jsonl")):
        for index, events in raw_episodes(source):
            counts["episodes"] += 1
            exits = results(events)
            if not exits:
                counts["skipped_no_structured_exit_code"] += 1
                continue
            bucket = "success" if all(item["exit_code"] == 0 for item in exits) else "failure"
            counts[bucket] += 1
            write(artifacts / bucket / f"{source.stem}-episode-{index:04d}.json", {
                "source_file": source.name, "episode_index": index,
                "classification": {"bucket": bucket, "reason": "all_exit_codes_zero" if bucket == "success" else "nonzero_exit_code", "exit_codes": exits},
                "events": events,
            })

    seen: set[str] = set()
    for source in sorted(raw.glob("*.jsonl")):
        for index, item in enumerate(iter_episodes(source)):
            item["index"] = index
            record, _, task = inspect(item)
            if record is None:
                counts["template_dropped"] += 1
                continue
            key = hashlib.sha256(re.sub(r"\s+", " ", task.lower()).strip().encode()).hexdigest()
            if key in seen:
                counts["template_dropped"] += 1
                continue
            seen.add(key)
            write(templates / f"{source.stem}-episode-{index:04d}.txt", template(record["messages"]))
            counts["templates"] += 1
    return counts


def main() -> None:
    data = Path(__file__).resolve().parents[1] / "data/post_train/data"
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=data / "raw")
    parser.add_argument("--traj-output", type=Path, default=data / "traj_artifact")
    parser.add_argument("--template-output", type=Path, default=data / "template")
    args = parser.parse_args()
    print(json.dumps(build(args.raw, args.traj_output, args.template_output), ensure_ascii=False))


if __name__ == "__main__":
    main()
