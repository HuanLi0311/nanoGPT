"""Build raw trajectory artifacts and readable SFT-template projections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    from .filter import event_message
    from .tool_message import message_content
except ImportError:  # Supports direct execution from this scripts directory.
    from filter import event_message
    from tool_message import message_content


def payload_for(event: dict) -> dict:
    payload = event.get("payload", event)
    if isinstance(event.get("message"), dict):
        payload = {**event["message"], "type": event.get("type")}
    return payload if isinstance(payload, dict) else {}


def raw_episodes(source: Path):
    """Use the same closed-call user-message boundary as the Codex filter."""
    current: list[dict] | None = None
    pending: set[str] = set()
    index = 0
    with source.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            payload = payload_for(event)
            message = event_message(payload)
            if message and message.get("role") == "user" and current and not pending:
                yield index, current
                index += 1
                current, pending = None, set()
            if current is None:
                if not message or message.get("role") != "user":
                    continue
                current = []
            current.append(event)
            if message:
                for call in message.get("tool_calls", []):
                    call_id = call.get("id") or call.get("call_id")
                    if call_id:
                        pending.add(call_id)
                if message.get("role") == "tool":
                    pending.discard(message.get("tool_call_id") or payload.get("tool_call_id") or payload.get("call_id"))
    if current:
        yield index, current


def execution_results(events: list[dict]) -> list[dict]:
    results = []
    for event in events:
        payload = payload_for(event)
        code = payload.get("exit_code")
        if isinstance(code, int) and not isinstance(code, bool):
            results.append({
                "timestamp": event.get("timestamp"),
                "type": event.get("type"),
                "payload_type": payload.get("type"),
                "call_id": payload.get("call_id"),
                "tool_use_id": payload.get("tool_use_id"),
                "exit_code": code,
                "status": payload.get("status"),
                "output": payload.get("output", payload.get("aggregated_output", payload.get("formatted_output", ""))),
            })
    return results


def bucket_for(results: list[dict]) -> tuple[str | None, str]:
    if not results:
        return None, "no_structured_exit_code"
    if all(result["exit_code"] == 0 for result in results):
        return "success", "all_exit_codes_zero"
    return "failure", "nonzero_exit_code"


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_trajectory_artifacts(raw: Path, output: Path) -> dict[str, int]:
    counts = {"episodes": 0, "success": 0, "failure": 0, "skipped_no_structured_exit_code": 0}
    for source in sorted(raw.glob("*.jsonl")):
        for index, events in raw_episodes(source):
            results = execution_results(events)
            bucket, reason = bucket_for(results)
            counts["episodes"] += 1
            if bucket is None:
                counts["skipped_no_structured_exit_code"] += 1
                continue
            artifact = {
                "source_file": source.name,
                "episode_index": index,
                "classification": {"bucket": bucket, "reason": reason, "exit_codes": results},
                "events": events,
            }
            write_json(output / bucket / f"{source.stem}-episode-{index:04d}.json", artifact)
            counts[bucket] += 1
    return counts


def sft_template(messages: list[dict]) -> str:
    return "".join(
        f"<|im_start|>{message.get('role', '')}\n{message_content(message)}<|im_end|>\n"
        for message in messages
    )


def build_templates(filtered: Path, output: Path) -> int:
    count = 0
    for source in sorted(filtered.rglob("*.jsonl")):
        rows = []
        for line in source.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(sft_template(row.get("messages", [])))
        if not rows:
            continue
        relative = source.relative_to(filtered).with_suffix(".txt")
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(rows), encoding="utf-8")
        count += len(rows)
    return count


def self_check() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        raw, filtered = root / "raw", root / "filtered"
        raw.mkdir(); filtered.mkdir()
        events = [
            {"timestamp": "t0", "type": "response_item", "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "first"}]}},
            {"timestamp": "t1", "type": "response_item", "payload": {"type": "function_call", "call_id": "c1", "name": "exec_command", "arguments": "{}"}},
            {"timestamp": "t2", "type": "response_item", "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Warning: use apply_patch"}]}},
            {"timestamp": "t3", "type": "response_item", "payload": {"type": "function_call_output", "call_id": "c1", "exit_code": 0, "output": "ok"}},
            {"timestamp": "t4", "type": "response_item", "payload": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "done"}]}},
            {"timestamp": "t5", "type": "response_item", "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "second"}]}},
            {"timestamp": "t6", "type": "response_item", "payload": {"type": "function_call", "call_id": "c2", "name": "exec_command", "arguments": "{}"}},
            {"timestamp": "t7", "type": "response_item", "payload": {"type": "function_call_output", "call_id": "c2", "exit_code": 1, "output": "bad"}},
            {"timestamp": "t8", "type": "response_item", "payload": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "failed"}]}},
            {"timestamp": "t9", "type": "response_item", "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "unknown"}]}},
        ]
        (raw / "sample.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
        row = {"messages": [{"role": "user", "content": "task"}, {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "exec_command", "arguments": {"cmd": "pwd"}}}]}]}
        (filtered / "sample.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        counts = build_trajectory_artifacts(raw, root / "artifacts")
        templates = build_templates(filtered, root / "template")
        assert counts == {"episodes": 3, "success": 1, "failure": 1, "skipped_no_structured_exit_code": 1}
        assert templates == 1
        assert (root / "artifacts/success/sample-episode-0000.json").is_file()
        assert (root / "artifacts/failure/sample-episode-0001.json").is_file()
        success = json.loads((root / "artifacts/success/sample-episode-0000.json").read_text(encoding="utf-8"))
        assert success["events"][0]["payload"]["content"][0]["text"] == "first"
        assert success["events"][2]["payload"]["content"][0]["text"] == "Warning: use apply_patch"
        assert success["events"][3]["payload"]["call_id"] == "c1"
        assert "{\"tool_calls\":[{\"id\":null,\"type\":\"function\",\"function\":{\"name\":\"exec_command\",\"arguments\":{\"cmd\":\"pwd\"}}}]}" in (root / "template/sample.txt").read_text(encoding="utf-8")


def main() -> None:
    data = Path(__file__).resolve().parents[1] / "data/post_train/data"
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=data / "raw")
    parser.add_argument("--filtered", type=Path, default=data / "filtered")
    parser.add_argument("--traj-output", type=Path, default=data / "traj_artifact")
    parser.add_argument("--template-output", type=Path, default=data / "template")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    artifacts = build_trajectory_artifacts(args.raw, args.traj_output)
    templates = build_templates(args.filtered, args.template_output)
    print(json.dumps({"traj_artifacts": artifacts, "templates": templates}, ensure_ascii=False))


if __name__ == "__main__":
    main()
