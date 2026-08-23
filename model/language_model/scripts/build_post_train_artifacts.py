"""Build raw trajectory artifacts and readable SFT-template projections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    from .filter import event_message
except ImportError:  # Supports direct execution from this scripts directory.
    from filter import event_message


def payload_for(event: dict) -> dict:
    payload = event.get("payload", event)
    if isinstance(event.get("message"), dict):
        payload = {**event["message"], "type": event.get("type")}
    return payload if isinstance(payload, dict) else {}


def raw_episodes(source: Path):
    """Use the same user-message boundary as the existing Codex filter."""
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


def sft_message_content(message: dict) -> str:
    """Mirror prepare_post_train_sft.message_content without loading the model stack."""
    content = str(message.get("content") or "")
    calls = message.get("tool_calls") or []
    if not calls:
        return content
    functions = []
    for call in calls:
        function = call.get("function", call)
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                pass
        functions.append({"name": function.get("name"), "arguments": arguments})
    payload = {"tool_call": functions[0]} if len(functions) == 1 else {"tool_calls": functions}
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"{content}\n{encoded}" if content else encoded


def sft_template(messages: list[dict]) -> str:
    return "".join(
        f"<|im_start|>{message.get('role', '')}\n{sft_message_content(message)}<|im_end|>\n"
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
            {"timestamp": "t1", "type": "event_msg", "payload": {"type": "exec_command_end", "call_id": "c1", "exit_code": 0, "output": "ok"}},
            {"timestamp": "t2", "type": "response_item", "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "second"}]}},
            {"timestamp": "t3", "type": "event_msg", "payload": {"type": "exec_command_end", "call_id": "c2", "exit_code": 1, "output": "bad"}},
            {"timestamp": "t4", "type": "response_item", "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "unknown"}]}},
        ]
        (raw / "sample.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
        row = {"messages": [{"role": "user", "content": "task"}, {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "exec_command", "arguments": {"cmd": "pwd"}}}]}]}
        (filtered / "sample.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        counts = build_trajectory_artifacts(raw, root / "artifacts")
        templates = build_templates(filtered, root / "template")
        assert counts["success"] == counts["failure"] == templates == 1
        assert counts["skipped_no_structured_exit_code"] == 1
        assert (root / "artifacts/success/sample-episode-0000.json").is_file()
        assert not (root / "artifacts/failure/sample-episode-0002.json").exists()
        assert "{\"tool_call\":{\"name\":\"exec_command\",\"arguments\":{\"cmd\":\"pwd\"}}}" in (root / "template/sample.txt").read_text(encoding="utf-8")


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
