"""Small regression checks for Codex trajectory filtering."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from filter import inspect, iter_episodes


def item(events):
    return {"meta": {}, "source": "test", "index": 0, "events": events}


def main() -> None:
    valid = item([
        ({"role": "user", "content": "do it"}, {"type": "message"}),
        ({"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]}, {"type": "message"}),
        ({"role": "tool", "content": "ok", "tool_call_id": "c1"}, {"type": "tool_result", "call_id": "c1"}),
        ({"role": "assistant", "content": "done"}, {"type": "message"}),
    ])
    record, reason, _ = inspect(valid)
    assert record and reason is None

    unresolved = item([
        ({"role": "user", "content": "do it"}, {"type": "message"}),
        ({"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]}, {"type": "message"}),
        ({"role": "assistant", "content": "done"}, {"type": "message"}),
    ])
    assert inspect(unresolved)[1] == "unresolved_tool_calls"

    orphan = item([
        ({"role": "user", "content": "do it"}, {"type": "message"}),
        ({"role": "tool", "content": "orphan", "tool_call_id": "c0"}, {"type": "tool_result", "call_id": "c0"}),
        ({"role": "assistant", "content": "done"}, {"type": "message"}),
    ])
    assert inspect(orphan)[1] == "orphan_tool_results"

    duplicate = item([
        ({"role": "user", "content": "do it"}, {"type": "message"}),
        ({"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}, {"id": "c1"}]}, {"type": "message"}),
    ])
    assert inspect(duplicate)[1] == "duplicate_tool_call_id"

    with TemporaryDirectory() as temporary:
        path = Path(temporary) / "cross_turn.jsonl"
        events = [
            {"type": "response_item", "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "task"}]}},
            {"type": "response_item", "payload": {"type": "function_call", "call_id": "c1", "name": "exec_command", "arguments": "{}"}},
            {"type": "response_item", "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Warning: use apply_patch"}]}},
            {"type": "response_item", "payload": {"type": "function_call_output", "call_id": "c1", "output": "ok"}},
            {"type": "response_item", "payload": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "done"}]}},
            {"type": "response_item", "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "next"}]}},
            {"type": "response_item", "payload": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "next done"}]}},
        ]
        path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
        episodes = list(iter_episodes(path))
        assert len(episodes) == 2
        assert inspect(episodes[0])[1] is None
    print("filter checks passed")


if __name__ == "__main__":
    main()
