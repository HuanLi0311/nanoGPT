"""Small regression checks for Codex trajectory filtering."""

from filter import inspect


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
    print("filter checks passed")


if __name__ == "__main__":
    main()
