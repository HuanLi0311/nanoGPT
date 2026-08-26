"""Record Verl tool events in the same shape as the TypeScript agent loop."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

try:
    from agent.env.workspace import snapshot, state_delta
except ModuleNotFoundError:  # direct script execution
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from agent.env.workspace import snapshot, state_delta


def preload_worker() -> None:
    """Warm stdlib imports before Ray starts concurrent TransferQueue actors."""

    # Ray can start several actors at once; torch/Ray imports these modules
    # through different paths and Python 3.11 occasionally observes a
    # half-initialized package.  Keep this hook limited to existing runtime deps.
    import asyncio.base_events  # noqa: F401
    import email.errors  # noqa: F401
    import email.feedparser  # noqa: F401
    import email.parser  # noqa: F401
    import importlib._abc  # noqa: F401
    import json.decoder  # noqa: F401
    import multiprocessing.context  # noqa: F401
    import xml.etree.ElementTree  # noqa: F401
    import urllib3.exceptions  # noqa: F401
    import unittest.mock  # noqa: F401
    import unittest.result  # noqa: F401


def _extra_fields(agent_data: Any) -> dict[str, Any]:
    if agent_data is None:
        return {}
    fields = getattr(agent_data, "extra_fields", None)
    if fields is None:
        fields = {}
        setattr(agent_data, "extra_fields", fields)
    return fields


def _tool_call_id(agent_data: Any, operation: str, arguments: Any) -> tuple[str, str]:
    """Reuse Verl's generated id when visible, otherwise record a stable id."""

    encoded = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
    for call in getattr(agent_data, "tool_calls", []) or []:
        if getattr(call, "name", None) != operation:
            continue
        try:
            same_arguments = json.loads(getattr(call, "arguments", "")) == arguments
        except (TypeError, json.JSONDecodeError):
            same_arguments = getattr(call, "arguments", None) == encoded
        if same_arguments:
            call_id = getattr(call, "tool_call_id", None)
            if call_id:
                return str(call_id), "linked"
    fields = _extra_fields(agent_data)
    index = int(fields.get("agent_event_count", 0))
    return f"adapter_call_{index}", "synthetic"


def record_tool_event(
    agent_data: Any,
    *,
    operation: str,
    arguments: Any,
    result: dict[str, Any],
    state_before: dict[str, Any],
    state_after: dict[str, Any],
) -> list[dict[str, Any]]:
    """Append linked call/result events and preserve state transitions."""

    fields = _extra_fields(agent_data)
    call_id, linkage = _tool_call_id(agent_data, operation, arguments)
    timestamp = datetime.now(timezone.utc).isoformat()
    call_event = {
        "role": "assistant",
        "kind": "tool_call",
        "tool": operation,
        "tool_call_id": call_id,
        "arguments": arguments,
        "state_before": state_before,
        "timestamp": timestamp,
    }
    result_event = {
        "role": "tool",
        "kind": "tool_result",
        "tool": operation,
        "tool_call_id": call_id,
        "tool_result": result,
        "exit_code": result.get("exit_code", 0),
        "state_after": state_after,
        "state_delta": state_delta(state_before, state_after),
        "protocol_status": "valid",
        "call_result_linkage_complete": linkage == "linked",
        "trace_fidelity": linkage,
        "timestamp": timestamp,
    }
    events = fields.setdefault("agent_events", [])
    events.extend([call_event, result_event])
    fields["agent_event_count"] = int(fields.get("agent_event_count", 0)) + 1
    fields["last_state"] = state_after
    return [call_event, result_event]


def finish_episode(agent_data: Any, *, final: str = "", termination_reason: str = "final") -> dict[str, Any]:
    """Add the framework-owned termination marker without scoring the task."""

    fields = _extra_fields(agent_data)
    marker = {
        "role": "assistant",
        "kind": "message",
        "content": final,
        "termination_reason": termination_reason,
        "protocol_status": "valid",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    fields.setdefault("agent_events", []).append(marker)
    fields["termination_reason"] = termination_reason
    return marker


if __name__ == "__main__":
    from types import SimpleNamespace
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as directory:
        data = SimpleNamespace(tool_calls=[], extra_fields={})
        before = snapshot(directory)
        Path(directory, "x.txt").write_text("x", encoding="utf-8")
        after = snapshot(directory)
        record_tool_event(data, operation="apply_patch", arguments={}, result={"exit_code": 0}, state_before=before, state_after=after)
        assert len(data.extra_fields["agent_events"]) == 2
    print("loop adapter self-check passed")
