"""Lossless text projection for Codex-style tool messages."""

from __future__ import annotations

import json


def _arguments(value: object) -> object:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def message_content(message: dict) -> str:
    """Keep call/result IDs when rendering structured messages into ChatML text."""
    content = str(message.get("content") or "")
    calls = message.get("tool_calls") or []
    if calls:
        rendered = []
        for call in calls:
            function = call.get("function", call)
            rendered.append({
                "id": call.get("id", call.get("call_id")),
                "type": call.get("type", "function"),
                "function": {
                    "name": function.get("name"),
                    "arguments": _arguments(function.get("arguments", {})),
                },
            })
        encoded = json.dumps({"tool_calls": rendered}, ensure_ascii=False, separators=(",", ":"))
        return f"{content}\n{encoded}" if content else encoded
    if message.get("role") == "tool" and message.get("tool_call_id") is not None:
        return json.dumps(
            {"tool_call_id": message["tool_call_id"], "content": content},
            ensure_ascii=False,
            separators=(",", ":"),
        )
    return content


def is_supervised_message(message: dict, tool_calls_only: bool) -> bool:
    return message.get("role") == "assistant" and (not tool_calls_only or bool(message.get("tool_calls")))
