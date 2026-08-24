"""Small regression checks for SFT message rendering and label selection."""

import json

from model.language_model.scripts.tool_message import is_supervised_message, message_content


def main() -> None:
    prose = {"role": "assistant", "content": "done"}
    call = {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "exec_command"}}]}
    user = {"role": "user", "content": "do it"}
    assert is_supervised_message(prose, False)
    assert not is_supervised_message(prose, True)
    assert is_supervised_message(call, True)
    assert not is_supervised_message(user, False)
    call_text = message_content({"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "exec_command", "arguments": "{\"cmd\":\"true\"}"}}]})
    assert json.loads(call_text)["tool_calls"][0]["id"] == "call_1"
    result_text = message_content({"role": "tool", "tool_call_id": "call_1", "content": "ok"})
    assert json.loads(result_text) == {"tool_call_id": "call_1", "content": "ok"}
    print("SFT label-selection checks passed")


if __name__ == "__main__":
    main()
