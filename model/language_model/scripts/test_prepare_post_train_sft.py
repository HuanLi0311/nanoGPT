"""Small regression checks for SFT label selection."""

from model.language_model.scripts.prepare_post_train_sft import is_supervised_message


def main() -> None:
    prose = {"role": "assistant", "content": "done"}
    call = {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "exec_command"}}]}
    user = {"role": "user", "content": "do it"}
    assert is_supervised_message(prose, False)
    assert not is_supervised_message(prose, True)
    assert is_supervised_message(call, True)
    assert not is_supervised_message(user, False)
    print("SFT label-selection checks passed")


if __name__ == "__main__":
    main()
