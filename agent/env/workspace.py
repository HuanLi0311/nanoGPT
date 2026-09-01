"""Compatibility imports; use :mod:`agent.workspace` in new code."""

if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.workspace.snapshot import snapshot, state_delta

__all__ = ["snapshot", "state_delta"]


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as directory:
        file = Path(directory) / "answer.txt"
        before = snapshot(directory)
        file.write_text("ok\n", encoding="utf-8")
        after = snapshot(directory)
        assert before["state_hash"] != after["state_hash"]
        assert state_delta(before, after)["added"] == ["answer.txt"]
    print("workspace compatibility self-check passed")
