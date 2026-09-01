"""Dependency-free file-state snapshots for agent episodes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def snapshot(root: str | Path) -> dict[str, Any]:
    """Return a stable JSON-serializable snapshot of files below ``root``."""

    base = Path(root).resolve()
    files: list[dict[str, Any]] = []
    for path in sorted(base.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        relative = path.relative_to(base).as_posix()
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
        files.append({"path": relative, "size": size, "sha256": digest.hexdigest()})
    payload = json.dumps(files, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return {"state_hash": hashlib.sha256(payload).hexdigest(), "files": files}


def state_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, list[str]]:
    """Describe only added, removed, and changed relative file paths."""

    old = {item["path"]: item for item in before.get("files", [])}
    new = {item["path"]: item for item in after.get("files", [])}
    return {
        "added": sorted(set(new) - set(old)),
        "removed": sorted(set(old) - set(new)),
        "changed": sorted(path for path in set(old) & set(new) if old[path] != new[path]),
    }


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        file = Path(directory) / "answer.txt"
        before = snapshot(directory)
        file.write_text("ok\n", encoding="utf-8")
        after = snapshot(directory)
        assert before["state_hash"] != after["state_hash"]
        assert state_delta(before, after)["added"] == ["answer.txt"]
    print("workspace snapshot self-check passed")
