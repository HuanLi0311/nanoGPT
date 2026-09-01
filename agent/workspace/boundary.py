"""File-level workspace boundary.

This module constrains paths supplied to tools.  It is not an OS sandbox:
commands still run as host processes with the workspace as their cwd.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _inside(root: Path, path: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _reject_symlink_components(root: Path, path: Path) -> None:
    try:
        parts = path.relative_to(root).parts
    except ValueError as error:
        raise ValueError(f"path escapes workspace: {path}") from error
    current = root
    for part in parts:
        current /= part
        try:
            if current.is_symlink():
                raise ValueError(f"symlink path is not allowed: {path}")
        except OSError as error:
            raise ValueError(f"unable to inspect workspace path: {path}") from error
        if not current.exists():
            break


def workspace_path(root: str | Path, value: Any = ".", *, must_exist: bool = False) -> Path:
    """Resolve a tool path under ``root`` or the virtual ``/workspace`` root."""

    base = Path(root).resolve(strict=True)
    raw = str(value or ".")
    path = Path(raw)
    if path.is_absolute() and (path == Path("/workspace") or Path("/workspace") in path.parents):
        candidate = base / path.relative_to("/workspace")
    else:
        candidate = path if path.is_absolute() else base / path
    candidate = candidate.resolve(strict=False)
    if not _inside(base, candidate):
        raise ValueError(f"path escapes workspace: {value}")
    _reject_symlink_components(base, candidate)
    if must_exist and not candidate.exists():
        raise FileNotFoundError(candidate)
    return candidate
