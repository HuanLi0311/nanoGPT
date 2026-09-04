"""Minimal Bubblewrap command builder for generated-task execution."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .boundary import workspace_path


def bubblewrap_command(root: str | Path, command: str, workdir: Any = ".") -> list[str]:
    executable = shutil.which("bwrap")
    if executable is None:
        raise RuntimeError("sandbox_backend=bwrap requires bubblewrap")
    root = Path(root).resolve()
    relative = workspace_path(root, workdir).relative_to(root)
    inside = (Path("/workspace") / relative).as_posix()
    argv = [executable, "--die-with-parent", "--unshare-all", "--new-session",
            "--setenv", "PATH", "/usr/bin:/bin", "--setenv", "HOME", "/workspace",
            "--setenv", "TMPDIR", "/tmp", "--setenv", "LANG", "C.UTF-8"]
    # ponytail: bind only the platform runtime. Add explicit read-only mounts
    # when a measured task family needs a compiler environment or dependency.
    for path in ("/usr", "/bin", "/lib", "/lib64"):
        if Path(path).exists():
            argv.extend(("--ro-bind", path, path))
    argv.extend(("--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp",
                 "--bind", str(root), "/workspace", "--chdir", inside,
                 "/bin/sh", "-c", command))
    return argv
