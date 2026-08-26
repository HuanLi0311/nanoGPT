#!/usr/bin/env python3
"""Read-only SSH probe for the fixed DLLM compute host."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def probe(host: str) -> dict:
    if not host or host.startswith("-") or any(c.isspace() for c in host):
        raise ValueError("host must be a single non-empty hostname")
    if shutil.which("ssh") is None:
        raise RuntimeError("ssh executable not found")
    remote = (
        "if [ -x /home/JJ_Group/lih2511/.conda/envs/smdm-baseline/bin/python ]; "
        "then DLLM_PYTHON=/home/JJ_Group/lih2511/.conda/envs/smdm-baseline/bin/python; "
        "else DLLM_PYTHON=$(command -v python3); fi; "
        "hostname; uname -a; printf 'PYTHON_PATH='; printf '%s\\n' \"$DLLM_PYTHON\"; "
        "\"$DLLM_PYTHON\" -V 2>&1; "
        "nvidia-smi --query-gpu=index,name,memory.total,driver_version "
        "--format=csv,noheader 2>&1 || true; "
        "df -Pk . 2>&1 || true; "
        "\"$DLLM_PYTHON\" -c "
        + shlex.quote(
            "import importlib.util, json; "
            "names=['torch','safetensors','transformers','numpy']; "
            "print('PACKAGES_JSON='+json.dumps({n: bool(importlib.util.find_spec(n)) for n in names}, sort_keys=True))"
        )
    )
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ServerAliveInterval=3",
        "-o",
        "ServerAliveCountMax=1",
        host,
        remote,
    ]
    started = _utc_now()
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=45, check=False)
        exit_code = result.returncode
        stdout = result.stdout
        stderr = result.stderr
    except subprocess.TimeoutExpired as exc:
        exit_code = 124
        stdout = exc.stdout or ""
        stderr = (exc.stderr or "") + "\nprobe timeout"
    parsed = {
        "hostname": None,
        "python": None,
        "python_path": None,
        "gpus": [],
        "packages": {},
        "disk": None,
    }
    lines = stdout.splitlines()
    for line in lines:
        if line.startswith("PACKAGES_JSON="):
            try:
                parsed["packages"] = json.loads(line.split("=", 1)[1])
            except json.JSONDecodeError:
                pass
        elif line.startswith("Python "):
            parsed["python"] = line
        elif line.startswith("PYTHON_PATH="):
            parsed["python_path"] = line.split("=", 1)[1]
        elif line.startswith("/usr/") or line.startswith("/"):
            if parsed["python_path"] is None:
                parsed["python_path"] = line
        elif line.startswith("Filesystem"):
            parsed["disk"] = line
        elif line and parsed["hostname"] is None:
            parsed["hostname"] = line
    for line in lines:
        if "," in line and "MiB" in line:
            fields = [field.strip() for field in line.split(",")]
            if len(fields) >= 4 and fields[0].isdigit():
                parsed["gpus"].append(
                    {"index": fields[0], "name": fields[1], "memory": fields[2], "driver": fields[3]}
                )
    return {
        "schema_version": 1,
        "probe_started_utc": started,
        "probe_finished_utc": _utc_now(),
        "host": host,
        "command": command,
        "ssh_exit_code": exit_code,
        "ok": exit_code == 0 and parsed["hostname"] == host,
        "parsed": parsed,
        "stdout": stdout,
        "stderr": stderr,
        "local_python": sys.version,
        "local_cwd": os.getcwd(),
    }


def self_check() -> None:
    assert _utc_now().endswith("+00:00")
    try:
        probe("bad host")
    except ValueError:
        return
    raise AssertionError("invalid host was accepted")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="air-node-03")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    if args.self_check:
        self_check()
        print(json.dumps({"self_check": "ok"}))
        return 0
    result = probe(args.host)
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
