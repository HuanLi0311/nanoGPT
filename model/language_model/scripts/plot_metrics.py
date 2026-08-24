"""Plot JSONL training metrics into the requested assets directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--metric", default="loss")
    args = parser.parse_args()
    import matplotlib.pyplot as plt

    rows = [json.loads(line) for line in args.log.read_text(encoding="utf-8").splitlines() if line.strip()]
    points = [(row["step"], row[args.metric]) for row in rows if args.metric in row]
    if not points:
        raise SystemExit(f"no metric {args.metric!r} in {args.log}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.plot(*zip(*points)); plt.xlabel("step"); plt.ylabel(args.metric); plt.tight_layout(); plt.savefig(args.output, dpi=160)


if __name__ == "__main__":
    main()
