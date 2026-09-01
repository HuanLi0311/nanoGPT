"""Expand the verified task graph into balanced trajectory-length variants."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


PROFILE_PADDING = {"short": 0, "medium": 4, "long": 10}
PROFILE_TARGETS = {"short": 2286, "medium": 4000, "long": 0}
DOMAIN_TARGETS = {
    "workspace": 11428,
    "coding": 11429,
    "configuration": 11429,
    "data": 11428,
    "process": 11429,
    "artifact": 11428,
    "documentation": 11429,
}


def distribute(total: int, count: int) -> list[int]:
    base, remainder = divmod(total, count)
    return [base + int(index < remainder) for index in range(count)]


def padding_patterns(task_id: str, profile: str, count: int) -> list[dict]:
    patterns = []
    previous = "trajectory:base_complete"
    commands = [
        "mkdir -p .trace && printf 'candidate-{candidate_index}-step-01\\n' > .trace/step-01-{candidate_index}.txt",
        "wc -c < .trace/step-01-{candidate_index}.txt > .trace/step-02-{candidate_index}.txt",
        "sha256sum .trace/step-01-{candidate_index}.txt | awk '{print $1}' > .trace/step-03-{candidate_index}.txt",
        "grep -q . .trace/step-03-{candidate_index}.txt && printf checked > .trace/step-04-{candidate_index}.ok",
        "find .trace -maxdepth 1 -type f -printf '%f\\n' | sort > .trace/index-{candidate_index}.txt",
        "test -s .trace/index-{candidate_index}.txt && printf indexed > .trace/step-06-{candidate_index}.ok",
        "cat .trace/step-03-{candidate_index}.txt > /dev/null",
        "wc -l .trace/index-{candidate_index}.txt > .trace/step-08-{candidate_index}.txt",
        "test -s .trace/step-08-{candidate_index}.txt && printf measured > .trace/step-09-{candidate_index}.ok",
        "find .trace -type f | sort | tail -n 1 > .trace/step-10-{candidate_index}.txt",
    ]
    tools = ["exec_command"] * len(commands)
    for index in range(count):
        step = index + 1
        pattern_id = f"trace_{profile}_{step:02d}"
        patterns.append({
            "id": pattern_id,
            "tool": tools[index],
            "arguments_template": {"cmd": commands[index]},
            "preconditions": [previous],
            "effects": [f"trajectory:step:{step:02d}"],
            "novelty": "new_path" if profile != "short" else "known_replay",
        })
        previous = f"trajectory:step:{step:02d}"
    return patterns


def variant(base: dict, profile: str, rollouts: int) -> dict:
    task = copy.deepcopy(base)
    task_id = f"{base['task_id']}__{profile}"
    padding_count = PROFILE_PADDING[profile]
    patterns = copy.deepcopy(base["action_patterns"])
    if padding_count:
        patterns[-1]["effects"] = [*patterns[-1].get("effects", []), "trajectory:base_complete"]
        patterns.extend(padding_patterns(task_id, profile, padding_count))
    padding_ids = [f"trace_{profile}_{index:02d}" for index in range(1, padding_count + 1)]
    task.update({
        "task_id": task_id,
        "concepts": [*base.get("concepts", []), f"trajectory_length_{profile}"],
        "materials": [*base.get("materials", []), f"trajectory_profile_{profile}"],
        "action_patterns": patterns,
        "seed_patterns": [*base["seed_patterns"], *padding_ids],
        "candidate_patterns": [*base.get("candidate_patterns", base["seed_patterns"]), *padding_ids],
        "max_steps": int(base.get("max_steps", 8)) + padding_count,
        "rollouts": rollouts,
        "trajectory_profile": profile,
        "padding_steps": padding_count,
        "expected_tool_calls": len(base["action_patterns"]) + padding_count,
        "verifier_version": "synthesis-80k-v1",
    })
    return task


def build(source: Path) -> list[dict]:
    bases = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_domain: dict[str, list[dict]] = {domain: [] for domain in DOMAIN_TARGETS}
    for task in bases:
        if task.get("domain") not in by_domain:
            raise ValueError(f"unexpected domain: {task.get('domain')}")
        by_domain[task["domain"]].append(task)

    output = []
    for domain, target in DOMAIN_TARGETS.items():
        tasks = by_domain[domain]
        if not tasks:
            raise ValueError(f"no tasks for domain {domain}")
        short = PROFILE_TARGETS["short"]
        medium = PROFILE_TARGETS["medium"]
        long = target - short - medium
        profile_totals = {"short": short, "medium": medium, "long": long}
        for profile in ("short", "medium", "long"):
            for base, rollouts in zip(tasks, distribute(profile_totals[profile], len(tasks))):
                output.append(variant(base, profile, rollouts))

    if sum(task["rollouts"] for task in output) != 80000:
        raise AssertionError("task rollout budget must be exactly 80,000")
    if sum(task["rollouts"] for task in output if task["trajectory_profile"] == "long") != 35998:
        raise AssertionError("long trajectory budget must be 35,998")
    return output


def main() -> None:
    root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=root / "agent/tasks/synthesis_full.jsonl")
    parser.add_argument("--output", type=Path, default=root / "agent/tasks/synthesis_80k.jsonl")
    args = parser.parse_args()
    tasks = build(args.source)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(args.output)
    print(json.dumps({
        "tasks": len(tasks),
        "rollouts": sum(task["rollouts"] for task in tasks),
        "domains": {domain: sum(task["rollouts"] for task in tasks if task["domain"] == domain) for domain in DOMAIN_TARGETS},
        "profiles": {profile: sum(task["rollouts"] for task in tasks if task["trajectory_profile"] == profile) for profile in PROFILE_PADDING},
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
