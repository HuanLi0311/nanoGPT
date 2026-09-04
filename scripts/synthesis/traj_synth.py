"""Stages 2-3: construct environments, sample action paths, prove solvability."""

from __future__ import annotations

import random
import re
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

if __package__:
    from .schema import fingerprint, read_jsonl, relative_path, render, stable_json, validate_task, write_json, write_jsonl
else:
    from schema import fingerprint, read_jsonl, relative_path, render, stable_json, validate_task, write_json, write_jsonl


OUTPUT_REF = re.compile(r"\{\{output:([^}]+)\}\}")


def construct_tasks(materials_path: Path, output: Path) -> list[dict[str, Any]]:
    tasks = []
    for index, material in enumerate(read_jsonl(materials_path)):
        content_path = materials_path.parent / relative_path(material["content_path"])
        template = deepcopy(material["task_template"])
        executable = stable_json({"verifier": template.get("verifier"),
                                  "actions": template.get("actions", template.get("action_patterns"))})
        if "{{material}}" in executable or "{{source_uri}}" in executable:
            raise ValueError(f"{material['material_id']}: untrusted material may only enter prompts or initial files")
        values = {"material": content_path.read_text(encoding="utf-8"), "material_id": material["material_id"],
                  "domain": material["domain"], "subdomain": material["subdomain"],
                  "concept": material["concept"], "index": f"{index:04d}",
                  "source_uri": material["provenance"]["uri"],
                  "material_sha256": material["provenance"]["sha256"]}
        recipe = render(template, values)
        actions = recipe.get("actions", recipe.get("action_patterns", []))
        tools = recipe.get("available_tools") or sorted({action.get("tool") for action in actions})
        files = {relative_path(render(path, values)): render(value, values) for path, value in recipe.get("files", {}).items()}
        stem = re.sub(r"[^A-Za-z0-9._-]+", "-", str(recipe.get("id", material["concept"]))).strip("-")
        task_id = f"{stem}-{material['material_id'][4:]}"
        task = {
            "task_id": task_id, "prompt": recipe["prompt"], "domain": material["domain"],
            "subdomain": material["subdomain"], "task_family": material["family"],
            "concepts": [material["concept"]], "materials": [material["material_id"]],
            "material_provenance": [material["provenance"]], "files": files,
            "available_tools": tools, "verifier": recipe["verifier"],
            "verifier_version": recipe.get("verifier_version", "material-task-v1"),
            "harness_version": "workspace-host-v2", "tool_schema_version": "workspace-tools-v2",
            "initial_facts": recipe.get("initial_facts", ["workspace:ready"]),
            "target_facts": recipe.get("target_facts", []),
            "required_actions": recipe.get("required_actions", []),
            "action_patterns": actions, "max_steps": int(recipe.get("max_steps", len(actions))),
        }
        task["initial_facts"] = sorted(set(task["initial_facts"]) | {f"file:{path}:exists" for path in files})
        task["seed_patterns"] = recipe.get("seed_patterns", [action["id"] for action in actions])
        task["candidate_patterns"] = recipe.get("candidate_patterns", task["seed_patterns"])
        validate_task(task)
        tasks.append(task)
    write_jsonl(output, tasks)
    return tasks


def trajectory_graph(task: dict[str, Any]) -> dict[str, Any]:
    validate_task(task)
    actions, edges = task["action_patterns"], []
    ids = {action["id"] for action in actions}
    for action in actions:
        arguments = action.get("arguments", action.get("arguments_template", {}))
        refs = set(OUTPUT_REF.findall(str(arguments)))
        dependencies = set(action.get("depends_on", [])) | refs
        if not dependencies.issubset(ids):
            raise ValueError(f"{task['task_id']}:{action['id']}: unknown dependencies {sorted(dependencies - ids)}")
        edges.extend({"from": dependency, "to": action["id"], "relation": "parameter_dependency"}
                     for dependency in sorted(dependencies))
        for previous in actions:
            shared = set(previous.get("effects", [])) & set(action.get("preconditions", []))
            edges.extend({"from": previous["id"], "to": action["id"], "relation": "fact_dependency", "fact": fact}
                         for fact in sorted(shared) if previous["id"] != action["id"])
    return {"task_id": task["task_id"], "nodes": actions, "edges": edges,
            "initial_facts": task["initial_facts"], "target_facts": task["target_facts"]}


def sample_path(task: dict[str, Any], *, seed: int, policy: str = "goal", candidate_index: int = 0) -> list[dict[str, Any]]:
    graph = trajectory_graph(task)
    values = {"candidate_index": f"{candidate_index:04d}"}
    if policy not in {"goal", "uniform"}:
        raise ValueError(f"unknown path policy: {policy}")
    actions = {item["id"]: {**item,
                            "preconditions": render(item.get("preconditions", []), values),
                            "effects": render(item.get("effects", []), values)}
               for item in graph["nodes"]}
    target = set(render(graph["target_facts"], values))
    initial = set(render(graph["initial_facts"], values))
    required = set(task.get("required_actions", []))
    if not required.issubset(actions):
        raise ValueError(f"{task['task_id']}: unknown required_actions")
    relevant, needed = set(), set(target)
    for _ in actions:
        for action in actions.values():
            if set(action.get("effects", [])) & needed:
                relevant.add(action["id"])
                needed.update(action.get("preconditions", []))
                needed.update(f"action:{item}" for item in action.get("depends_on", []))
    rng = random.Random(seed)

    # ponytail: exhaustive DFS is fine for the intended small recipes; switch
    # to beam/A* search if action graphs grow enough for this to become slow.
    def search(facts: set[str], used: tuple[str, ...]) -> list[str] | None:
        if target.issubset(facts) and required.issubset(used):
            return list(used)
        if len(used) >= task["max_steps"]:
            return None
        possible = []
        for action in actions.values():
            dependencies = set(action.get("depends_on", [])) | set(OUTPUT_REF.findall(str(action.get("arguments", action.get("arguments_template", {})))))
            if action["id"] in used or not dependencies.issubset(used) or not set(action.get("preconditions", [])).issubset(facts):
                continue
            gain = len(set(action.get("effects", [])) & (target - facts))
            score = 100 * gain + 10 * (action["id"] in relevant) + rng.random() if policy == "goal" else rng.random()
            possible.append((score, action))
        for _, action in sorted(possible, key=lambda item: item[0], reverse=True):
            found = search(facts | set(action.get("effects", [])), (*used, action["id"]))
            if found is not None:
                return found
        return None

    selected = search(initial, ())
    if selected is None:
        raise ValueError(f"{task['task_id']}: no path reaches the declared goal")
    return [{**deepcopy(actions[action_id]),
             "arguments": render(actions[action_id].get("arguments", actions[action_id].get("arguments_template", {})), values)}
            for action_id in selected]


def _initial_verifier(task: dict[str, Any]) -> dict[str, Any]:
    from agent.verifier.verifier import run_verifier

    with TemporaryDirectory(prefix="synthesis-initial-") as temporary:
        root = Path(temporary)
        for relative, content in task["files"].items():
            path = root / relative_path(relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(content), encoding="utf-8")
        return run_verifier(str(root), task["verifier"], task_id=task["task_id"],
                            verifier_version=task["verifier_version"])


def validate_oracles(tasks_path: Path, output: Path, *, seed: int = 0, policy: str = "goal") -> dict[str, Any]:
    from model.language_model.scripts.synthesis.trace_runner import ProgrammaticTraceRunner

    tasks, accepted, rejected, episodes, graphs = read_jsonl(tasks_path), [], [], [], []
    runner = ProgrammaticTraceRunner(output / "workspaces")
    for index, task in enumerate(tasks):
        try:
            initial = _initial_verifier(task)
            if initial["harness_status"] != "healthy" or initial["task_success"]:
                raise ValueError("verifier is unhealthy or task already passes in the initial state")
            actions = sample_path(task, seed=seed + index, policy=policy, candidate_index=index)
            episode = runner.run(task, actions, f"oracle:{task['task_id']}:{index:04d}", candidate_index=index)
            episode["policy"] = {"kind": "programmatic_oracle", "model": "deterministic_action_graph"}
            episodes.append(episode)
            outcome = episode["outcome"]
            passed = (outcome["task_success"] and outcome["independent_verifier_passed"]
                      and outcome["harness_status"] == "healthy"
                      and outcome["call_result_linkage_complete"] and outcome["trace_fidelity"])
            if not passed:
                raise ValueError(f"oracle failed: {outcome.get('failure_class')}")
            proof = {"episode_id": episode["episode_id"], "path": [action["id"] for action in actions],
                     "path_policy": policy, "episode_sha256": fingerprint(episode)}
            accepted.append({**task, "oracle_proof": proof})
            graphs.append(trajectory_graph(task))
        except Exception as error:
            rejected.append({"task_id": task.get("task_id"), "reason": str(error)})
    write_jsonl(output / "oracle_episodes.jsonl", episodes)
    write_jsonl(output / "validated_tasks.jsonl", accepted)
    write_jsonl(output / "rejected_tasks.jsonl", rejected)
    write_json(output / "trajectory_graph.json", {"version": "trajectory-graph-v1", "policy": policy, "tasks": graphs})
    lengths = [len(episode["actions"]) for episode in episodes if episode["outcome"]["task_success"]]
    return {"tasks": len(tasks), "validated": len(accepted), "rejected": len(rejected),
            "path_lengths": lengths,
            "unique_tool_combinations": len({tuple(action["tool"] for action in episode["actions"])
                                             for episode in episodes if episode["outcome"]["task_success"]}),
            "validated_tasks": str(output / "validated_tasks.jsonl"),
            "oracle_episodes": str(output / "oracle_episodes.jsonl")}
