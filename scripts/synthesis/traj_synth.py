"""Stages 2-3: construct environments, sample action paths, prove solvability."""

from __future__ import annotations

import random
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

if __package__:
    from .schema import fingerprint, iter_jsonl, relative_path, render, stable_json, validate_task, write_jsonl
else:
    from schema import fingerprint, iter_jsonl, relative_path, render, stable_json, validate_task, write_jsonl


OUTPUT_REF = re.compile(r"\{\{output:([^}]+)\}\}")


def construct_tasks(materials_path: Path, output: Path, *, variants_per_material: int = 1) -> dict[str, Any]:
    if variants_per_material < 1:
        raise ValueError("variants_per_material must be positive")
    material_count = task_count = 0
    signatures: set[str] = set()
    domain_counts: Counter[str] = Counter()
    task_type_counts: Counter[str] = Counter()
    capability_counts: Counter[str] = Counter()
    interface_counts: Counter[str] = Counter()

    def rows():
        nonlocal material_count, task_count
        for material_index, material in enumerate(iter_jsonl(materials_path)):
            material_count += 1
            content_path = materials_path.parent / relative_path(material["content_path"])
            material_text = content_path.read_text(encoding="utf-8")
            templates = material.get("task_templates") or [material.get("task_template")]
            if not templates or any(not isinstance(template, dict) for template in templates):
                raise ValueError(f"{material['material_id']}: no task templates")
            for variant_index in range(variants_per_material):
                template = deepcopy(templates[variant_index % len(templates)])
                executable = stable_json({"verifier": template.get("verifier"),
                                          "actions": template.get("actions", template.get("action_patterns"))})
                if "{{material}}" in executable or "{{source_uri}}" in executable:
                    raise ValueError(f"{material['material_id']}: untrusted material may only enter prompts or initial files")
                values = {"material": material_text,
                          "material_id": material["material_id"], "domain": material["domain"],
                          "subdomain": material["subdomain"], "concept": material["concept"],
                          "index": f"{task_count:06d}", "material_index": f"{material_index:06d}",
                          "variant_index": f"{variant_index:02d}",
                          "source_uri": material["provenance"]["uri"],
                          "material_sha256": material["provenance"]["sha256"]}
                recipe = render(template, values)
                actions = recipe.get("actions", recipe.get("action_patterns", []))
                tools = recipe.get("available_tools") or sorted({action.get("tool") for action in actions})
                files = {relative_path(render(path, values)): render(value, values)
                         for path, value in recipe.get("files", {}).items()}
                stem = re.sub(r"[^A-Za-z0-9._-]+", "-", str(recipe.get("id", material["concept"]))).strip("-")
                suffix = f"-v{variant_index:02d}" if variants_per_material > 1 else ""
                sandbox_backend = str(recipe.get("sandbox_backend", "bwrap"))
                task = {
                    "task_id": f"{stem}-{material['material_id'][4:]}{suffix}",
                    "task_variant": variant_index, "task_type": recipe.get("task_type", stem),
                    "capability": recipe.get("capability"), "interface": recipe.get("interface"),
                    "prompt": recipe["prompt"], "domain": material["domain"],
                    "subdomain": material["subdomain"], "task_family": material["family"],
                    "concepts": [material["concept"], *material.get("related_concepts", [])],
                    "materials": [material["material_id"]],
                    "material_provenance": [material["provenance"]], "files": files,
                    "available_tools": tools, "verifier": recipe["verifier"],
                    "verifier_version": recipe.get("verifier_version", "material-task-v1"),
                    "sandbox_backend": sandbox_backend,
                    "harness_version": recipe.get("harness_version", "bwrap-v1" if sandbox_backend == "bwrap" else "workspace-host-v2"),
                    "tool_schema_version": "workspace-tools-v2",
                    "initial_facts": recipe.get("initial_facts", ["workspace:ready"]),
                    "target_facts": recipe.get("target_facts", []),
                    "required_actions": recipe.get("required_actions", []),
                    "action_patterns": actions, "min_steps": int(recipe.get("min_steps", 1)),
                    "max_steps": int(recipe.get("max_steps", len(actions))),
                }
                task["initial_facts"] = sorted(set(task["initial_facts"]) | {f"file:{path}:exists" for path in files})
                task["seed_patterns"] = recipe.get("seed_patterns", [action["id"] for action in actions])
                task["candidate_patterns"] = recipe.get("candidate_patterns", task["seed_patterns"])
                validate_task(task)
                signature = fingerprint({key: task[key] for key in ("prompt", "files", "available_tools", "verifier",
                                                                     "initial_facts", "target_facts", "action_patterns")})
                task["task_signature"] = signature
                signatures.add(signature)
                domain_counts[str(task["domain"])] += 1
                task_type_counts[str(task["task_type"])] += 1
                capability_counts[str(task["capability"] or "unspecified")] += 1
                interface_counts[str(task["interface"] or "unspecified")] += 1
                task_count += 1
                yield task

    write_jsonl(output, rows())
    return {"materials": material_count, "tasks": task_count,
            "variants_per_material": variants_per_material,
            "unique_task_signatures": len(signatures),
            "domain_distribution": dict(sorted(domain_counts.items())),
            "task_type_distribution": dict(sorted(task_type_counts.items())),
            "capability_distribution": dict(sorted(capability_counts.items())),
            "interface_distribution": dict(sorted(interface_counts.items())),
            "manifest": str(output)}


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
        for previous in actions:
            if previous["id"] == action["id"]:
                continue
            shared = sorted(set(previous.get("effects", [])) & set(action.get("preconditions", [])))
            if previous["id"] in dependencies:
                relation, weight = "strong_dependency", 3
            elif shared:
                relation, weight = "weak_dependency", 2
            else:
                relation, weight = "independent", 1
            edge = {"from": previous["id"], "to": action["id"],
                    "relation": relation, "weight": weight}
            if shared:
                edge["facts"] = shared
            edges.append(edge)
    return {"version": "agentworld-weighted-action-graph-v1",
            "task_id": task["task_id"], "nodes": actions, "edges": edges,
            "initial_facts": task["initial_facts"], "target_facts": task["target_facts"]}


def sample_path(task: dict[str, Any], *, seed: int, policy: str = "goal", candidate_index: int = 0) -> list[dict[str, Any]]:
    graph = trajectory_graph(task)
    values = {"candidate_index": f"{candidate_index:04d}"}
    if policy not in {"agentworld", "goal", "uniform"}:
        raise ValueError(f"unknown path policy: {policy}")
    actions = {item["id"]: {**item,
                            "preconditions": render(item.get("preconditions", []), values),
                            "effects": render(item.get("effects", []), values)}
               for item in graph["nodes"]}
    target = set(render(graph["target_facts"], values))
    initial = set(render(graph["initial_facts"], values))
    required = set(task.get("required_actions", []))
    min_steps = int(task.get("min_steps", 1))
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
    weights = {(edge["from"], edge["to"]): int(edge["weight"]) for edge in graph["edges"]}
    strong_predecessors = {edge["to"] for edge in graph["edges"] if edge["weight"] == 3}

    # ponytail: exhaustive DFS is fine for the intended small recipes; switch
    # to beam/A* search if action graphs grow enough for this to become slow.
    def search(facts: set[str], used: tuple[str, ...]) -> list[str] | None:
        if target.issubset(facts) and required.issubset(used) and len(used) >= min_steps:
            return list(used)
        if len(used) >= task["max_steps"]:
            return None
        possible = []
        for action in actions.values():
            dependencies = set(action.get("depends_on", [])) | set(OUTPUT_REF.findall(str(action.get("arguments", action.get("arguments_template", {})))))
            if action["id"] in used or not dependencies.issubset(used) or not set(action.get("preconditions", [])).issubset(facts):
                continue
            gain = len(set(action.get("effects", [])) & (target - facts))
            if policy == "goal":
                score = 100 * gain + 10 * (action["id"] in relevant) + rng.random()
            elif policy == "agentworld":
                weight = weights.get((used[-1], action["id"]), 1) if used else (
                    3 if action["id"] not in strong_predecessors else 1)
                # Weighted random ordering followed by backtracking keeps the
                # Agent-World walk bias while guaranteeing a solvable path.
                score = rng.random() ** (1 / weight)
            else:
                score = rng.random()
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
                            verifier_version=task["verifier_version"],
                            sandbox_backend=task["sandbox_backend"])


def validate_oracles(tasks_path: Path, output: Path, *, seed: int = 0, policy: str = "agentworld",
                     trajectories_per_task: int = 1) -> dict[str, Any]:
    from model.language_model.scripts.synthesis.trace_runner import ProgrammaticTraceRunner
    from itertools import groupby

    if trajectories_per_task < 1:
        raise ValueError("trajectories_per_task must be positive")
    output.mkdir(parents=True, exist_ok=True)
    candidate_path = output / "candidate_trajectories.jsonl"
    graph_path = output / "trajectory_graphs.jsonl"
    generation_rejected: list[dict[str, Any]] = []
    task_count = candidate_count = 0
    path_signatures: set[str] = set()
    structure_signatures: set[str] = set()
    tool_sequences: set[tuple[str, ...]] = set()
    candidate_domains: Counter[str] = Counter()

    write_jsonl(graph_path, (trajectory_graph(task) for task in iter_jsonl(tasks_path)))

    def candidates():
        nonlocal task_count, candidate_count
        for task_index, task in enumerate(iter_jsonl(tasks_path)):
            task_count += 1
            for candidate_index in range(trajectories_per_task):
                try:
                    actions = sample_path(task, seed=seed + task_index * trajectories_per_task + candidate_index,
                                          policy=policy, candidate_index=candidate_index)
                    path_signature = fingerprint(actions)
                    structure_signature = fingerprint([(action["id"], action["tool"]) for action in actions])
                    tools = tuple(action["tool"] for action in actions)
                    path_signatures.add(path_signature)
                    structure_signatures.add(structure_signature)
                    tool_sequences.add(tools)
                    candidate_domains[str(task["domain"])] += 1
                    candidate_count += 1
                    yield {"candidate_id": f"candidate-{task_index:06d}-{candidate_index:03d}",
                           "task_id": task["task_id"], "candidate_index": candidate_index,
                           "path_policy": policy, "actions": actions, "path_signature": path_signature,
                           "structure_signature": structure_signature, "tool_sequence": list(tools)}
                except Exception as error:
                    generation_rejected.append({"task_id": task.get("task_id"),
                                                "candidate_index": candidate_index,
                                                "reason": str(error)})

    write_jsonl(candidate_path, candidates())
    write_jsonl(output / "rejected_candidate_generation.jsonl", generation_rejected)

    runner = ProgrammaticTraceRunner(output / "workspaces")
    destinations = [output / name for name in ("oracle_episodes.jsonl", "validated_tasks.jsonl",
                                                "rejected_tasks.jsonl", "rejected_trajectories.jsonl")]
    temporary = [path.with_suffix(path.suffix + ".tmp") for path in destinations]
    handles = [path.open("w", encoding="utf-8") for path in temporary]
    episodes_count = validated = rejected = passed_trajectories = rejected_trajectories = 0
    lengths: list[int] = []
    groups = iter(groupby(iter_jsonl(candidate_path), key=lambda row: row["task_id"]))
    current = next(groups, None)
    try:
        for task in iter_jsonl(tasks_path):
            rows = []
            if current and current[0] == task["task_id"]:
                rows = list(current[1])
                current = next(groups, None)
            proofs = []
            if not rows:
                handles[2].write(stable_json({"task_id": task.get("task_id"),
                                              "reason": "no candidate trajectory was generated"}) + "\n")
                rejected += 1
                continue
            try:
                initial = _initial_verifier(task)
                if initial["harness_status"] != "healthy" or initial["task_success"]:
                    raise ValueError("verifier is unhealthy or task already passes in the initial state")
            except Exception as error:
                reason = str(error)
                handles[2].write(stable_json({"task_id": task.get("task_id"), "reason": reason}) + "\n")
                for row in rows:
                    handles[3].write(stable_json({"candidate_id": row["candidate_id"],
                                                  "task_id": task["task_id"], "reason": reason}) + "\n")
                    rejected_trajectories += 1
                rejected += 1
                continue
            for row in rows:
                episode_id = f"oracle:{task['task_id']}:{row['candidate_index']:04d}"
                try:
                    episode = runner.run(task, row["actions"], episode_id,
                                         candidate_index=row["candidate_index"])
                except Exception as error:
                    handles[3].write(stable_json({"candidate_id": row["candidate_id"],
                                                  "task_id": task["task_id"],
                                                  "reason": f"oracle execution error: {error}"}) + "\n")
                    rejected_trajectories += 1
                    continue
                episode["policy"] = {"kind": "programmatic_oracle", "model": "agentworld-weighted-graph"}
                handles[0].write(stable_json(episode) + "\n")
                episodes_count += 1
                outcome = episode["outcome"]
                passed = (outcome["task_success"] and outcome["independent_verifier_passed"]
                          and outcome["harness_status"] == "healthy"
                          and outcome["call_result_linkage_complete"] and outcome["trace_fidelity"])
                if passed:
                    proof = {"episode_id": episode["episode_id"],
                             "candidate_id": row["candidate_id"],
                             "path": [action["id"] for action in row["actions"]],
                             "path_policy": policy, "episode_sha256": fingerprint(episode)}
                    proofs.append(proof)
                    lengths.append(len(row["actions"]))
                    passed_trajectories += 1
                else:
                    handles[3].write(stable_json({"candidate_id": row["candidate_id"],
                                                  "task_id": task["task_id"],
                                                  "reason": f"oracle failed: {outcome.get('failure_class')}"}) + "\n")
                    rejected_trajectories += 1
            if proofs:
                handles[1].write(stable_json({**task, "oracle_proof": proofs[0],
                                              "oracle_proofs": proofs}) + "\n")
                validated += 1
            else:
                handles[2].write(stable_json({"task_id": task.get("task_id"),
                                              "reason": "all candidate trajectories failed"}) + "\n")
                rejected += 1
    except Exception:
        for handle in handles:
            handle.close()
        for path in temporary:
            path.unlink(missing_ok=True)
        raise
    else:
        for handle in handles:
            handle.close()
        for source, destination in zip(temporary, destinations, strict=True):
            source.replace(destination)

    path_lengths = ({"count": len(lengths), "min": min(lengths), "max": max(lengths),
                     "mean": sum(lengths) / len(lengths)} if lengths else
                    {"count": 0, "min": None, "max": None, "mean": None})
    return {"tasks": task_count, "requested_trajectories": task_count * trajectories_per_task,
            "candidate_trajectories": candidate_count,
            "candidate_generation_rejected": len(generation_rejected),
            "trajectories_per_task": trajectories_per_task,
            "unique_path_signatures": len(path_signatures),
            "unique_path_structures": len(structure_signatures),
            "unique_tool_combinations": len(tool_sequences),
            "domain_distribution": dict(sorted(candidate_domains.items())),
            "validated": validated, "rejected": rejected,
            "passed_trajectories": passed_trajectories,
            "rejected_trajectories": rejected_trajectories,
            "executed_trajectories": episodes_count, "path_lengths": path_lengths,
            "candidate_manifest": str(candidate_path), "trajectory_graphs": str(graph_path),
            "validated_tasks": str(output / "validated_tasks.jsonl"),
            "oracle_episodes": str(output / "oracle_episodes.jsonl")}
