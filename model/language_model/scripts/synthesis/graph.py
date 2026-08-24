"""Build and compose the semantic/execution graphs used by synthesis."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

from .schema import fingerprint, validate_action, validate_episode, validate_task


def _format(value: Any, candidate_index: int) -> Any:
    if isinstance(value, str):
        return value.replace("{candidate_index}", f"{candidate_index:04d}")
    if isinstance(value, list):
        return [_format(item, candidate_index) for item in value]
    if isinstance(value, dict):
        return {key: _format(item, candidate_index) for key, item in value.items()}
    return value


def _facts_from_state(state: dict[str, Any]) -> set[str]:
    environment = state.get("environment_state", {})
    facts = {"workspace:ready"}
    for path in environment.get("files", {}):
        facts.add(f"file:{path}:exists")
    return facts


def _pattern(pattern: dict[str, Any], task_id: str) -> dict[str, Any]:
    validate_action(pattern, allow_template=True)
    normalized = deepcopy(pattern)
    normalized.setdefault("id", f"{task_id}:{pattern['tool']}")
    normalized.setdefault("preconditions", [])
    normalized.setdefault("effects", [])
    normalized["task_id"] = task_id
    normalized["provenance"] = {
        "source": "task_action_pattern",
        "task_id": task_id,
    }
    return normalized


def build_graph(tasks: Iterable[dict[str, Any]], episodes: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Create a graph with semantic edges and concrete execution evidence."""

    tasks = list(tasks)
    episodes = list(episodes)
    for task in tasks:
        validate_task(task)
    for episode in episodes:
        validate_episode(episode)

    task_by_id = {task["task_id"]: task for task in tasks}
    semantic_edges: list[dict[str, str]] = []
    action_patterns: list[dict[str, Any]] = []
    for task in tasks:
        domain = str(task.get("domain", "unknown"))
        subdomain = str(task.get("subdomain", task.get("task_family", "unknown")))
        family = str(task.get("task_family", "unknown"))
        semantic_edges.append({"subject": domain, "relation": "has_subdomain", "object": subdomain})
        semantic_edges.append({"subject": subdomain, "relation": "has_task_family", "object": family})
        semantic_edges.append({"subject": family, "relation": "synthesizes", "object": task["task_id"]})
        for concept in task.get("concepts", []):
            semantic_edges.append({"subject": subdomain, "relation": "has_concept", "object": str(concept)})
            semantic_edges.append({"subject": str(concept), "relation": "instantiates", "object": task["task_id"]})
        for material in task.get("materials", []):
            semantic_edges.append({"subject": task["task_id"], "relation": "uses_material", "object": str(material)})
        normalized_patterns = [_pattern(pattern, task["task_id"]) for pattern in task["action_patterns"]]
        action_patterns.extend(normalized_patterns)
        for pattern in normalized_patterns:
            pattern_id = str(pattern["id"])
            semantic_edges.append({"subject": task["task_id"], "relation": "has_action_pattern", "object": pattern_id})
            semantic_edges.append({"subject": pattern_id, "relation": "uses_tool", "object": str(pattern["tool"])})
            for precondition in pattern.get("preconditions", []):
                semantic_edges.append({"subject": pattern_id, "relation": "requires", "object": str(precondition)})
            for effect in pattern.get("effects", []):
                semantic_edges.append({"subject": pattern_id, "relation": "produces", "object": str(effect)})

    state_nodes: dict[str, dict[str, Any]] = {}
    transition_edges: list[dict[str, Any]] = []
    event_nodes: list[dict[str, Any]] = []
    for episode in episodes:
        task = task_by_id.get(episode["task_id"], {})
        for event in episode["events"]:
            before = event["state_before"]
            after = event["state_after"]
            before_id = before["state_hash"]
            after_id = after["state_hash"]
            state_nodes.setdefault(before_id, {
                "id": before_id,
                "task_id": episode["task_id"],
                "environment_facts": sorted(_facts_from_state(before)),
                "components": ["environment_state", "agent_context_state", "harness_state", "goal_verifier_state"],
            })
            state_nodes.setdefault(after_id, {
                "id": after_id,
                "task_id": episode["task_id"],
                "environment_facts": sorted(_facts_from_state(after)),
                "components": ["environment_state", "agent_context_state", "harness_state", "goal_verifier_state"],
            })
            if event["kind"] == "tool_call":
                action = event["action"]
                event_id = event["event_id"]
                event_nodes.append({
                    "id": event_id,
                    "kind": "transition",
                    "task_id": episode["task_id"],
                    "episode_id": episode["episode_id"],
                    "tool_call_id": event["tool_call_id"],
                    "tool": action["tool"],
                    "state_before": before_id,
                    "state_after": after_id,
                    "state_delta": event["state_delta"],
                    "provenance": event["provenance"],
                })
                transition_edges.append({
                    "subject": before_id,
                    "relation": "from",
                    "object": event_id,
                    "provenance": {"episode_id": episode["episode_id"], "event_id": event_id},
                })
                transition_edges.append({
                    "subject": event_id,
                    "relation": "to",
                    "object": after_id,
                    "provenance": {"episode_id": episode["episode_id"], "event_id": event_id},
                })
                transition_edges.append({
                    "subject": event_id,
                    "relation": "invokes",
                    "object": action["tool"],
                    "provenance": {"episode_id": episode["episode_id"], "event_id": event_id},
                })
            else:
                event_nodes.append({
                    "id": event["event_id"],
                    "kind": "independent_verifier",
                    "task_id": episode["task_id"],
                    "episode_id": episode["episode_id"],
                    "state_before": before_id,
                    "state_after": after_id,
                    "provenance": event["provenance"],
                })

    return {
        "graph_version": "execution-graph-v1",
        "semantic_graph_version": "concept-task-v1",
        "semantic_edges": semantic_edges,
        "state_nodes": sorted(state_nodes.values(), key=lambda item: item["id"]),
        "event_nodes": event_nodes,
        "transition_edges": transition_edges,
        "action_patterns": action_patterns,
        "provenance": {
            "episode_ids": [episode["episode_id"] for episode in episodes],
            "task_ids": [task["task_id"] for task in tasks],
        },
    }


def _patterns_for_task(graph: dict[str, Any], task_id: str) -> dict[str, dict[str, Any]]:
    return {
        pattern["id"]: pattern
        for pattern in graph.get("action_patterns", [])
        if pattern.get("task_id") == task_id
    }


def compose(
    task: dict[str, Any],
    graph: dict[str, Any],
    *,
    candidate_index: int = 0,
    required_patterns: list[str] | None = None,
    max_steps: int = 8,
) -> dict[str, Any]:
    """Compose a compatible concrete action sequence from abstract patterns.

    Only precondition/effect-compatible patterns are joined.  The result is
    still a candidate until a live trace runner executes it.
    """

    validate_task(task)
    patterns = _patterns_for_task(graph, task["task_id"])
    required = list(required_patterns or task.get("required_patterns", []))
    missing = [pattern_id for pattern_id in required if pattern_id not in patterns]
    if missing:
        raise ValueError(f"{task['task_id']}: missing required patterns {missing}")
    if not required:
        required = list(patterns)

    start_facts = set(task.get("initial_facts", ["workspace:ready"]))
    start_facts.update(f"file:{path}:exists" for path in task.get("files", {}))
    target = set(task.get("target_facts", []))
    order = list(patterns)

    def search(
        facts: set[str],
        used: tuple[str, ...],
        actions: list[dict[str, Any]],
    ) -> list[dict[str, Any]] | None:
        if set(required).issubset(used) and target.issubset(facts):
            return actions
        if len(actions) >= max_steps:
            return None
        for pattern_id in order:
            if pattern_id in used:
                continue
            pattern = patterns[pattern_id]
            preconditions = set(_format(pattern.get("preconditions", []), candidate_index))
            if not preconditions.issubset(facts):
                continue
            effects = _format(pattern.get("effects", []), candidate_index)
            action = {
                "id": pattern_id,
                "tool": pattern["tool"],
                "arguments": _format(
                    pattern.get("arguments", pattern.get("arguments_template", {})),
                    candidate_index,
                ),
                "preconditions": sorted(preconditions),
                "effects": effects,
                "provenance": {
                    "pattern_id": pattern_id,
                    "task_id": task["task_id"],
                    "graph_version": graph.get("graph_version"),
                },
            }
            next_facts = facts | set(effects)
            result = search(next_facts, (*used, pattern_id), [*actions, action])
            if result is not None:
                return result
        return None

    actions = search(start_facts, (), [])
    if actions is None:
        raise ValueError(f"{task['task_id']}: no compatible action sequence")
    candidate = {
        "candidate_id": f"{task['task_id']}:candidate:{candidate_index:04d}",
        "task_id": task["task_id"],
        "candidate_index": candidate_index,
        "action_pattern_ids": [action["id"] for action in actions],
        "predicted_facts": sorted(start_facts | {
            effect for action in actions for effect in action.get("effects", [])
        }),
        "actions": actions,
    }
    candidate["candidate_fingerprint"] = fingerprint({
        "task_id": task["task_id"],
        "files": task["files"],
        "actions": actions,
    })
    return candidate
