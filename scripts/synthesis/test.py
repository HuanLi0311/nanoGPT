"""One dependency-light end-to-end check for the four-stage pipeline."""

from __future__ import annotations

import json
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread

if __package__:
    from .graph import _discover, _retrieve, _sample, expand_knowledge_graph
    from .policy_rollout import run_policy_tasks
    from .runner import diagnose, finalize, prepare
    from .schema import read_json, read_jsonl, validate_plan, write_json, write_jsonl
    from .traj_synth import construct_tasks, synthesize_environment_templates, trajectory_graph
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from graph import _discover, _retrieve, _sample, expand_knowledge_graph
    from policy_rollout import run_policy_tasks
    from runner import diagnose, finalize, prepare
    from schema import read_json, read_jsonl, validate_plan, write_json, write_jsonl
    from traj_synth import construct_tasks, synthesize_environment_templates, trajectory_graph


def _plan(search_endpoint: str | None = None) -> dict:
    coding_source = ({"kind": "web_search", "search_endpoint": search_endpoint,
                      "provider": "test-search", "max_results": 2, "min_chars": 1,
                      "license": "test-only"}
                     if search_endpoint else
                     {"kind": "document", "uri": "alpha.txt", "license": "test-only"})
    return {
        "profiles": {
            "narrow": {"count": 2, "distribution": {"coding.transform": 1, "artifact.create": 0}},
            "diverse": {"count": 2, "distribution": {"coding.transform": 1, "artifact.create": 1}},
            "domains": {"count": 3, "distribution": {"coding": 1, "artifact": 1}},
        },
        "domains": [
            {"name": "coding", "subdomains": [{"name": "transform", "concepts": [{
                "name": "uppercase", "source": coding_source,
                "task": {
                    "id": "digest-{{variant_index}}",
                    "prompt": "Compute input.txt's SHA-256 and save it in answer-{{variant_index}}.txt.",
                    "files": {"input.txt": "{{material}}"},
                    "available_tools": ["exec_command"],
                    "verifier": {"command": "test \"$(cat answer-{{variant_index}}.txt)\" = {{material_sha256}}"},
                    "initial_facts": ["workspace:ready"],
                    "target_facts": ["file:answer-{{variant_index}}.txt:exists"],
                    "actions": [
                        {"id": "write_digest", "tool": "exec_command",
                         "arguments": {"cmd": "test ! -e /home && set -- $(sha256sum input.txt) && printf '%s\\n' \"$1\" > answer-{{variant_index}}.txt"},
                         "preconditions": ["file:input.txt:exists"],
                         "effects": ["file:answer-{{variant_index}}.txt:exists"]},
                    ],
                },
            }]}]},
            {"name": "artifact", "subdomains": [{"name": "create", "concepts": [{
                "name": "report", "source": {"kind": "repo", "uri": "repo", "glob": "*.txt", "license": "test-only"},
                "task": {
                    "id": "report-{{variant_index}}",
                    "prompt": "Create report-{{variant_index}}.txt containing BETA.", "files": {},
                    "available_tools": ["apply_patch"],
                    "verifier": "test \"$(cat report-{{variant_index}}.txt)\" = BETA",
                    "target_facts": ["file:report-{{variant_index}}.txt:exists"],
                    "actions": [{"id": "write_report", "tool": "apply_patch",
                                 "arguments": {"patch": "*** Begin Patch\n*** Add File: report-{{variant_index}}.txt\n+BETA\n*** End Patch"},
                                 "preconditions": ["workspace:ready"],
                                 "effects": ["file:report-{{variant_index}}.txt:exists"]}],
                },
            }]}]},
        ],
    }


def main() -> None:
    seed_plan = read_json(Path(__file__).with_name("k3_15domain_seeds.json"))
    validate_plan(seed_plan, require_tasks=False)
    seed_sample = _sample(seed_plan, "pilot", seed=7, count=None, weights=None)
    seed_counts = [sum(row["domain"] == domain["name"] for row in seed_sample)
                   for domain in seed_plan["domains"]]
    assert len(seed_sample) == 200 and set(seed_counts) == {13, 14}

    cyclic = _plan()
    first = cyclic["domains"][0]["subdomains"][0]["concepts"][0]
    second = cyclic["domains"][1]["subdomains"][0]["concepts"][0]
    first.update({"node_id": "concept:a", "parent_ids": ["concept:b"]})
    second.update({"node_id": "concept:b", "parent_ids": ["concept:a"]})
    try:
        validate_plan(cyclic)
        raise AssertionError("cyclic concept graph was accepted")
    except ValueError as error:
        assert "cycle" in str(error)

    sampled = _sample(_plan(), "domains", seed=7, count=None, weights=None)
    assert sorted(sum(row["domain"] == domain for row in sampled) for domain in {"coding", "artifact"}) == [1, 2]

    weighted = trajectory_graph({
        "task_id": "weighted-edges", "prompt": "fixture", "files": {}, "verifier": "false",
        "available_tools": ["exec_command"], "sandbox_backend": "workspace_host",
        "initial_facts": ["workspace:ready"], "target_facts": ["done"], "max_steps": 4,
        "action_patterns": [
            {"id": "source", "tool": "exec_command", "arguments": {"cmd": "true"},
             "preconditions": ["workspace:ready"], "effects": ["value:id"]},
            {"id": "strong", "tool": "exec_command", "arguments": {"cmd": "echo {{output:source}}"},
             "depends_on": ["source"], "preconditions": ["value:id"], "effects": ["done"]},
            {"id": "weak", "tool": "exec_command", "arguments": {"cmd": "true"},
             "preconditions": ["value:id"], "effects": ["optional"]},
            {"id": "independent", "tool": "exec_command", "arguments": {"cmd": "true"},
             "preconditions": ["workspace:ready"], "effects": ["other"]},
        ]})
    assert {edge["weight"] for edge in weighted["edges"]} == {1, 2, 3}

    with TemporaryDirectory(prefix="four-stage-synthesis-") as temporary:
        root = Path(temporary)

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path.startswith("/search"):
                    if self.path.startswith("/search-json"):
                        body = json.dumps({"web": {"results": [{"url": "/json-page", "title": "JSON"}]}}).encode()
                    else:
                        body = (b'<a class="result__a" href="/alpha-a">Alpha A</a>'
                                b'<a class="result__a" href="/alpha-b">Alpha B</a>')
                elif self.path.startswith("/alpha-a"):
                    body = b"<html><body>alpha a</body></html>"
                elif self.path.startswith("/alpha-b"):
                    body = b"<html><body>alpha b</body></html>"
                else:
                    body = b"<html><body>web material</body></html>"
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{server.server_port}"
        web, provenance = _retrieve({"kind": "web", "uri": f"{base_url}/"}, root)
        assert web.strip() == "web material" and provenance["resolved_uri"].startswith(base_url)
        json_results, _ = _discover({"kind": "web_search", "provider": "test-json",
                                     "search_endpoint": f"{base_url}/search-json"}, "fixture")
        assert json_results[0]["url"] == f"{base_url}/json-page"

        (root / "repo").mkdir()
        (root / "repo/context.txt").write_text("beta\n", encoding="utf-8")
        plan = root / "plan.json"
        write_json(plan, _plan(f"{base_url}/search"))
        expanded_plan = root / "expanded-plan.json"

        def expand_one(messages, tools):
            assert not tools and "search_results" in messages[0]["content"]
            return {"content": json.dumps({"atomic": False, "children": [{
                "name": "checksum workflow", "atomic": True, "related_to": ["report"]}]})}

        expansion = expand_knowledge_graph(plan, expanded_plan, complete=expand_one,
                                            max_nodes=3, max_depth=1, children_per_node=2)
        assert expansion["nodes"] == 3 and expansion["expanded"] == 1
        expanded = read_json(expanded_plan)
        child = expanded["domains"][0]["subdomains"][0]["concepts"][-1]
        assert child["parent_ids"] and child["related_ids"]

        report = prepare(plan, root / "run", profile="diverse", seed=7, count=None,
                         path_policy="agentworld", tasks_per_material=2, trajectories_per_task=2)
        assert report["stage1"]["realized_distribution"] == {"artifact.create": 1, "coding.transform": 1}
        assert report["stage1"]["unique_domains"] == report["stage1"]["unique_subdomains"] == 2
        assert report["stage1"]["search_queries"] == 1 and report["stage1"]["discovered_urls"] == 2
        assert report["counts_before_verifier"] == {
            "stage1_materials": 2, "stage1_unique_materials": 2,
            "stage2_candidate_tasks": 4,
            "stage3_candidate_trajectories": 8, "gross_artifacts": 14,
            "verifier_filtered_counts_excluded": True}
        assert report["stage2"]["unique_task_signatures"] == 4
        assert report["stage3"]["validated"] == 4 and report["stage3"]["rejected"] == 0
        assert report["stage3"]["candidate_trajectories"] == 8
        assert (root / "run/stage1/domain_distribution.png").is_file()
        assert (root / "run/stage1/domain_distribution.pdf").is_file()
        assert (root / "run/REPORT.md").is_file()
        assert report["stage4"]["sft_status"].startswith("awaiting")

        def synthesize_one(_messages, tools):
            assert not tools
            return {"content": json.dumps({"tasks": [{
                "id": "generated", "task_type": "fixture", "capability": "create",
                "interface": "workspace", "prompt": "Create done.txt containing OK.",
                "files": {"context.txt": "{{material}}"}, "available_tools": ["apply_patch"],
                "verifier": "test \"$(cat done.txt)\" = OK", "target_facts": ["file:done.txt:exists"],
                "actions": [{"id": "write", "tool": "apply_patch", "arguments": {
                    "patch": "*** Begin Patch\n*** Add File: done.txt\n+OK\n*** End Patch"},
                    "preconditions": ["workspace:ready"], "effects": ["file:done.txt:exists"]}],
                "min_steps": 1, "max_steps": 1}]})}

        generated_materials = root / "run/stage1/generated_materials.jsonl"
        generated = synthesize_environment_templates(
            root / "run/stage1/materials.jsonl", generated_materials, complete=synthesize_one,
            variants_per_material=1, model="scripted-synthesis")
        generated_tasks = construct_tasks(generated_materials, root / "generated/tasks.jsonl")
        assert generated["materials_generated"] == generated_tasks["tasks"] == 2
        assert all(task["sandbox_backend"] == "bwrap"
                   for task in read_jsonl(root / "generated/tasks.jsonl"))

        taskless = _plan(f"{base_url}/search")
        for domain in taskless["domains"]:
            for subdomain in domain["subdomains"]:
                for concept in subdomain["concepts"]:
                    concept.pop("task")
        taskless["profiles"]["single"] = {"count": 1, "distribution": {"coding": 1}}
        taskless_plan = root / "taskless-plan.json"
        write_json(taskless_plan, taskless)
        generated_run = prepare(
            taskless_plan, root / "run-generated", profile="single", seed=7, count=None,
            path_policy="agentworld", tasks_per_material=1, trajectories_per_task=1,
            environment_complete=synthesize_one, synthesis_model="scripted-synthesis")
        assert generated_run["counts_before_verifier"]["stage3_candidate_trajectories"] == 1
        assert generated_run["stage2"]["environment_synthesis"]["materials_generated"] == 1

        control = prepare(plan, root / "run-narrow", profile="narrow", seed=7, count=None,
                          path_policy="uniform")
        assert control["stage1"]["realized_distribution"] == {"coding.transform": 2}
        assert control["stage3"]["validated"] == 2 and control["path_policy"] == "uniform"
        narrow_materials = read_jsonl(root / "run-narrow/stage1/materials.jsonl")
        assert len({row["provenance"]["discovered_url"] for row in narrow_materials}) == 2
        assert not any(row["provenance"]["reused_candidate"] for row in narrow_materials)

        tasks = read_jsonl(root / "run/stage3/validated_tasks.jsonl")
        episodes = read_jsonl(root / "run/stage3/oracle_episodes.jsonl")
        assert all(task["sandbox_backend"] == "bwrap" for task in tasks)
        assert all(row["provenance"].get("discovery_id") for row in
                   read_jsonl(root / "run/stage1/materials.jsonl") if row["provenance"]["kind"] == "web_search")
        assert all(episode["execution_mode"] == "bwrap" for episode in episodes)
        assert all(episode["outcome"]["call_result_linkage_complete"]
                   and episode["outcome"]["trace_fidelity"] for episode in episodes)
        from model.language_model.scripts.prepare_verl_tasks import task_rows
        verl_rows = list(task_rows(root / "run/stage4/rl_tasks.jsonl"))
        assert all(set(row["extra_info"]["tools_kwargs"]) == set(task["available_tools"])
                   for row, task in zip(verl_rows, tasks, strict=True))
        assert all(set(row["extra_info"]["tool_selection"]) == set(task["available_tools"])
                   for row, task in zip(verl_rows, tasks, strict=True))
        assert all(row["extra_info"]["sandbox_backend"] == "bwrap" for row in verl_rows)
        legacy = list(task_rows(Path("agent/tasks/synthesis_seed.jsonl"), limit=1))[0]
        assert set(legacy["extra_info"]["tool_selection"]) == {"exec_command", "apply_patch"}
        assert legacy["extra_info"]["sandbox_backend"] == "workspace_host"
        by_task = {episode["task_id"]: episode for episode in episodes}
        passed_task = tasks[0]
        failed_task = next(task for task in tasks if task["task_family"] != passed_task["task_family"])
        oracle_result = finalize(root / "run", [root / "run/stage3/oracle_episodes.jsonl"],
                                 policy_kind="teacher", model="test-teacher")
        assert oracle_result["accepted_sft"] == 0 and oracle_result["rejected"] == 8

        def scripted(messages, _tools):
            if messages[-1]["role"] == "tool":
                return {"content": "DONE"}
            prompt = next(message["content"] for message in messages if message["role"] == "user")
            output = re.search(r"answer-\d+\.txt", prompt)
            if output:
                name, arguments = "exec_command", {
                    "cmd": f"test ! -e /home && set -- $(sha256sum input.txt) && printf '%s\\n' \"$1\" > {output.group()}"}
            else:
                output = re.search(r"report-\d+\.txt", prompt)
                name, arguments = "apply_patch", {"patch":
                    f"*** Begin Patch\n*** Add File: {output.group()}\n+BETA\n*** End Patch"}
            return {"tool_calls": [{"id": "policy_call", "type": "function",
                                    "function": {"name": name, "arguments": json.dumps(arguments)}}]}

        policy_rollouts = root / "policy-rollouts.jsonl"
        policy_report = run_policy_tasks(root / "run/stage3/validated_tasks.jsonl", policy_rollouts,
                                         complete=scripted, policy_kind="teacher", model="scripted-policy")
        assert policy_report["passed"] == 4
        for row in read_jsonl(policy_rollouts):
            calls = {event["tool_call_id"] for event in row["agent_events"] if event["kind"] == "tool_call"}
            results = {event["tool_call_id"] for event in row["agent_events"] if event["kind"] == "tool_result"}
            assert calls == results and calls
            assert all(event.get("state_before", {}).get("state_hash") for event in row["agent_events"] if event["kind"] == "tool_call")
            assert all(event.get("state_after", {}).get("state_hash") for event in row["agent_events"] if event["kind"] == "tool_result")
        assert finalize(root / "run", [policy_rollouts], policy_kind="teacher",
                        model="scripted-policy")["accepted_sft"] == 4

        serialized_action = by_task[passed_task["task_id"]]["actions"][0]
        serialized = ("<tool_call>\n" + json.dumps({"name": serialized_action["tool"],
                      "arguments": serialized_action["arguments"]}) +
                      "\n</tool_call>\nuser\n<tool_response>\nexit_code=0\nok\n</tool_response>\nassistant\nDONE")
        rollouts = root / "rollouts.jsonl"
        write_jsonl(rollouts, [
            {"task_id": passed_task["task_id"], "policy": {"kind": "teacher"},
             "messages": by_task[passed_task["task_id"]]["messages"],
             "outcome": {"task_success": 1, "harness_status": "healthy", "eligible": True}},
            {"task_id": failed_task["task_id"], "policy": {"kind": "teacher"},
             "messages": by_task[failed_task["task_id"]]["messages"],
             "outcome": {"task_success": 0, "harness_status": "healthy", "eligible": True}},
            {"task_id": passed_task["task_id"], "policy": {"kind": "programmatic_oracle"},
             "messages": by_task[passed_task["task_id"]]["messages"],
             "outcome": {"task_success": 1, "harness_status": "healthy"}},
            {"task_id": passed_task["task_id"], "policy": {"kind": "teacher"}, "output": serialized,
             "gts": {"task_id": passed_task["task_id"],
                     "verifier_version": passed_task["verifier_version"]},
             "task_success": 1, "harness_status": "healthy", "eligible": True},
        ])
        result = finalize(root / "run", [rollouts], policy_kind="teacher", model="test-teacher")
        assert result["accepted_sft"] == 2 and result["rejected"] == 2
        assert read_jsonl(root / "run/stage4/sft.jsonl")[0]["metadata"]["model_provider"] == "test-teacher"

        weights = root / "weights.json"
        diagnosed = diagnose(root / "run", [rollouts], weights)
        assert diagnosed["distribution"][failed_task["task_family"]] > diagnosed["distribution"][passed_task["task_family"]]
        assert read_json(weights)["version"] == "diagnostic-distribution-v1"
        server.shutdown()
        thread.join()
        server.server_close()
    print("four-stage synthesis self-check passed")


if __name__ == "__main__":
    main()
