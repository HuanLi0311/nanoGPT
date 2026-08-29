"""Build the reproducible multi-domain synthesis manifest.

The manifest uses only the programmatic workspace tools already implemented by
the synthesis runner: exec_command and apply_patch. Verifiers check semantic
task results and accept candidate-indexed artifacts without requiring opaque
marker files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROMPTS = {
    "workspace_create_read": "Create answer.txt containing exactly ALPHA, then read it back.",
    "workspace_update_snapshot": "Change legacy.txt from legacy to modern and save a snapshot copy containing modern.",
    "workspace_directory_layout": "Create a source file containing 42 and copy it into the out directory as a value text file.",
    "workspace_transform_artifact": "Transform input.txt to uppercase and save the two-line result in an output text file.",
    "code_search_return": "Search src/app.py for the return statement and save the matching line in a search result text file.",
    "code_inspect_documentation": "Inspect README.md, confirm STATUS is READY, and save that READY result in an inspection file.",
    "code_inventory_files": "Create an inventory text file that lists a.txt and b.txt, one path per line.",
    "repair_python_syntax": "Fix the syntax error in buggy.py so python3 can compile it successfully.",
    "repair_unit_test": "Fix calc.add so add(2, 3) returns 5.",
    "repair_parser_regression": "Fix parse.parse so parse(' 7 ') returns the integer 7.",
    "config_json_validate": "Validate config.json: enabled must be true and items must contain three values. Save valid in a result file.",
    "config_csv_aggregate": "Sum the value column in data.csv and save exactly sum=5 in an output file.",
    "config_mode_migration": "Migrate settings.ini from mode=dev to mode=prod.",
    "data_sort_pipeline": "Sort numbers.txt numerically and save 1, 2, 3 on separate lines in a new text file.",
    "data_reproducible_report": "Read metrics.txt and save exactly pass in a report file when accuracy is at least 0.9.",
    "process_command_lifecycle": "Create and run a small command or script whose log contains a line exactly equal to done.",
    "process_log_review": "Count ERROR lines in service.log and save exactly 1 in a result file.",
    "artifact_svg_manifest": "Create an SVG file with an <svg element and width=\"10\".",
    "artifact_checksum_record": "Compute the SHA-256 of payload.txt and save the exact hexadecimal digest in a checksum text file.",
    "documentation_changelog": "Create docs/CHANGELOG.md with a meaningful entry describing a verified workspace update.",
    "documentation_manifest": "Create a manifest text file listing src/main.py and src/util.py, one path per line.",
}


def add_file(path: str, content: str, *, action_id: str, pre: list[str] | None = None, effects: list[str] | None = None) -> dict:
    body = "\n".join(f"+{line}" for line in content.rstrip("\n").splitlines())
    return {
        "id": action_id,
        "tool": "apply_patch",
        "arguments": {"patch": f"*** Begin Patch\n*** Add File: {path}\n{body}\n*** End Patch"},
        "preconditions": pre or ["workspace:ready"],
        "effects": effects or [f"file:{path}:exists"],
    }


def update_file(path: str, old: str, new: str, *, action_id: str, pre: list[str], effects: list[str]) -> dict:
    return {
        "id": action_id,
        "tool": "apply_patch",
        "arguments": {
            "patch": f"*** Begin Patch\n*** Update File: {path}\n@@\n-{old}\n+{new}\n*** End Patch"
        },
        "preconditions": pre,
        "effects": effects,
    }


def command(action_id: str, value: str, *, pre: list[str], effects: list[str]) -> dict:
    return {
        "id": action_id,
        "tool": "exec_command",
        "arguments_template": {"command": value},
        "preconditions": pre,
        "effects": effects,
    }


def task(
    task_id: str,
    domain: str,
    subdomain: str,
    concepts: list[str],
    files: dict[str, str],
    verifier: str,
    patterns: list[dict],
) -> dict:
    ids = [pattern["id"] for pattern in patterns]
    return {
        "task_id": task_id,
        "prompt": f"[{domain}/{subdomain}] {PROMPTS[task_id]}",
        "domain": domain,
        "subdomain": subdomain,
        "task_family": f"{domain}.{subdomain}",
        "concepts": concepts,
        "materials": [f"{subdomain}_workspace", "independent_verifier"],
        "files": files,
        "verifier": verifier,
        "verifier_version": "synthesis-semantic-v2",
        "harness_version": "workspace-tool-v1",
        "tool_schema_version": "workspace-tools-v1",
        "initial_facts": ["workspace:ready"],
        "target_facts": [],
        "seed_patterns": ids,
        "candidate_patterns": ids,
        "max_steps": len(patterns) + 2,
        "action_patterns": patterns,
    }


def tasks() -> list[dict]:
    return [
        task(
            "workspace_create_read",
            "workspace",
            "file_creation",
            ["file_creation", "readback", "audit"],
            {},
            "test \"$(cat answer.txt)\" = ALPHA",
            [
                add_file("answer.txt", "ALPHA", action_id="create_answer", effects=["file:answer.txt:exists", "file:answer.txt:content:ALPHA"]),
                command("audit", "printf 'candidate-{candidate_index}\\n' > audit-{candidate_index}.txt", pre=["file:answer.txt:exists"], effects=["file:audit-{candidate_index}:exists"]),
                command("readback", "cat answer.txt", pre=["file:answer.txt:exists"], effects=["observation:answer.txt"]),
            ],
        ),
        task(
            "workspace_update_snapshot",
            "workspace",
            "file_update",
            ["patch", "versioned_file", "snapshot"],
            {"legacy.txt": "legacy\n"},
            "test \"$(cat legacy.txt)\" = modern && find . -maxdepth 1 -type f -name 'snapshot*.txt' -exec grep -qx modern {} \\; -print -quit | grep -q .",
            [
                update_file("legacy.txt", "legacy", "modern", action_id="update_legacy", pre=["workspace:ready"], effects=["file:legacy.txt:content:modern"]),
                command("snapshot", "cp legacy.txt snapshot-{candidate_index}.txt", pre=["file:legacy.txt:content:modern"], effects=["file:snapshot-{candidate_index}:exists"]),
                command("check_snapshot", "test \"$(cat snapshot-{candidate_index}.txt)\" = modern", pre=["file:snapshot-{candidate_index}:exists"], effects=["observation:snapshot"]),
            ],
        ),
        task(
            "workspace_directory_layout",
            "workspace",
            "directory_layout",
            ["directory", "artifact_copy", "layout"],
            {},
            "find out -maxdepth 1 -type f -name 'value*.txt' -exec grep -qx 42 {} \\; -print -quit | grep -q .",
            [
                add_file("src/value.txt", "42", action_id="create_source", effects=["file:src/value.txt:exists"]),
                command("make_layout", "mkdir -p out && cp src/value.txt out/value-{candidate_index}.txt", pre=["file:src/value.txt:exists"], effects=["file:out/value-{candidate_index}:exists"]),
                command("verify_layout", "test -s out/value-{candidate_index}.txt && printf ok > layout-{candidate_index}.ok", pre=["file:out/value-{candidate_index}:exists"], effects=["file:layout-{candidate_index}:exists"]),
            ],
        ),
        task(
            "workspace_transform_artifact",
            "workspace",
            "artifact_transform",
            ["copy", "text_transform", "artifact"],
            {"input.txt": "alpha\nbeta\n"},
            "find . -maxdepth 1 -type f -name 'output*.txt' -exec sh -c 'printf \"ALPHA\\nBETA\\n\" | cmp -s - \"$1\"' _ {} \\; -print -quit | grep -q .",
            [
                command("uppercase", "tr '[:lower:]' '[:upper:]' < input.txt > output-{candidate_index}.txt", pre=["file:input.txt:exists"], effects=["file:output-{candidate_index}:exists"]),
                command("check_transform", "printf ok > transform-{candidate_index}.ok && printf 'ALPHA\\nBETA\\n' | cmp -s - output-{candidate_index}.txt", pre=["file:output-{candidate_index}:exists"], effects=["file:transform-{candidate_index}:exists"]),
            ],
        ),
        task(
            "code_search_return",
            "coding",
            "repository_search",
            ["search", "source_observation", "result_record"],
            {"src/app.py": "def value():\n    return 7\n"},
            "find . -maxdepth 1 -type f -name 'search*.txt' ! -name 'search-marker-*' -exec grep -Eq 'return +7' {} \\; -print -quit | grep -q .",
            [
                command("search_return", "grep -n 'return' src/app.py > search-{candidate_index}.txt", pre=["file:src/app.py:exists"], effects=["file:search-{candidate_index}:exists", "observation:return"]),
                command("record_search", "test -s search-{candidate_index}.txt && printf found > search-marker-{candidate_index}.txt", pre=["file:search-{candidate_index}:exists"], effects=["file:search-marker-{candidate_index}:exists"]),
            ],
        ),
        task(
            "code_inspect_documentation",
            "coding",
            "repository_inspection",
            ["inspection", "documentation", "observation"],
            {"README.md": "# Demo\nSTATUS: READY\n"},
            "find . -maxdepth 1 -type f -name 'inspect*' -exec grep -Eqi 'ready' {} \\; -print -quit | grep -q .",
            [
                command("inspect_status", "grep -q 'STATUS: READY' README.md && printf ready > inspect-{candidate_index}.ok", pre=["file:README.md:exists"], effects=["file:inspect-{candidate_index}:exists", "observation:status_ready"]),
                command("read_inspection", "cat inspect-{candidate_index}.ok", pre=["file:inspect-{candidate_index}:exists"], effects=["observation:inspection_marker"]),
            ],
        ),
        task(
            "code_inventory_files",
            "coding",
            "workspace_inventory",
            ["file_inventory", "sorting", "workspace_observation"],
            {"a.txt": "A\n", "b.txt": "B\n"},
            "find . -maxdepth 1 -type f -name 'inventory*.txt' -exec sh -c 'grep -qx a.txt \"$1\" && grep -qx b.txt \"$1\"' _ {} \\; -print -quit | grep -q .",
            [
                command("list_files", "find . -maxdepth 1 -type f -printf '%f\\n' | sort > inventory-{candidate_index}.txt", pre=["file:a.txt:exists", "file:b.txt:exists"], effects=["file:inventory-{candidate_index}:exists", "observation:file_list"]),
                command("check_inventory", "grep -qx a.txt inventory-{candidate_index}.txt && grep -qx b.txt inventory-{candidate_index}.txt && printf complete > inventory-{candidate_index}.ok", pre=["file:inventory-{candidate_index}:exists"], effects=["file:inventory-{candidate_index}.ok:exists"]),
            ],
        ),
        task(
            "repair_python_syntax",
            "coding",
            "syntax_repair",
            ["bug", "patch", "compile"],
            {"buggy.py": "def main()\n    return 1\n"},
            "python3 -m py_compile buggy.py",
            [
                update_file("buggy.py", "def main()", "def main():", action_id="fix_syntax", pre=["workspace:ready"], effects=["pattern:source_fixed"]),
                command("compile", "python3 -m py_compile buggy.py && printf pass > compile-{candidate_index}.ok", pre=["pattern:source_fixed"], effects=["file:compile-{candidate_index}:exists", "pattern:compile_passed"]),
            ],
        ),
        task(
            "repair_unit_test",
            "coding",
            "unit_test_repair",
            ["bug", "unit_test", "regression"],
            {"calc.py": "def add(a, b):\n    return a - b\n"},
            "python3 -c 'from calc import add; assert add(2, 3) == 5'",
            [
                update_file("calc.py", "    return a - b", "    return a + b", action_id="fix_add", pre=["workspace:ready"], effects=["pattern:source_fixed"]),
                add_file("test_calc.py", "import unittest\nfrom calc import add\n\nclass TestAdd(unittest.TestCase):\n    def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n", action_id="add_test", pre=["pattern:source_fixed"], effects=["file:test_calc.py:exists", "pattern:test_present"]),
                command("run_unit_test", "python3 -m unittest -q test_calc.py > unit-{candidate_index}.log 2>&1 && printf pass > unit-{candidate_index}.ok", pre=["pattern:test_present"], effects=["file:unit-{candidate_index}:exists", "pattern:tests_passed"]),
            ],
        ),
        task(
            "repair_parser_regression",
            "coding",
            "regression_test",
            ["parser", "regression", "verification"],
            {"parse.py": "def parse(value):\n    return value.strip()\n"},
            "python3 -c 'from parse import parse; result = parse(\" 7 \"); assert result == 7 and isinstance(result, int)'",
            [
                update_file("parse.py", "    return value.strip()", "    return int(value.strip())", action_id="fix_parser", pre=["workspace:ready"], effects=["pattern:parser_fixed"]),
                add_file("test_parse.py", "import unittest\nfrom parse import parse\n\nclass TestParse(unittest.TestCase):\n    def test_number(self):\n        self.assertEqual(parse(' 7 '), 7)\n", action_id="add_parser_test", pre=["pattern:parser_fixed"], effects=["pattern:parser_test_present"]),
                command("run_regression", "python3 -m unittest -q test_parse.py > regression-{candidate_index}.log 2>&1 && printf pass > regression-{candidate_index}.ok", pre=["pattern:parser_test_present"], effects=["pattern:regression_passed"]),
            ],
        ),
        task(
            "config_json_validate",
            "configuration",
            "json_validation",
            ["json", "schema", "validation"],
            {"config.json": '{"name":"demo","enabled":true,"items":[1,2,3]}\n'},
            "python3 -c 'import json; data=json.load(open(\"config.json\")); assert data[\"enabled\"] and len(data[\"items\"]) == 3' && find . -maxdepth 1 -type f ! -name config.json -exec grep -Eqi '^valid' {} \\; -print -quit | grep -q .",
            [
                command("validate_json", "python3 -c \"import json; d=json.load(open('config.json')); assert d['enabled'] and len(d['items']) == 3; open('json-{candidate_index}.ok','w').write('valid\\\\n')\"", pre=["file:config.json:exists"], effects=["file:json-{candidate_index}:exists", "pattern:json_valid"]),
                command("read_json_result", "cat json-{candidate_index}.ok", pre=["pattern:json_valid"], effects=["observation:json_valid"]),
            ],
        ),
        task(
            "config_csv_aggregate",
            "configuration",
            "csv_aggregation",
            ["csv", "aggregation", "shell_pipeline"],
            {"data.csv": "name,value\nA,2\nB,3\n"},
            "find . -maxdepth 1 -type f ! -name data.csv -exec grep -qx 'sum=5' {} \\; -print -quit | grep -q .",
            [
                command("aggregate_csv", "awk -F, 'NR > 1 {s += $2} END {if (s == 5) print \"sum=5\"}' data.csv > csv-{candidate_index}.out", pre=["file:data.csv:exists"], effects=["file:csv-{candidate_index}.out:exists", "pattern:csv_aggregated"]),
                command("validate_csv", "grep -qx 'sum=5' csv-{candidate_index}.out && printf valid > csv-{candidate_index}.ok", pre=["pattern:csv_aggregated"], effects=["file:csv-{candidate_index}.ok:exists"]),
            ],
        ),
        task(
            "config_mode_migration",
            "configuration",
            "config_migration",
            ["config", "migration", "patch"],
            {"settings.ini": "mode=dev\n"},
            "grep -qx 'mode=prod' settings.ini",
            [
                update_file("settings.ini", "mode=dev", "mode=prod", action_id="migrate_mode", pre=["workspace:ready"], effects=["pattern:config_migrated"]),
                command("check_mode", "grep -qx 'mode=prod' settings.ini && printf prod > mode-{candidate_index}.ok", pre=["pattern:config_migrated"], effects=["file:mode-{candidate_index}:exists"]),
            ],
        ),
        task(
            "data_sort_pipeline",
            "data",
            "deterministic_pipeline",
            ["sort", "pipeline", "determinism"],
            {"numbers.txt": "3\n1\n2\n"},
            "find . -maxdepth 1 -type f ! -name numbers.txt -exec sh -c 'printf \"1\\n2\\n3\\n\" | cmp -s - \"$1\"' _ {} \\; -print -quit | grep -q .",
            [
                command("sort_numbers", "sort -n numbers.txt > sorted-{candidate_index}.txt", pre=["file:numbers.txt:exists"], effects=["file:sorted-{candidate_index}:exists", "pattern:data_sorted"]),
                command("check_sorted", "test \"$(tr '\\n' ' ' < sorted-{candidate_index}.txt)\" = '1 2 3 ' && printf sorted > pipeline-{candidate_index}.ok", pre=["pattern:data_sorted"], effects=["file:pipeline-{candidate_index}:exists"]),
            ],
        ),
        task(
            "data_reproducible_report",
            "data",
            "report_generation",
            ["metrics", "report", "threshold"],
            {"metrics.txt": "accuracy=0.9\nloss=0.1\n"},
            "find . -maxdepth 1 -type f ! -name metrics.txt -exec grep -qx pass {} \\; -print -quit | grep -q .",
            [
                command("evaluate_metric", "awk -F= '/accuracy/ {if ($2 >= 0.9) print \"pass\"}' metrics.txt > report-{candidate_index}.txt", pre=["file:metrics.txt:exists"], effects=["file:report-{candidate_index}:exists", "pattern:metric_evaluated"]),
                command("approve_report", "grep -qx pass report-{candidate_index}.txt && printf approved > report-{candidate_index}.ok", pre=["pattern:metric_evaluated"], effects=["file:report-{candidate_index}.ok:exists"]),
            ],
        ),
        task(
            "process_command_lifecycle",
            "process",
            "command_lifecycle",
            ["process", "stdout", "completion"],
            {},
            "find . -maxdepth 1 -type f -name '*.log' -exec grep -qx done {} \\; -print -quit | grep -q .",
            [
                add_file("job.sh", "#!/bin/sh\nprintf 'start\\n'\nsleep 0.01\nprintf 'done\\n'\n", action_id="create_job", effects=["file:job.sh:exists"]),
                command("run_job", "chmod +x job.sh && ./job.sh > process-{candidate_index}.log", pre=["file:job.sh:exists"], effects=["file:process-{candidate_index}:exists", "pattern:process_finished"]),
                command("check_job", "grep -qx done process-{candidate_index}.log && printf complete > process-{candidate_index}.ok", pre=["pattern:process_finished"], effects=["file:process-{candidate_index}.ok:exists"]),
            ],
        ),
        task(
            "process_log_review",
            "process",
            "log_review",
            ["log", "error_count", "review"],
            {"service.log": "INFO boot\nERROR retry\nINFO ready\n"},
            "find . -maxdepth 1 -type f ! -name service.log -exec grep -qx 1 {} \\; -print -quit | grep -q .",
            [
                command("count_errors", "grep -c ERROR service.log > error-count-{candidate_index}.txt", pre=["file:service.log:exists"], effects=["file:error-count-{candidate_index}:exists", "pattern:log_counted"]),
                command("review_log", "test \"$(cat error-count-{candidate_index}.txt)\" = 1 && printf reviewed > log-{candidate_index}.ok", pre=["pattern:log_counted"], effects=["file:log-{candidate_index}:exists"]),
            ],
        ),
        task(
            "artifact_svg_manifest",
            "artifact",
            "structured_output",
            ["svg", "artifact", "manifest"],
            {},
            "find . -maxdepth 1 -type f -name '*.svg' -exec sh -c 'grep -q \"<svg\" \"$1\" && grep -q '\"'\"'width=\"10\"'\"'\"' \"$1\"' _ {} \\; -print -quit | grep -q .",
            [
                add_file("artifact-{candidate_index}.svg", "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"10\" height=\"10\"><rect width=\"10\" height=\"10\"/></svg>", action_id="create_svg", effects=["file:artifact-{candidate_index}.svg:exists"]),
                command("validate_svg", "grep -q '<svg' artifact-{candidate_index}.svg && grep -q 'width=\"10\"' artifact-{candidate_index}.svg && printf valid > artifact-{candidate_index}.ok", pre=["file:artifact-{candidate_index}.svg:exists"], effects=["file:artifact-{candidate_index}.ok:exists", "pattern:artifact_valid"]),
            ],
        ),
        task(
            "artifact_checksum_record",
            "artifact",
            "integrity_record",
            ["checksum", "integrity", "record"],
            {"payload.txt": "stable payload\n"},
            "expected=$(sha256sum payload.txt | awk '{print $1}'); for file in checksum*.txt; do test -f \"$file\" && test \"$(tr -d '[:space:]' < \"$file\")\" = \"$expected\" && exit 0; done; exit 1",
            [
                command("hash_payload", "sha256sum payload.txt | awk '{print $1}' > checksum-{candidate_index}.txt", pre=["file:payload.txt:exists"], effects=["file:checksum-{candidate_index}:exists", "pattern:checksum_recorded"]),
                command("validate_checksum", "test \"$(wc -c < checksum-{candidate_index}.txt)\" -gt 10 && printf recorded > checksum-{candidate_index}.ok", pre=["pattern:checksum_recorded"], effects=["file:checksum-{candidate_index}.ok:exists"]),
            ],
        ),
        task(
            "documentation_changelog",
            "documentation",
            "change_record",
            ["documentation", "changelog", "audit"],
            {"README.md": "# Demo\nUsage: run the tool\n"},
            "test -s docs/CHANGELOG.md && grep -Eqi 'verified|change|update|usage' docs/CHANGELOG.md",
            [
                add_file("docs/CHANGELOG.md", "## Run candidate-{candidate_index}\n- verified workspace update\n", action_id="add_changelog", effects=["file:docs/CHANGELOG.md:exists"]),
                command("check_docs", "grep -q 'Usage:' README.md && grep -q 'candidate-' docs/CHANGELOG.md && printf documented > docs-{candidate_index}.ok", pre=["file:docs/CHANGELOG.md:exists", "file:README.md:exists"], effects=["file:docs-{candidate_index}:exists", "pattern:docs_verified"]),
            ],
        ),
        task(
            "documentation_manifest",
            "documentation",
            "manifest_generation",
            ["documentation", "manifest", "inventory"],
            {"src/main.py": "print('hello')\n", "src/util.py": "VALUE = 1\n"},
            "find . -maxdepth 1 -type f -name 'manifest*.txt' -exec sh -c 'grep -qx src/main.py \"$1\" && grep -qx src/util.py \"$1\"' _ {} \\; -print -quit | grep -q .",
            [
                command("write_manifest", "find src -maxdepth 1 -type f -printf '%p\\n' | sort > manifest-{candidate_index}.txt", pre=["file:src/main.py:exists", "file:src/util.py:exists"], effects=["file:manifest-{candidate_index}:exists"]),
                command("check_manifest", "grep -qx src/main.py manifest-{candidate_index}.txt && grep -qx src/util.py manifest-{candidate_index}.txt && printf complete > manifest-{candidate_index}.ok", pre=["file:manifest-{candidate_index}:exists"], effects=["file:manifest-{candidate_index}.ok:exists"]),
            ],
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for item in tasks():
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({"output": str(args.output), "tasks": len(tasks())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
