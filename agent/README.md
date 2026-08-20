# NanoAgent harness

The harness uses Codex-like event semantics while keeping its own canonical
trajectory format. Extraction skips session metadata, developer instructions,
world state, and reasoning, while preserving structured tool calls/results.

```text
data/src/       Codex logs and dataset preparation
shared/src/     canonical protocol shared by data and runtime
runtime/src/    parser, runner, tools, reward
package/src/    Node CLI/TUI entrypoints and package configuration
```

Run extraction from `agent/package`:

```bash
cd agent/package
npm run extract -- ~/.codex/sessions ../data/canonical/codex-trajectories.jsonl
```

Run a DeepSeek-backed workspace episode (the key is read only from the
environment):

```bash
DEEPSEEK_API_KEY=... npm run run -- "inspect the repository and report its status"
```

Each episode is checkpointed atomically in `agent/runs/<run-id>.json`; set
`NANOAGENT_RUN_ID` to resume a specific episode. `exec_command` resolves its
working directory inside the workspace root and `apply_patch` accepts unified
git diffs.
