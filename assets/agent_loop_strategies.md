# Agent loop strategies figure

`agent_loop_strategies.svg`, `agent_loop_strategies.pdf`, and
`agent_loop_strategies.png` compare the common interaction states and runtime
boundaries of Verl `ToolAgentLoop`, Codex, and Prime.

The top strip is the shared semantic loop:

`PENDING → GENERATING → PROCESSING_TOOLS → (tool observation) → GENERATING`

with a termination path after a no-tool response or a stop/length policy. The
three lower panels distinguish the inner loop from outer runtime logic:

- Verl's outer `AgentLoopManager / Worker` owns rollout scheduling and exposes
  token-level `AgentLoopOutput` fields such as `response_mask` and `logprobs`.
- Codex's `SessionTask / Session` owns cancellation, persistence, compaction,
  queued input, hooks, and the streaming tool runtime around `run_turn`.
- Prime's `AgentSession` owns persistence, tool events, compaction, goals, and
  child lifecycles; autonomous/headless modes add continuation quality gates.

The footer marks the current nanoGPT `runner.ts` placement: it currently spans
the inner interaction loop, session state/persistence, and task verification.
