# Agent rollout pipes

`agent_loop_strategies.svg`, `agent_loop_strategies.pdf`, and
`agent_loop_strategies.png` show three independent implementation-specific
rollout pipes rather than one shared runtime.

Each lane has four stages:

`INPUT / PENDING → GENERATING → PROCESSING_TOOLS → OUTPUT / TERMINATED`

The large return arrow is the semantic rollout boundary. It means that the
observation or output at round `t` becomes the input/context for round `t+1`:

- Verl returns token-level rollout information and feeds tool-response tokens
  back into the next generation; its final `AgentLoopOutput` is consumed by
  the trainer.
- Codex feeds transcript updates, tool outputs, and queued follow-ups into the
  next `run_turn` context.
- Prime feeds session/tool events and autonomous/headless quality-gate results
  into the next prompt or continuation turn.

The upper band in each lane is deliberately outside the inner pipe: Verl's
manager/worker, Codex's session task, and Prime's session/continuation runtime
own scheduling and lifecycle behavior. The footer records the current
nanoGPT `runner.ts` placement and the proposed separation from session state
and task verification.
