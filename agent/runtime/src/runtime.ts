import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import type { Event, ToolSpec } from "../../shared/src/types.ts";
import { runCoreLoop, type Model, type TerminationReason } from "./core_loop.ts";
import { callTool, type ToolContext } from "./tools/registry.ts";

export type { Model } from "./core_loop.ts";
export type Verification = {
  score: number;
  passed: boolean;
  reason: string;
  harnessStatus: string;
  failureClass?: string;
  rewardSource?: string;
  eligible?: boolean;
};
export type RunState = { id: string; status: "running" | "done" | "failed"; attempt: number; events: Event[]; final?: string; verification?: Verification; terminationReason?: TerminationReason };
type Options = { id: string; prompt: string; model: Model; tools: ToolSpec[]; context: ToolContext; statePath: string; maxTurns?: number; retries?: number };

function verify(events: Event[], final: string): Verification {
  const calls = events.filter((event) => event.kind === "tool_call");
  const results = events.filter((event) => event.kind === "tool_result");
  const protocolError = events.find((event) => event.protocolStatus === "invalid");
  if (protocolError) return { score: 0, passed: false, reason: protocolError.failureClass ?? "invalid tool protocol", harnessStatus: "protocol", failureClass: protocolError.failureClass ?? "protocol" };
  if (!final.trim()) return { score: 0, passed: false, reason: "empty final answer", harnessStatus: "protocol", failureClass: "empty_final" };
  const callIds = calls.map((event) => event.toolCallId).filter((id): id is string => Boolean(id));
  const resultIds = results.map((event) => event.toolCallId).filter((id): id is string => Boolean(id));
  if (callIds.length !== calls.length || resultIds.length !== results.length) {
    return { score: 0, passed: false, reason: "tool call/result missing tool_call_id", harnessStatus: "protocol", failureClass: "protocol" };
  }
  if (new Set(callIds).size !== callIds.length) {
    return { score: 0, passed: false, reason: "duplicate tool_call_id", harnessStatus: "protocol", failureClass: "protocol" };
  }
  if (new Set(resultIds).size !== resultIds.length || resultIds.length !== callIds.length || resultIds.some((id) => !callIds.includes(id))) {
    return { score: 0, passed: false, reason: "tool result IDs do not match tool calls", harnessStatus: "protocol", failureClass: "protocol" };
  }
  const failures = results.filter((event) => (event.exitCode ?? 0) !== 0).length;
  if (failures) return { score: 0, passed: false, reason: `${failures} tool failure(s)`, harnessStatus: "tool_failure", failureClass: "tool_failure" };
  return { score: 0, passed: false, reason: "missing independent verifier", harnessStatus: "unscored", failureClass: "missing_verifier", eligible: false };
}

async function taskVerification(events: Event[], final: string, context: ToolContext): Promise<Verification> {
  const protocol = verify(events, final);
  if (protocol.harnessStatus === "protocol") return protocol;
  if (!context.verifyTask) return protocol;
  try {
    const outcome = await context.verifyTask();
    return {
      score: Math.max(0, Math.min(1, Number(outcome.score) || 0)),
      passed: Boolean(outcome.passed), reason: String(outcome.reason),
      harnessStatus: String(outcome.harnessStatus ?? "healthy"),
      failureClass: outcome.failureClass,
      rewardSource: outcome.rewardSource,
      eligible: outcome.eligible,
    };
  } catch (error) {
    return { score: 0, passed: false, reason: String(error), harnessStatus: "harness_fault" };
  }
}

async function save(path: string, state: RunState) {
  await mkdir(dirname(path), { recursive: true });
  const tmp = `${path}.tmp-${process.pid}`;
  await writeFile(tmp, JSON.stringify(state, null, 2));
  await rename(tmp, path);
}

export async function run(options: Options): Promise<RunState> {
  if (!options.context.spawnAgent) {
    type Child = { queue: string[]; promise: Promise<{ completed?: string; error?: string }>; interrupted: boolean; runs: number };
    const children = new Map<string, Child>();
    const childContext = (): ToolContext => ({
      ...options.context, state: undefined, spawnAgent: undefined, sendAgentMessage: undefined,
      interruptAgent: undefined, verifyTask: undefined,
    });
    const launch = async (taskName: string, child: Child, prompt: string) => {
      if (child.interrupted) return { error: "subagent interrupted" };
      const id = `${options.id}.${taskName}.${child.runs++}`;
      try {
        const childState = await run({
          ...options,
          id,
          prompt,
          context: childContext(),
          statePath: join(dirname(options.statePath), `${id}.json`),
        });
        return childState.final ? { completed: childState.final } : { error: childState.verification?.reason ?? "subagent returned no final message" };
      } catch (error) {
        return { error: String(error) };
      }
    };
    options.context.spawnAgent = async ({ taskName, message }) => {
      if (children.has(taskName)) return { error: `agent already exists: ${taskName}` };
      const child = { queue: [], interrupted: false, runs: 0, promise: Promise.resolve({} as { completed?: string; error?: string }) };
      children.set(taskName, child);
      child.promise = launch(taskName, child, message);
      return child.promise;
    };
    options.context.sendAgentMessage = async (taskName, message, trigger) => {
      const child = children.get(taskName);
      if (!child) throw new Error(`unknown agent: ${taskName}`);
      child.queue.push(message);
      if (trigger) {
        child.promise = child.promise.then(() => launch(taskName, child, child.queue.splice(0).join("\n")));
        const agent = options.context.state?.agents.get(`/root/${taskName}`);
        if (agent) {
          agent.status = "running";
          agent.promise = child.promise.then((result) => {
            agent.completed = result.completed;
            agent.error = result.error;
            agent.status = result.error ? "errored" : "completed";
          }).catch((error) => { agent.status = "errored"; agent.error = String(error); });
        }
      }
    };
    options.context.interruptAgent = async (taskName) => {
      const child = children.get(taskName);
      if (!child) throw new Error(`unknown agent: ${taskName}`);
      child.interrupted = true;
    };
  }
  let state: RunState = { id: options.id, status: "running", attempt: 0, events: [{ role: "user", kind: "message", content: options.prompt }] };
  try {
    const saved = JSON.parse(await readFile(options.statePath, "utf8"));
    if (!saved || saved.id !== options.id || !Array.isArray(saved.events) || saved.events[0]?.role !== "user" || saved.events[0].content !== options.prompt || !["running", "done", "failed"].includes(saved.status)) throw new Error("invalid or mismatched run state");
    state = saved;
  } catch (error: any) {
    if (error?.code !== "ENOENT") throw error;
    await save(options.statePath, state);
  }
  if (state.status === "done") return state;
  const maxTurns = options.maxTurns ?? 20;
  const retries = options.retries ?? 2;
  state.status = "running";
  const model: Model = {
    complete: async (messages, tools) => {
      for (let attempt = 0; ; attempt++) {
        try { return await options.model.complete(messages, tools); }
        catch (error) {
          state.attempt++;
          if (attempt >= retries) {
            state.status = "failed";
            await save(options.statePath, state);
            throw error;
          }
          await new Promise((resolve) => setTimeout(resolve, 250 * 2 ** attempt));
        }
      }
    },
  };
  const result = await runCoreLoop({
    events: state.events,
    model,
    tools: options.tools,
    context: options.context,
    maxTurns,
    executeTool: async (call, context) => callTool(call.name, call.arguments, context),
    onCheckpoint: async (events) => {
      state.events = events;
      await save(options.statePath, state);
    },
  });
  state.events = result.events;
  state.terminationReason = result.terminationReason;
  if (result.final !== undefined) {
    state.final = result.final;
    state.verification = await taskVerification(state.events, result.final, options.context);
    state.status = state.verification.passed ? "done" : "failed";
  } else {
    state.status = "failed";
    state.verification = await taskVerification(state.events, "", options.context);
    if (result.terminationReason === "turn_limit" && state.verification.reason === "empty final answer") {
      state.verification = { score: 0, passed: false, reason: "turn limit exceeded", harnessStatus: "protocol" };
    }
  }
  await save(options.statePath, state);
  return state;
}
