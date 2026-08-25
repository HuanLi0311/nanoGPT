import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import type { ChatMessage, Event, ToolSpec } from "../../shared/src/types.ts";
import { parseAction, eventFromAction } from "./parser.ts";
import { callTool, type ToolContext } from "./tools/registry.ts";
import { verify, type Verification } from "./reward.ts";
import { renderEvents } from "../../shared/src/renderer.ts";

export type Model = { complete(messages: ChatMessage[], tools: ToolSpec[]): Promise<{ content?: string; tool_calls?: { id?: string; function: { name: string; arguments: string } }[] }> };
export type RunState = { id: string; status: "running" | "done" | "failed"; attempt: number; events: Event[]; final?: string; verification?: Verification };
type Options = { id: string; prompt: string; model: Model; tools: ToolSpec[]; context: ToolContext; statePath: string; maxTurns?: number; retries?: number };
const parseArguments = (value: string): unknown => {
  try { return JSON.parse(value || "{}"); }
  catch { return value; }
};

const renderArguments = (value: unknown): string => typeof value === "string" ? value : JSON.stringify(value ?? {});

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
  const maxTurns = options.maxTurns ?? 20, retries = options.retries ?? 2;
  for (let turn = state.events.filter((x) => x.role === "assistant").length; turn < maxTurns; turn++) {
    const messages = renderEvents(state.events);
    let response;
    for (let attempt = 0; ; attempt++) {
      try { response = await options.model.complete(messages, options.tools); break; }
      catch (error) { state.attempt++; if (attempt >= retries) { state.status = "failed"; await save(options.statePath, state); throw error; } await new Promise((r) => setTimeout(r, 250 * 2 ** attempt)); }
    }
    const calls = response.tool_calls?.length
      ? response.tool_calls
      : (() => {
          const action = parseAction(response.content ?? "");
          return action.kind === "tool_call"
            ? [{ function: { name: action.tool, arguments: renderArguments(action.arguments) } }]
            : [];
        })();
    if (calls.length) {
      for (const call of calls) {
        const event = eventFromAction({ kind: "tool_call", tool: call.function.name, arguments: parseArguments(call.function.arguments) });
        event.toolCallId = typeof call.id === "string" && call.id.trim()
          ? call.id
          : `call_${turn}_${state.events.filter((item) => item.kind === "tool_call").length}`;
        state.events.push(event);
      }
      for (const event of state.events.slice(-calls.length)) {
        try {
          const result = await callTool(event.tool!, event.arguments, options.context);
          state.events.push({ role: "tool", kind: "tool_result", toolCallId: event.toolCallId, content: result.content, toolResult: result.data, exitCode: result.exitCode });
        }
        catch (error) { state.events.push({ role: "tool", kind: "tool_result", toolCallId: event.toolCallId, content: String(error), exitCode: 1 }); }
      }
    } else {
      const action = parseAction(response.content ?? ""); state.events.push(eventFromAction(action));
      if (action.kind === "message") {
        state.final = action.content;
        state.verification = await taskVerification(state.events, action.content, options.context);
        state.status = state.verification.passed ? "done" : "failed";
        await save(options.statePath, state);
        return state;
      }
    }
    await save(options.statePath, state);
  }
  state.status = "failed";
  state.verification = await taskVerification(state.events, "", options.context);
  if (state.verification.reason === "empty final answer") state.verification = { score: 0, passed: false, reason: "turn limit exceeded", harnessStatus: "protocol" };
  await save(options.statePath, state);
  return state;
}
