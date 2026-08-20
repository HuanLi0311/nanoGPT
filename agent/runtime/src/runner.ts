import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import type { ChatMessage, Event, ToolSpec } from "../../shared/src/types.ts";
import { parseAction, eventFromAction } from "./parser.ts";
import { callTool, type ToolContext } from "./tools/registry.ts";
import { verify, type Verification } from "./reward.ts";

export type Model = { complete(messages: ChatMessage[], tools: ToolSpec[]): Promise<{ content?: string; tool_calls?: { id?: string; function: { name: string; arguments: string } }[] }> };
export type RunState = { id: string; status: "running" | "done" | "failed"; attempt: number; events: Event[]; final?: string; verification?: Verification };
type Options = { id: string; prompt: string; model: Model; tools: ToolSpec[]; context: ToolContext; statePath: string; maxTurns?: number; retries?: number };
const parseArguments = (value: string): Record<string, unknown> => {
  try { const parsed = JSON.parse(value || "{}"); return parsed && typeof parsed === "object" ? parsed : {}; }
  catch { return { raw: value }; }
};

async function save(path: string, state: RunState) {
  await mkdir(dirname(path), { recursive: true });
  const tmp = `${path}.tmp-${process.pid}`;
  await writeFile(tmp, JSON.stringify(state, null, 2));
  await rename(tmp, path);
}

export async function run(options: Options): Promise<RunState> {
  let state: RunState = { id: options.id, status: "running", attempt: 0, events: [{ role: "user", kind: "message", content: options.prompt }] };
  try { state = JSON.parse(await readFile(options.statePath, "utf8")); } catch { /* first run */ }
  if (state.status === "done") return state;
  const maxTurns = options.maxTurns ?? 20, retries = options.retries ?? 2;
  for (let turn = state.events.filter((x) => x.role === "assistant").length; turn < maxTurns; turn++) {
    const messages: ChatMessage[] = state.events.map((event) => event.kind === "tool_call"
      ? { role: "assistant", content: null, tool_calls: [{ id: event.toolCallId ?? "call_0", type: "function", function: { name: event.tool!, arguments: JSON.stringify(event.arguments ?? {}) } }] }
      : { role: event.role, content: event.content ?? "", ...(event.kind === "tool_result" ? { tool_call_id: event.toolCallId } : {}) });
    let response;
    for (let attempt = 0; ; attempt++) {
      try { response = await options.model.complete(messages, options.tools); break; }
      catch (error) { state.attempt++; if (attempt >= retries) { state.status = "failed"; await save(options.statePath, state); throw error; } await new Promise((r) => setTimeout(r, 250 * 2 ** attempt)); }
    }
    if (response.tool_calls?.length) {
      for (const call of response.tool_calls) {
        const event = eventFromAction({ kind: "tool_call", tool: call.function.name, arguments: parseArguments(call.function.arguments) });
        event.toolCallId = call.id ?? `call_${turn}`; state.events.push(event);
        try { const result = await callTool(event.tool!, event.arguments, options.context); state.events.push({ role: "tool", kind: "tool_result", toolCallId: event.toolCallId, content: result.content, exitCode: result.exitCode }); }
        catch (error) { state.events.push({ role: "tool", kind: "tool_result", toolCallId: event.toolCallId, content: String(error), exitCode: 1 }); }
      }
    } else {
      const action = parseAction(response.content ?? ""); state.events.push(eventFromAction(action));
      if (action.kind === "message") { state.final = action.content; state.verification = verify(state.events, action.content); state.status = state.verification.passed ? "done" : "failed"; await save(options.statePath, state); return state; }
    }
    await save(options.statePath, state);
  }
  state.status = "failed"; state.verification = { score: 0, passed: false, reason: "turn limit exceeded" }; await save(options.statePath, state); return state;
}
