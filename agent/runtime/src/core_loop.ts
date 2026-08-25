import type { ChatMessage, Event, ToolSpec } from "../../shared/src/types.ts";
import { renderEvents } from "../../shared/src/renderer.ts";
import { eventFromAction, parseAction } from "./parser.ts";
import type { ToolContext, ToolResult } from "./tools/registry.ts";

export type ModelToolCall = {
  id?: string;
  function: { name: string; arguments: string };
};

export type ModelResponse = {
  content?: string;
  tool_calls?: ModelToolCall[];
};

export type Model = {
  complete(messages: ChatMessage[], tools: ToolSpec[]): Promise<ModelResponse>;
};

export type LoopPhase = "PENDING" | "GENERATING" | "PROCESSING_TOOLS" | "TERMINATED";
export type TerminationReason = "final" | "turn_limit";

export type LoopToolCall = {
  id: string;
  name: string;
  arguments: unknown;
};

export type CoreLoopOptions = {
  events: Event[];
  model: Model;
  tools: ToolSpec[];
  context: ToolContext;
  maxTurns: number;
  executeTool: (call: LoopToolCall, context: ToolContext) => Promise<ToolResult>;
  onCheckpoint?: (events: Event[]) => Promise<void>;
};

export type CoreLoopResult = {
  events: Event[];
  final?: string;
  turns: number;
  terminationReason: TerminationReason;
};

const parseArguments = (value: string): unknown => {
  try { return JSON.parse(value || "{}"); }
  catch { return value; }
};

const renderArguments = (value: unknown): string => typeof value === "string" ? value : JSON.stringify(value ?? {});

function completedTurns(events: Event[]): number {
  let turns = 0;
  for (let index = 0; index < events.length; index++) {
    const event = events[index];
    if (event.role !== "assistant") continue;
    if (event.kind === "message" || events[index - 1]?.role !== "assistant") turns++;
  }
  return turns;
}

function modelCalls(response: ModelResponse): ModelToolCall[] {
  if (response.tool_calls?.length) return response.tool_calls;
  const action = parseAction(response.content ?? "");
  return action.kind === "tool_call"
    ? [{ function: { name: action.tool, arguments: renderArguments(action.arguments) } }]
    : [];
}

function appendCalls(events: Event[], calls: ModelToolCall[], turn: number): LoopToolCall[] {
  const existingCalls = events.filter((event) => event.kind === "tool_call").length;
  return calls.map((call, index) => {
    const id = typeof call.id === "string" && call.id.trim()
      ? call.id
      : `call_${turn}_${existingCalls + index}`;
    const parsed = parseArguments(call.function.arguments);
    const event = eventFromAction({ kind: "tool_call", tool: call.function.name, arguments: parsed });
    event.toolCallId = id;
    events.push(event);
    return { id, name: call.function.name, arguments: parsed };
  });
}

/**
 * The framework-owned interaction loop.
 *
 * Session persistence, retries, subagents, and task verification deliberately
 * stay outside this function.  A Verl adapter can reuse this state-machine
 * contract while adding token-level rollout bookkeeping around `Model`.
 */
export async function runCoreLoop(options: CoreLoopOptions): Promise<CoreLoopResult> {
  const events = options.events;
  let phase: LoopPhase = "PENDING";
  let pendingCalls: LoopToolCall[] = [];
  let final: string | undefined;
  let terminationReason: TerminationReason = "turn_limit";
  let turns = completedTurns(events);

  while (phase !== "TERMINATED") {
    if (phase === "PENDING") {
      phase = "GENERATING";
      continue;
    }

    if (phase === "GENERATING") {
      if (turns >= options.maxTurns) {
        phase = "TERMINATED";
        continue;
      }
      turns++;
      const response = await options.model.complete(renderEvents(events), options.tools);
      const calls = modelCalls(response);
      if (calls.length) {
        pendingCalls = appendCalls(events, calls, turns);
        phase = "PROCESSING_TOOLS";
        continue;
      }

      const action = parseAction(response.content ?? "");
      events.push(eventFromAction(action));
      if (action.kind === "message") {
        final = action.content;
        terminationReason = "final";
        phase = "TERMINATED";
      }
      continue;
    }

    if (phase === "PROCESSING_TOOLS") {
      for (const call of pendingCalls) {
        try {
          const result = await options.executeTool(call, options.context);
          events.push({
            role: "tool",
            kind: "tool_result",
            toolCallId: call.id,
            content: result.content,
            toolResult: result.data,
            exitCode: result.exitCode,
          });
        } catch (error) {
          events.push({
            role: "tool",
            kind: "tool_result",
            toolCallId: call.id,
            content: String(error),
            exitCode: 1,
          });
        }
      }
      pendingCalls = [];
      await options.onCheckpoint?.(events);
      phase = "GENERATING";
      continue;
    }
  }

  return { events, final, turns, terminationReason };
}
