import type { Event } from "../../shared/src/types.ts";

export type Action = { kind: "message"; content: string } | { kind: "tool_call"; tool: string; arguments: Record<string, unknown> };

function objectText(text: string): string {
  const match = text.match(/```(?:json)?\s*([\s\S]*?)```/) ?? text.match(/\{[\s\S]*\}/);
  return match?.[1] ?? match?.[0] ?? text;
}

export function parseAction(text: string): Action {
  try {
    const value = JSON.parse(objectText(text));
    if (value.tool_call?.name) return { kind: "tool_call", tool: value.tool_call.name, arguments: value.tool_call.arguments ?? {} };
    if (value.name && value.arguments !== undefined) return { kind: "tool_call", tool: value.name, arguments: value.arguments };
    if (value.message !== undefined) return { kind: "message", content: String(value.message) };
  } catch { /* plain assistant text is a valid final answer */ }
  return { kind: "message", content: text };
}

export function eventFromAction(action: Action): Event {
  return action.kind === "tool_call"
    ? { role: "assistant", kind: "tool_call", tool: action.tool, arguments: action.arguments }
    : { role: "assistant", kind: "message", content: action.content };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  console.assert(parseAction('{"tool_call":{"name":"x","arguments":{"a":1}}}').kind === "tool_call");
}
