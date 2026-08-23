import type { Event } from "../../shared/src/types.ts";

export type Action = { kind: "message"; content: string } | { kind: "tool_call"; tool: string; arguments: unknown };

function objectText(text: string): string {
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/);
  if (fenced) return fenced[1];
  const start = text.indexOf("{");
  for (let end = text.lastIndexOf("}") + 1; start >= 0 && end > start; end = text.lastIndexOf("}", end - 2) + 1) {
    try { JSON.parse(text.slice(start, end)); return text.slice(start, end); }
    catch { /* tolerate trailing prose or a duplicated closing brace */ }
  }
  return text;
}

export function parseAction(text: string): Action {
  try {
    const value = JSON.parse(objectText(text));
    if (value.tool_call?.name) return { kind: "tool_call", tool: value.tool_call.name, arguments: value.tool_call.arguments ?? {} };
    if (Array.isArray(value.tool_calls) && value.tool_calls[0]) {
      const call = value.tool_calls[0].function ?? value.tool_calls[0];
      if (call.name) return { kind: "tool_call", tool: call.name, arguments: call.arguments ?? {} };
    }
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
