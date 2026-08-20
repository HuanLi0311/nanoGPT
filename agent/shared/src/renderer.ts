import type { ChatMessage, Event, Trajectory } from "./types.ts";

export function renderMessages(trajectory: Trajectory): ChatMessage[] {
  return trajectory.events.map((event): ChatMessage => {
    if (event.kind === "tool_call") {
      return { role: "assistant", content: null, tool_calls: [{
        id: event.toolCallId ?? event.id ?? "call_0", type: "function",
        function: { name: event.tool ?? "unknown", arguments: JSON.stringify(event.arguments ?? {}) },
      }] };
    }
    return {
      role: event.role,
      content: event.content ?? "",
      ...(event.kind === "tool_result" ? { tool_call_id: event.toolCallId ?? event.id } : {}),
    };
  });
}

export function renderSftRow(events: Event[]): { messages: ChatMessage[] } {
  return { messages: events.filter((event) => event.kind !== "tool_result").map((event) => ({
    role: event.role, content: event.content ?? null,
  })) };
}
