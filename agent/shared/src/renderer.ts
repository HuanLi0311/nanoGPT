import type { ChatMessage, Event, Trajectory } from "./types.ts";

export function renderEvents(events: Event[]): ChatMessage[] {
  const messages: ChatMessage[] = [];
  for (let index = 0; index < events.length;) {
    if (events[index].kind === "tool_call") {
      const tool_calls = [];
      while (events[index]?.kind === "tool_call") {
        const event = events[index++];
        tool_calls.push({
          id: event.toolCallId ?? event.id ?? `call_${tool_calls.length}`,
          type: "function" as const,
          function: { name: event.tool ?? "unknown", arguments: JSON.stringify(event.arguments ?? {}) },
        });
      }
      messages.push({ role: "assistant", content: null, tool_calls });
      continue;
    }
    const event = events[index++];
    messages.push({
      role: event.role,
      content: event.content ?? "",
      ...(event.kind === "tool_result" ? { tool_call_id: event.toolCallId ?? event.id } : {}),
    });
  }
  return messages;
}

export function renderMessages(trajectory: Trajectory): ChatMessage[] { return renderEvents(trajectory.events); }

export function renderSftRow(events: Event[]): { messages: ChatMessage[] } { return { messages: renderEvents(events) }; }
