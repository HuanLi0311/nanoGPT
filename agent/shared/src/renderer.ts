import type { ChatMessage, Event, MessageContent, Trajectory } from "./types.ts";

const toolArguments = (value: unknown): string => typeof value === "string" ? value : JSON.stringify(value ?? {});
function toolContent(event: Event): MessageContent {
  if (!event.toolResult || typeof event.toolResult !== "object" || Array.isArray(event.toolResult)) {
    return event.toolResult === undefined ? (event.content ?? "") : JSON.stringify(event.toolResult);
  }
  const value = event.toolResult as Record<string, unknown>;
  if (typeof value.image_url !== "string") return JSON.stringify(value);
  const { image_url: url, detail, ...text } = value;
  return [
    { type: "text", text: JSON.stringify(text) },
    { type: "image_url", image_url: { url, ...(typeof detail === "string" ? { detail } : {}) } },
  ];
}

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
          function: { name: event.tool ?? "unknown", arguments: toolArguments(event.arguments) },
        });
      }
      messages.push({ role: "assistant", content: null, tool_calls });
      continue;
    }
    const event = events[index++];
    messages.push({
      role: event.role,
      content: toolContent(event),
      ...(event.kind === "tool_result" ? { tool_call_id: event.toolCallId ?? event.id } : {}),
    });
  }
  return messages;
}

export function renderMessages(trajectory: Trajectory): ChatMessage[] { return renderEvents(trajectory.events); }

export function renderSftRow(events: Event[]): { messages: ChatMessage[] } { return { messages: renderEvents(events) }; }
