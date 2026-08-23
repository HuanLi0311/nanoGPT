export type Role = "system" | "user" | "assistant" | "tool";

export type Event = {
  id?: string;
  role: Role;
  kind: "message" | "tool_call" | "tool_result";
  content?: string;
  tool?: string;
  arguments?: unknown;
  toolCallId?: string;
  exitCode?: number;
  toolResult?: unknown;
  timestamp?: string;
};

export type Trajectory = {
  trajectoryId: string;
  source: "codex";
  sessionId?: string;
  cwd?: string;
  events: Event[];
};

export type MessageContent = string | null | { type: "text"; text: string }[] | { type: "image_url"; image_url: { url: string; detail?: string } }[] | ({ type: "text"; text: string } | { type: "image_url"; image_url: { url: string; detail?: string } })[];

export type ChatMessage = {
  role: Role;
  content: MessageContent;
  tool_call_id?: string;
  tool_calls?: { id: string; type: "function"; function: { name: string; arguments: string } }[];
};

export type FunctionToolSpec = {
  type: "function";
  function: { name: string; description?: string; parameters: Record<string, unknown> };
};

export type CustomToolSpec = {
  type: "custom";
  name: string;
  description?: string;
  input?: Record<string, unknown>;
};

export type ToolSpec = FunctionToolSpec | CustomToolSpec;
