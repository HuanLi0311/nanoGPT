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
  timestamp?: string;
};

export type Trajectory = {
  trajectoryId: string;
  source: "codex";
  sessionId?: string;
  cwd?: string;
  events: Event[];
};

export type ChatMessage = {
  role: Role;
  content: string | null;
  tool_call_id?: string;
  tool_calls?: { id: string; type: "function"; function: { name: string; arguments: string } }[];
};

export type ToolSpec = {
  type: "function";
  function: { name: string; description?: string; parameters: Record<string, unknown> };
};
