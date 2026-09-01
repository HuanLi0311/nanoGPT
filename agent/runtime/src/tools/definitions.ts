import type { ToolSpec } from "../../../shared/src/types.ts";

type JsonObject = Record<string, unknown>;

const functionTool = (name: string, description: string, properties: JsonObject = {}, required: string[] = []): ToolSpec => ({
  type: "function",
  function: { name, description, parameters: { type: "object", properties, required, additionalProperties: false } },
});

const stringProperty = { type: "string" };
const numberProperty = { type: "number" };
const booleanProperty = { type: "boolean" };
const cursorProperty = { type: "string" };

const execCommand = functionTool("exec_command", "Run a shell command with the workspace as its working directory.", {
  cmd: stringProperty,
  workdir: stringProperty,
  yield_time_ms: numberProperty,
  max_output_tokens: numberProperty,
}, ["cmd"]);

const writeStdin = functionTool("write_stdin", "Write to or poll an existing exec_command session.", {
  session_id: numberProperty,
  chars: stringProperty,
  yield_time_ms: numberProperty,
  max_output_tokens: numberProperty,
}, ["session_id"]);

const applyPatch: ToolSpec = {
  type: "custom",
  name: "apply_patch",
  description: "Apply a Codex Begin Patch or unified git diff inside the workspace.",
  input: { type: "string" },
};

const viewImage = functionTool("view_image", "Read a local workspace image and return a data URL.", {
  path: stringProperty,
  detail: { type: "string", enum: ["high", "original"] },
}, ["path"]);

export const workspaceToolSchemas: ToolSpec[] = [execCommand, writeStdin, applyPatch, viewImage];

export const rlToolSchemas: ToolSpec[] = [execCommand, applyPatch];

export const harnessToolSchemas: ToolSpec[] = [
  functionTool("update_plan", "Update the task plan.", { explanation: stringProperty, plan: { type: "array", items: { type: "object" } } }, ["plan"]),
  functionTool("request_user_input", "Request one to three user questions and wait for a response.", { questions: { type: "array", items: { type: "object" } } }, ["questions"]),
  functionTool("spawn_agent", "Spawn a subagent for a named task.", { task_name: stringProperty, message: stringProperty, fork_turns: stringProperty }, ["task_name", "message"]),
  functionTool("list_agents", "List live subagents.", { path_prefix: stringProperty }),
  functionTool("send_message", "Queue a message for a subagent.", { target: stringProperty, message: stringProperty }, ["target", "message"]),
  functionTool("followup_task", "Send a follow-up task to a subagent.", { target: stringProperty, message: stringProperty }, ["target", "message"]),
  functionTool("wait_agent", "Wait for mailbox activity from a subagent.", { timeout_ms: numberProperty }),
  functionTool("interrupt_agent", "Interrupt a subagent.", { target: stringProperty }, ["target"]),
  functionTool("list_mcp_resources", "List resources provided by configured MCP servers.", { server: stringProperty, cursor: cursorProperty }),
  functionTool("list_mcp_resource_templates", "List templates provided by configured MCP servers.", { server: stringProperty, cursor: cursorProperty }),
  functionTool("get_goal", "Get the current thread goal.", {}),
  functionTool("create_goal", "Create a thread goal.", { objective: stringProperty, token_budget: { type: "integer" } }, ["objective"]),
  functionTool("update_goal", "Mark the current goal complete or blocked.", { status: { type: "string", enum: ["complete", "blocked"] } }, ["status"]),
  functionTool("shell_command", "Compatibility alias for the legacy shell command fields.", { command: stringProperty, workdir: stringProperty, timeout_ms: numberProperty }, ["command"]),
  { type: "custom", name: "exec", description: "Run a raw JavaScript tool-orchestration cell.", input: { type: "string" } },
  functionTool("wait", "Wait for or terminate an exec cell.", { cell_id: stringProperty, yield_time_ms: numberProperty, max_tokens: numberProperty, terminate: booleanProperty }, ["cell_id"]),
];

/** Full interactive Codex-compatible harness profile. */
export const toolSchemas: ToolSpec[] = [...workspaceToolSchemas, ...harnessToolSchemas];
