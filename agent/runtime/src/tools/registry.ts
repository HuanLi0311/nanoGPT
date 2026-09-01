import { readFile } from "node:fs/promises";
import { Script, createContext } from "node:vm";
import type { ToolResult } from "../../../shared/src/types.ts";
import { workspacePath } from "../../../workspace/boundary.ts";
import { applyPatch as applyPatchFile } from "./apply-patch.ts";
import { ShellManager, type ExecCommandInput, type ExecResult, type WriteStdinInput } from "./exec-command.ts";
import { toolSchemas } from "./definitions.ts";

export { harnessToolSchemas, rlToolSchemas, toolSchemas, workspaceToolSchemas } from "./definitions.ts";
export type { ToolResult } from "../../../shared/src/types.ts";

type JsonObject = Record<string, unknown>;
type AgentStatus = "pending_init" | "running" | "interrupted" | "completed" | "errored";

export type McpResource = {
  server: string;
  uri: string;
  name?: string;
  description?: string;
  mime_type?: string;
  text?: string;
};

export type McpResourceTemplate = {
  server: string;
  uri_template: string;
  name?: string;
  description?: string;
  mime_type?: string;
};

export type SpawnAgentRequest = { taskName: string; message: string; forkTurns?: string };
export type SpawnAgentResponse = { completed?: string; error?: string };

export type ToolContext = {
  root: string;
  cwd?: string;
  goalThreadId?: string;
  state?: HarnessState;
  requestUserInput?: (questions: unknown[]) => Promise<unknown>;
  userResponses?: unknown[];
  mcpResources?: McpResource[];
  mcpResourceTemplates?: McpResourceTemplate[];
  spawnAgent?: (request: SpawnAgentRequest) => Promise<SpawnAgentResponse>;
  sendAgentMessage?: (taskName: string, message: string, trigger: boolean) => Promise<void>;
  interruptAgent?: (taskName: string) => Promise<void>;
  verifyTask?: () => Promise<{
    score: number;
    passed: boolean;
    reason: string;
    harnessStatus?: string;
    failureClass?: string;
    rewardSource?: string;
    eligible?: boolean;
  }>;
};

type Agent = {
  taskName: string;
  localName: string;
  status: AgentStatus;
  messages: string[];
  completed?: string;
  error?: string;
  promise?: Promise<void>;
};

type Goal = {
  threadId: string;
  objective: string;
  status: "active" | "complete" | "blocked";
  tokenBudget?: number;
  tokensUsed: number;
  timeUsedSeconds: number;
  createdAt: number;
  updatedAt: number;
};

type Cell = {
  id: string;
  output: string[];
  status: "running" | "completed" | "errored" | "terminated";
  exitCode?: number;
  promise: Promise<void>;
  cancelled: boolean;
  cursor: number;
  startedAt: number;
};

export type HarnessState = {
  shell: ShellManager;
  plan: unknown[];
  planExplanation?: string;
  goal?: Goal;
  agents: Map<string, Agent>;
  cells: Map<string, Cell>;
  memory: Map<string, unknown>;
  userResponses: unknown[];
  nextCellId: number;
};

const TOOL_NAMES = [
  "exec_command", "write_stdin", "apply_patch", "view_image", "update_plan", "request_user_input",
  "wait_agent", "send_message", "list_agents", "spawn_agent", "followup_task", "interrupt_agent",
  "list_mcp_resources", "list_mcp_resource_templates", "get_goal", "create_goal", "update_goal",
  "shell_command", "exec", "wait",
] as const;

const delay = (milliseconds: number) => new Promise<void>((resolveDelay) => setTimeout(resolveDelay, Math.max(0, milliseconds)));

function stateFor(context: ToolContext): HarnessState {
  if (!context.state) {
    context.state = {
      shell: new ShellManager(), plan: [], agents: new Map(), cells: new Map(), memory: new Map(),
      userResponses: [...(context.userResponses ?? [])], nextCellId: 1,
    };
  }
  return context.state;
}

function object(value: unknown, name: string): JsonObject {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`${name} requires a JSON object`);
  return value as JsonObject;
}

function string(value: unknown, field: string, required = true): string | undefined {
  if (value === undefined || value === null) {
    if (required) throw new Error(`${field} is required`);
    return undefined;
  }
  if (typeof value !== "string") throw new Error(`${field} must be a string`);
  if (required && !value.trim()) throw new Error(`${field} must not be empty`);
  return value;
}

function number(value: unknown, field: string, required = false): number | undefined {
  if (value === undefined || value === null) {
    if (required) throw new Error(`${field} is required`);
    return undefined;
  }
  if (typeof value !== "number" || !Number.isFinite(value)) throw new Error(`${field} must be a finite number`);
  return value;
}

function boolean(value: unknown, field: string): boolean | undefined {
  if (value === undefined || value === null) return undefined;
  if (typeof value !== "boolean") throw new Error(`${field} must be a boolean`);
  return value;
}

function response(data: unknown, exitCode = 0, content = JSON.stringify(data)): ToolResult {
  return { content, exitCode, data };
}

function parseText(value: unknown): string {
  return typeof value === "string" ? value : JSON.stringify(value ?? null);
}

function outputLimit(value: unknown): number | undefined {
  const parsed = number(value, "max_tokens");
  return parsed && parsed > 0 ? Math.floor(parsed) * 4 : undefined;
}

function truncate(value: string, limit?: number): string {
  return limit && value.length > limit ? value.slice(-limit) : value;
}

function patchSummary(patch: string): string {
  const codex = [...patch.matchAll(/^\*\*\* (Add|Update|Delete) File: (.+)$/gm)]
    .map((match) => `${match[1] === "Add" ? "A" : match[1] === "Delete" ? "D" : "M"} ${match[2]}`);
  const unified = [...patch.matchAll(/^\+\+\+ b\/([^\t\n]+)/gm)].map((match) => `M ${match[1]}`);
  const changes = [...new Set([...codex, ...unified])];
  return `Success. Updated the following files:\n${changes.join("\n")}\n`;
}

function execOutput(result: ExecResult): string {
  return [
    ...(result.chunk_id ? [`Chunk ID: ${result.chunk_id}`] : []),
    `Wall time: ${result.wall_time_seconds.toFixed(3)} seconds`,
    result.exit_code === undefined ? `Process running with session ID ${result.session_id}` : `Process exited with code ${result.exit_code}`,
    ...(result.original_token_count === undefined ? [] : [`Original token count: ${result.original_token_count}`]),
    "Output:", result.output,
  ].join("\n");
}

function legacyShellOutput(result: ExecResult): string {
  return `Exit code: ${result.exit_code ?? 124}\nWall time: ${Math.round(result.wall_time_seconds)} seconds\nOutput:\n${result.output}`;
}

async function workspaceFile(root: string, value: string): Promise<string> {
  return workspacePath(root, value, true);
}

function imageMime(bytes: Buffer): string {
  if (bytes.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]))) return "image/png";
  if (bytes.subarray(0, 3).equals(Buffer.from([255, 216, 255]))) return "image/jpeg";
  if (bytes.subarray(0, 6).toString("ascii") === "GIF87a" || bytes.subarray(0, 6).toString("ascii") === "GIF89a") return "image/gif";
  if (bytes.subarray(8, 12).toString("ascii") === "WEBP") return "image/webp";
  throw new Error("unable to process image: invalid or unsupported image data");
}

function execInput(input: JsonObject): ExecCommandInput {
  return {
    cmd: string(input.cmd, "cmd")!,
    workdir: string(input.workdir, "workdir", false),
    tty: boolean(input.tty, "tty"),
    yield_time_ms: number(input.yield_time_ms, "yield_time_ms"),
    max_output_tokens: number(input.max_output_tokens, "max_output_tokens"),
    shell: string(input.shell, "shell", false),
    login: boolean(input.login, "login"),
  };
}

function writeInput(input: JsonObject): WriteStdinInput {
  return {
    session_id: number(input.session_id, "session_id", true)!,
    chars: string(input.chars, "chars", false),
    yield_time_ms: number(input.yield_time_ms, "yield_time_ms"),
    max_output_tokens: number(input.max_output_tokens, "max_output_tokens"),
  };
}

async function execCommand(input: unknown, context: ToolContext): Promise<ToolResult> {
  const result = await stateFor(context).shell.exec(execInput(object(input, "exec_command")), context.root, context.cwd);
  return response(result, result.exit_code ?? 0, execOutput(result));
}

async function writeStdin(input: unknown, context: ToolContext): Promise<ToolResult> {
  const result = await stateFor(context).shell.write(writeInput(object(input, "write_stdin")));
  return response(result, result.exit_code ?? 0, execOutput(result));
}

async function applyPatchTool(input: unknown, context: ToolContext): Promise<ToolResult> {
  const patch = typeof input === "string" ? input : string(object(input, "apply_patch").patch, "patch")!;
  const startedAt = Date.now();
  await applyPatchText(patch, context.root);
  return response({ output: patchSummary(patch), metadata: { exit_code: 0, duration_seconds: (Date.now() - startedAt) / 1_000 } });
}

async function applyPatchText(patch: string, root: string): Promise<string> {
  return applyPatchFile(patch, root);
}

async function viewImage(input: unknown, context: ToolContext): Promise<ToolResult> {
  const args = object(input, "view_image");
  const path = string(args.path, "path")!;
  const detail = string(args.detail, "detail", false) ?? "high";
  if (detail !== "high" && detail !== "original") throw new Error("detail must be high or original");
  const bytes = await readFile(await workspaceFile(context.root, path));
  return response({ detail, image_url: `data:${imageMime(bytes)};base64,${bytes.toString("base64")}` });
}

async function updatePlan(input: unknown, context: ToolContext): Promise<ToolResult> {
  const args = object(input, "update_plan");
  if (!Array.isArray(args.plan) || !args.plan.length) throw new Error("plan must be a non-empty array");
  let active = 0;
  for (const item of args.plan) {
    const step = object(item, "plan item");
    string(step.step, "plan[].step");
    const status = string(step.status, "plan[].status")!;
    if (!["pending", "in_progress", "completed"].includes(status)) throw new Error("plan[].status is invalid");
    if (status === "in_progress") active++;
  }
  if (active > 1) throw new Error("at most one plan item may be in_progress");
  const state = stateFor(context);
  state.plan = args.plan;
  state.planExplanation = string(args.explanation, "explanation", false);
  return response({}, 0, "Plan updated");
}

async function requestUserInput(input: unknown, context: ToolContext): Promise<ToolResult> {
  const questions = object(input, "request_user_input").questions;
  if (!Array.isArray(questions) || !questions.length || questions.length > 3) throw new Error("questions must contain one to three entries");
  for (const question of questions) {
    const item = object(question, "question");
    string(item.id, "questions[].id"); string(item.header, "questions[].header"); string(item.question, "questions[].question");
    if (!Array.isArray(item.options) || item.options.length < 2 || item.options.length > 3) throw new Error("questions[].options must contain two to three entries");
    for (const option of item.options) {
      const value = object(option, "question option");
      string(value.label, "questions[].options[].label");
      string(value.description, "questions[].options[].description");
    }
  }
  const state = stateFor(context);
  const answer = context.requestUserInput
    ? await context.requestUserInput(questions)
    : state.userResponses.shift();
  if (answer === undefined) throw new Error("request_user_input requires a configured user-response provider");
  return response({ answers: answer });
}

function canonicalTaskName(taskName: string): string {
  return taskName === "/root" || taskName.startsWith("/root/") ? taskName : `/root/${taskName.replace(/^\/+/, "")}`;
}

function agentFor(state: HarnessState, target: string): Agent {
  const agent = state.agents.get(canonicalTaskName(target));
  if (!agent) throw new Error(`unknown agent: ${target}`);
  return agent;
}

function agentStatus(agent: Agent): string | JsonObject {
  if (agent.status === "completed") return { completed: agent.completed ?? null };
  if (agent.status === "errored") return { errored: agent.error ?? "agent failed" };
  return agent.status;
}

function agentView(agent: Agent): JsonObject {
  return {
    agent_name: agent.taskName,
    agent_status: agentStatus(agent),
    last_task_message: agent.messages.at(-1) ?? null,
  };
}

async function spawnAgent(input: unknown, context: ToolContext): Promise<ToolResult> {
  const args = object(input, "spawn_agent");
  const taskName = string(args.task_name, "task_name")!;
  const message = string(args.message, "message")!;
  if (!/^[a-z0-9_]+$/.test(taskName)) throw new Error("task_name must use lowercase letters, digits, and underscores");
  if (!context.spawnAgent) throw new Error("spawn_agent requires a configured subagent runner");
  const state = stateFor(context);
  const canonicalName = canonicalTaskName(taskName);
  if (state.agents.has(canonicalName)) throw new Error(`agent already exists: ${taskName}`);
  const agent: Agent = { taskName: canonicalName, localName: taskName, status: "pending_init", messages: [message] };
  state.agents.set(canonicalName, agent);
  agent.promise = Promise.resolve(context.spawnAgent({ taskName, message, forkTurns: string(args.fork_turns, "fork_turns", false) }))
    .then((result) => {
      agent.completed = result.completed;
      agent.error = result.error;
      agent.status = result.error ? "errored" : "completed";
    })
    .catch((error) => { agent.status = "errored"; agent.error = String(error); });
  agent.status = "running";
  return response({ task_name: canonicalName });
}

async function listAgents(input: unknown, context: ToolContext): Promise<ToolResult> {
  const args = object(input ?? {}, "list_agents");
  const prefix = string(args.path_prefix, "path_prefix", false);
  const agents = [
    { agent_name: "/root", agent_status: "running", last_task_message: "Main thread" },
    ...[...stateFor(context).agents.values()].map(agentView),
  ].filter((agent) => !prefix || agent.agent_name.startsWith(prefix));
  return response({ agents });
}

async function sendAgentMessage(input: unknown, context: ToolContext, trigger: boolean): Promise<ToolResult> {
  const args = object(input, trigger ? "followup_task" : "send_message");
  const target = string(args.target, "target")!;
  const message = string(args.message, "message")!;
  const agent = agentFor(stateFor(context), target);
  agent.messages.push(message);
  if (context.sendAgentMessage) await context.sendAgentMessage(agent.localName, message, trigger);
  return response({}, 0, "");
}

async function waitAgent(input: unknown, context: ToolContext): Promise<ToolResult> {
  const args = object(input ?? {}, "wait_agent");
  if (Object.keys(args).some((key) => key !== "timeout_ms")) throw new Error("wait_agent accepts only timeout_ms");
  const requested = number(args.timeout_ms, "timeout_ms");
  if (requested !== undefined && requested < 0) throw new Error("timeout_ms must be non-negative");
  const timeout = Math.max(10_000, requested ?? 30_000);
  const agents = [...stateFor(context).agents.values()];
  const completed = await Promise.race([
    ...(agents.length ? [Promise.all(agents.map((agent) => agent.promise ?? Promise.resolve())).then(() => true)] : []),
    delay(timeout).then(() => false),
  ]);
  const clamped = requested !== undefined && requested < timeout
    ? `\n\nRequested timeout of ${requested}ms was clamped to the minimum of ${timeout}ms.` : "";
  return response({ message: completed ? `Wait completed.${clamped}` : `Wait timed out.${clamped}`, timed_out: !completed });
}

async function interruptAgent(input: unknown, context: ToolContext): Promise<ToolResult> {
  const target = string(object(input, "interrupt_agent").target, "target")!;
  const agent = agentFor(stateFor(context), target);
  const previousStatus = agentStatus(agent);
  if (context.interruptAgent) await context.interruptAgent(agent.localName);
  if (agent.status === "running" || agent.status === "pending_init") agent.status = "interrupted";
  return response({ previous_status: previousStatus });
}

function selectResources<T extends { server: string }>(items: T[], args: JsonObject): T[] {
  const server = string(args.server, "server", false);
  return server ? items.filter((item) => item.server === server) : items;
}

async function listMcpResources(input: unknown, context: ToolContext): Promise<ToolResult> {
  const args = object(input ?? {}, "list_mcp_resources");
  return response({ resources: selectResources(context.mcpResources ?? [], args).map((resource) => ({
    server: resource.server, uri: resource.uri, ...(resource.name ? { name: resource.name } : {}),
    ...(resource.description ? { description: resource.description } : {}), ...(resource.mime_type ? { mimeType: resource.mime_type } : {}),
    ...(resource.text ? { text: resource.text } : {}),
  })) });
}

async function listMcpResourceTemplates(input: unknown, context: ToolContext): Promise<ToolResult> {
  const args = object(input ?? {}, "list_mcp_resource_templates");
  return response({ resourceTemplates: selectResources(context.mcpResourceTemplates ?? [], args).map((resource) => ({
    server: resource.server, uriTemplate: resource.uri_template, ...(resource.name ? { name: resource.name } : {}),
    ...(resource.description ? { description: resource.description } : {}), ...(resource.mime_type ? { mimeType: resource.mime_type } : {}),
  })) });
}

function goalView(goal: Goal | undefined): JsonObject {
  const remainingTokens = goal?.tokenBudget === undefined ? null : Math.max(0, goal.tokenBudget - goal.tokensUsed);
  const completionBudgetReport = goal?.status === "complete" && (goal.tokenBudget !== undefined || goal.timeUsedSeconds > 0)
    ? "Goal achieved. Report final usage from this tool result's structured goal fields. If `goal.tokenBudget` is present, include token usage from `goal.tokensUsed` and `goal.tokenBudget`. If `goal.timeUsedSeconds` is greater than 0, summarize elapsed time in a concise, human-friendly form appropriate to the response language."
    : null;
  return { goal: goal ?? null, remainingTokens, completionBudgetReport };
}

async function getGoal(input: unknown, context: ToolContext): Promise<ToolResult> {
  object(input ?? {}, "get_goal");
  return response(goalView(stateFor(context).goal));
}

async function createGoal(input: unknown, context: ToolContext): Promise<ToolResult> {
  const args = object(input, "create_goal");
  const objective = string(args.objective, "objective")!.trim();
  const tokenBudget = number(args.token_budget, "token_budget");
  if (tokenBudget !== undefined && (!Number.isInteger(tokenBudget) || tokenBudget <= 0)) throw new Error("token_budget must be a positive integer");
  const state = stateFor(context);
  if (state.goal?.status === "active") throw new Error("cannot create a new goal while an unfinished goal exists");
  const now = Math.floor(Date.now() / 1_000);
  state.goal = { threadId: context.goalThreadId ?? "root", objective, status: "active", ...(tokenBudget === undefined ? {} : { tokenBudget }), tokensUsed: 0, timeUsedSeconds: 0, createdAt: now, updatedAt: now };
  return response(goalView(state.goal));
}

async function updateGoal(input: unknown, context: ToolContext): Promise<ToolResult> {
  const status = string(object(input, "update_goal").status, "status")!;
  if (status !== "complete" && status !== "blocked") throw new Error("update_goal status must be complete or blocked");
  const state = stateFor(context);
  if (!state.goal) throw new Error("cannot update goal because no goal exists");
  state.goal.status = status;
  state.goal.updatedAt = Math.floor(Date.now() / 1_000);
  state.goal.timeUsedSeconds = Math.max(0, state.goal.updatedAt - state.goal.createdAt);
  return response(goalView(state.goal));
}

async function shellCommand(input: unknown, context: ToolContext): Promise<ToolResult> {
  const args = object(input, "shell_command");
  const timeout = number(args.timeout_ms, "timeout_ms") ?? 30_000;
  let result = await stateFor(context).shell.exec({
    cmd: string(args.command, "command")!, workdir: string(args.workdir, "workdir", false),
    yield_time_ms: Math.min(timeout, 30_000), login: boolean(args.login, "login"),
  }, context.root, context.cwd);
  let output = result.output;
  const deadline = Date.now() + timeout;
  while (result.session_id !== undefined && Date.now() < deadline) {
    result = await stateFor(context).shell.write({ session_id: result.session_id, yield_time_ms: Math.min(30_000, deadline - Date.now()) });
    output += result.output;
  }
  if (result.session_id !== undefined) stateFor(context).shell.terminate(result.session_id);
  const value = { ...result, output } as ExecResult;
  return response(value, value.exit_code ?? 124, legacyShellOutput(value));
}

function cellResult(cell: Cell, maxTokens?: number): JsonObject {
  const allOutput = cell.output.join("");
  const output = truncate(allOutput.slice(cell.cursor), maxTokens);
  cell.cursor = allOutput.length;
  return {
    cell_id: cell.id, status: cell.status, output,
    ...(cell.exitCode === undefined ? {} : { exit_code: cell.exitCode }),
  };
}

function cellTools(context: ToolContext): JsonObject {
  const tools: JsonObject = {};
  for (const name of TOOL_NAMES) {
    tools[name] = async (input: unknown) => {
      const result = await callTool(name, input, context);
      return result.data;
    };
  }
  return Object.freeze(tools);
}

function execOptions(source: string): { yieldTime: number; maxTokens?: number } {
  const match = source.match(/^\s*\/\/\s*@exec:\s*(\{[^\r\n]*\})/);
  if (!match) return { yieldTime: 10_000 };
  let options: JsonObject;
  try { options = object(JSON.parse(match[1]), "@exec"); }
  catch { throw new Error("@exec must contain a JSON object"); }
  return {
    yieldTime: number(options.yield_time_ms, "@exec.yield_time_ms") ?? 10_000,
    maxTokens: number(options.max_output_tokens, "@exec.max_output_tokens"),
  };
}

function cellOutput(cell: Cell, maxTokens?: number): ToolResult {
  const data = cellResult(cell, maxTokens);
  const status = cell.status === "running" ? `Script running with cell ID ${cell.id}`
    : cell.status === "terminated" ? "Script terminated" : cell.status === "errored" ? "Script failed" : "Script completed";
  const content = `${status}\nWall time ${((Date.now() - cell.startedAt) / 1_000).toFixed(1)} seconds\nOutput:\n${data.output}`;
  return response(data, cell.status === "errored" ? 1 : 0, content);
}

async function execCell(input: unknown, context: ToolContext): Promise<ToolResult> {
  if (input && typeof input === "object" && !Array.isArray(input) && "cmd" in input) {
    const args = { ...(input as JsonObject) };
    if (args.max_output_tokens === undefined && typeof args.max_output_chars === "number") args.max_output_tokens = Math.ceil(args.max_output_chars / 4);
    return execCommand(args, context);
  }
  if (typeof input !== "string") throw new Error("exec requires raw JavaScript source");
  const options = execOptions(input);
  const state = stateFor(context);
  const id = String(state.nextCellId++);
  const cell: Cell = { id, output: [], status: "running", promise: Promise.resolve(), cancelled: false, cursor: 0, startedAt: Date.now() };
  state.cells.set(id, cell);
  const append = (value: unknown) => { if (!cell.cancelled) cell.output.push(parseText(value)); };
  const sandbox = createContext({
    tools: cellTools(context),
    text: append,
    image: append,
    notify: append,
    store: (key: string, value: unknown) => state.memory.set(String(key), value),
    load: (key: string) => state.memory.get(String(key)),
    setTimeout,
    clearTimeout,
  }, { codeGeneration: { strings: false, wasm: false } });
  try {
    const script = new Script(`(async () => {\n${input}\n})()`, { filename: `cell-${id}.js` });
    const execution = Promise.resolve(script.runInContext(sandbox, { timeout: 1_000 }));
    cell.promise = execution.then((value) => {
      if (value !== undefined) append(value);
      if (!cell.cancelled) { cell.status = "completed"; cell.exitCode = 0; }
    }).catch((error) => {
      if (!cell.cancelled) { cell.status = "errored"; cell.exitCode = 1; append(String(error)); }
    });
  } catch (error) {
    cell.status = "errored"; cell.exitCode = 1; append(String(error));
    cell.promise = Promise.resolve();
  }
  await Promise.race([cell.promise, delay(Math.max(0, options.yieldTime))]);
  return cellOutput(cell, outputLimit(options.maxTokens));
}

async function waitCell(input: unknown, context: ToolContext): Promise<ToolResult> {
  const args = object(input, "wait");
  const cell = stateFor(context).cells.get(string(args.cell_id, "cell_id")!);
  if (!cell) throw new Error(`unknown cell_id: ${args.cell_id}`);
  if (boolean(args.terminate, "terminate")) {
    cell.cancelled = true; cell.status = "terminated"; cell.exitCode = 130;
  } else {
    await Promise.race([cell.promise, delay(number(args.yield_time_ms, "yield_time_ms") ?? 10_000)]);
  }
  return cellOutput(cell, outputLimit(args.max_tokens));
}

export async function callTool(name: string, input: unknown, context: ToolContext): Promise<ToolResult> {
  switch (name) {
    case "exec_command": return execCommand(input, context);
    case "write_stdin": return writeStdin(input, context);
    case "apply_patch": return applyPatchTool(input, context);
    case "view_image": return viewImage(input, context);
    case "update_plan": return updatePlan(input, context);
    case "request_user_input": return requestUserInput(input, context);
    case "spawn_agent": return spawnAgent(input, context);
    case "list_agents": return listAgents(input, context);
    case "send_message": return sendAgentMessage(input, context, false);
    case "followup_task": return sendAgentMessage(input, context, true);
    case "wait_agent": return waitAgent(input, context);
    case "interrupt_agent": return interruptAgent(input, context);
    case "list_mcp_resources": return listMcpResources(input, context);
    case "list_mcp_resource_templates": return listMcpResourceTemplates(input, context);
    case "get_goal": return getGoal(input, context);
    case "create_goal": return createGoal(input, context);
    case "update_goal": return updateGoal(input, context);
    case "shell_command": return shellCommand(input, context);
    case "exec": return execCell(input, context);
    case "wait": return waitCell(input, context);
    default: throw new Error(`unknown tool: ${name}`);
  }
}
