import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { run } from "../../runtime/src/runner.ts";
import { parseAction } from "../../runtime/src/parser.ts";
import { execCommand } from "../../runtime/src/tools/exec-command.ts";
import { callTool, toolSchemas, type ToolContext } from "../../runtime/src/tools/registry.ts";
import { renderSftRow } from "../../shared/src/renderer.ts";

const root = await mkdtemp(join(tmpdir(), "nanoagent-check-"));
const patch = ["*** Begin Patch", "*** Add File: result.txt", "+ok", "*** End Patch", ""].join("\n");
const verifyTask = async () => {
  const passed = await readFile(join(root, "result.txt"), "utf8").then((value) => value === "ok\n", () => false);
  return { score: passed ? 1 : 0, passed, reason: passed ? "file verifier passed" : "file verifier failed" };
};
let calls = 0;
const model = { complete: async (messages: any[]) => {
  calls++;
  return messages.length === 1
    ? { tool_calls: [
      { id: "patch", function: { name: "apply_patch", arguments: patch } },
      { id: "read", function: { name: "exec_command", arguments: JSON.stringify({ cmd: "cat result.txt" }) } },
    ] }
    : { content: JSON.stringify({ message: "completed" }) };
} };

const state = await run({ id: "check", prompt: "make the result file", model, tools: toolSchemas,
  context: { root, verifyTask }, statePath: join(root, "state.json") });
if (state.status !== "done" || calls !== 2 || await readFile(join(root, "result.txt"), "utf8") !== "ok\n") {
  throw new Error(`harness self-check failed: ${JSON.stringify(state)}`);
}

let mismatchRejected = false;
try { await run({ id: "other", prompt: "wrong run", model, tools: toolSchemas, context: { root, verifyTask }, statePath: join(root, "state.json") }); }
catch { mismatchRejected = true; }
if (!mismatchRejected) throw new Error("mismatched run state was accepted");

const cwd = await execCommand("pwd", root, ".");
if (cwd.exitCode !== 0 || cwd.output.trim() !== root) throw new Error(`relative cwd failed: ${cwd.output}`);

const nearRoot = await mkdtemp(join(tmpdir(), "nanoagent-near-json-"));
const nearPatch = ["*** Begin Patch", "*** Add File: near.txt", "+near", "*** End Patch", ""].join("\n");
const nearState = await run({ id: "near", prompt: "create near.txt", model: {
  complete: async (messages: any[]) => messages.some((message) => message.role === "tool")
    ? { content: JSON.stringify({ message: "done" }) }
    : { content: `${JSON.stringify({ tool_call: { name: "apply_patch", arguments: nearPatch } })}}` },
}, tools: toolSchemas, context: {
  root: nearRoot,
  verifyTask: async () => ({ score: 1, passed: true, reason: "near verifier" }),
}, statePath: join(nearRoot, "state.json") });
if (nearState.status !== "done" || await readFile(join(nearRoot, "near.txt"), "utf8") !== "near\n") throw new Error("raw text tool-call execution failed");

const toolRoot = await mkdtemp(join(tmpdir(), "nanoagent-tools-"));
const context: ToolContext = {
  root: toolRoot,
  userResponses: [{ selected: "first" }],
  mcpResources: [{ server: "demo", uri: "demo://one", name: "one", mime_type: "text/plain" }],
  mcpResourceTemplates: [{ server: "demo", uri_template: "demo://{id}", name: "template" }],
  spawnAgent: async ({ taskName, message }) => ({ nickname: taskName, completed: message.toUpperCase() }),
};
const data = async (name: string, input: unknown) => (await callTool(name, input, context)).data as any;

const planResult = await callTool("update_plan", { plan: [{ step: "one", status: "in_progress" }] }, context);
if (planResult.content !== "Plan updated") throw new Error("plan result format changed");
await data("create_goal", { objective: "finish check" });
if ((await data("get_goal", {})).goal.threadId !== "root") throw new Error("goal response fields changed");
if ((await data("update_goal", { status: "complete" })).goal.status !== "complete") throw new Error("goal was not completed");
if ((await data("request_user_input", { questions: [{ id: "q", header: "Q", question: "choose", options: [{ label: "one", description: "one" }, { label: "two", description: "two" }] }] })).answers.selected !== "first") throw new Error("user response failed");
await data("spawn_agent", { task_name: "child", message: "work" });
if ((await data("wait_agent", { timeout_ms: 1_000 })).timed_out) throw new Error("subagent did not complete");
await data("send_message", { target: "child", message: "note" });
await data("followup_task", { target: "child", message: "again" });
await data("interrupt_agent", { target: "child" });
const agents = await data("list_agents", {});
if (agents.agents.length !== 2 || agents.agents[0].agent_name !== "/root" || agents.agents[1].agent_name !== "/root/child") throw new Error("agent list failed");
const resources = await data("list_mcp_resources", {});
if (resources.resources.length !== 1 || resources.resources[0].mimeType !== "text/plain" || "mime_type" in resources.resources[0]) throw new Error("MCP resource list failed");
const templates = await data("list_mcp_resource_templates", {});
if (templates.resourceTemplates.length !== 1 || templates.resourceTemplates[0].uriTemplate !== "demo://{id}") throw new Error("MCP template list failed");

const firstCommand = await callTool("exec_command", { cmd: "printf first; sleep .01; printf second", yield_time_ms: 1 }, context);
if (!firstCommand.content.startsWith("Chunk ID:")) throw new Error("exec output format changed");
let shell = firstCommand.data as any;
if (!shell.session_id) throw new Error("exec_command did not create a session");
shell = await data("write_stdin", { session_id: shell.session_id, yield_time_ms: 1_000 });
if (shell.exit_code !== 0 || !shell.output.includes("second")) throw new Error("write_stdin failed");
const legacy = await callTool("shell_command", { command: "printf legacy", workdir: "." }, context);
if ((legacy.data as any).output !== "legacy" || !legacy.content.startsWith("Exit code:")) throw new Error("shell_command failed");
const patchResult = await data("apply_patch", "*** Begin Patch\n*** Add File: raw.txt\n+raw\n*** End Patch\n");
if (patchResult.metadata?.exit_code !== 0 || !patchResult.output?.startsWith("Success.")) throw new Error("patch response format changed");
if (await readFile(join(toolRoot, "raw.txt"), "utf8") !== "raw\n") throw new Error("raw apply_patch failed");
await writeFile(join(toolRoot, "old.txt"), "old\n");
let atomicRejected = false;
try {
  await data("apply_patch", "*** Begin Patch\n*** Add File: half.txt\n+half\n*** Update File: old.txt\n@@\n-missing\n+changed\n*** End Patch\n");
} catch { atomicRejected = true; }
if (!atomicRejected || await readFile(join(toolRoot, "old.txt"), "utf8") !== "old\n" || await readFile(join(toolRoot, "half.txt"), "utf8").then(() => true, () => false)) {
  throw new Error("failed patch was not atomic");
}
const cellResult = await callTool("exec", "const r = await tools.exec_command({cmd: 'printf cell'}); text(r.output);", context);
if (!cellResult.content.startsWith("Script completed")) throw new Error("exec cell output format changed");
const cell = cellResult.data as any;
if (cell.status !== "completed" || !cell.output.includes("cell")) throw new Error("exec cell failed");
if ((await data("wait", { cell_id: cell.cell_id, yield_time_ms: 1 })).exit_code !== 0) throw new Error("wait failed");
const yielded = await data("exec", "// @exec: {\"yield_time_ms\": 1}\nawait new Promise((resolve) => setTimeout(() => { text('late'); resolve(); }, 20));");
if (yielded.status !== "running") throw new Error("exec did not yield a live cell");
const waited = await data("wait", { cell_id: yielded.cell_id, yield_time_ms: 1_000 });
if (waited.status !== "completed" || !waited.output.includes("late")) throw new Error("wait did not return new cell output");
await writeFile(join(toolRoot, "image.png"), Buffer.from("iVBORw0KGgo=", "base64"));
if (!(await data("view_image", { path: "image.png" })).image_url.startsWith("data:image/png")) throw new Error("view_image failed");

if (toolSchemas.length !== 20 || new Set(toolSchemas.map((tool) => tool.type === "function" ? tool.function.name : tool.name)).size !== 20) {
  throw new Error("tool registry does not expose exactly 20 tools");
}

const rendered = renderSftRow([
  { role: "assistant", kind: "tool_call", tool: "apply_patch", toolCallId: "call_1", arguments: nearPatch },
  { role: "tool", kind: "tool_result", toolCallId: "call_1", toolResult: { output: "ok", exit_code: 0 }, exitCode: 0 },
]);
if (rendered.messages[0]?.tool_calls?.[0]?.function.arguments !== nearPatch || rendered.messages[1]?.tool_call_id !== "call_1") {
  throw new Error("raw arguments or tool result were not preserved in rendering");
}
const renderedImage = renderSftRow([{ role: "tool", kind: "tool_result", toolCallId: "image", toolResult: { detail: "high", image_url: "data:image/png;base64,AA==" } }]);
if (!Array.isArray(renderedImage.messages[0]?.content) || renderedImage.messages[0]?.content[1]?.type !== "image_url") {
  throw new Error("image tool result was flattened to text");
}
if (renderSftRow([{ role: "tool", kind: "tool_result", toolCallId: "plan", content: "Plan updated", toolResult: {} }]).messages[0]?.content !== "Plan updated") {
  throw new Error("tool text result was rewritten during rendering");
}
if (parseAction('{"name":"exec_command","arguments":{"cmd":"true"}}').kind !== "tool_call") throw new Error("JSON tool call was not recovered");

const verifierState = await run({ id: "verifier-overrides-tool-failure", prompt: "complete despite one command failure", tools: toolSchemas,
  context: { root: toolRoot, verifyTask: async () => ({ score: 1, passed: true, reason: "postcondition passed" }) }, statePath: join(toolRoot, "verifier-overrides-tool-failure.json"), model: {
    complete: async (messages: any[]) => messages.some((message) => message.role === "tool")
      ? { content: "done" }
      : { tool_calls: [{ id: "fails", function: { name: "exec_command", arguments: JSON.stringify({ cmd: "false" }) } }] },
  } });
if (verifierState.status !== "done" || verifierState.events.find((event) => event.kind === "tool_result" && event.toolCallId === "fails")?.exitCode !== 1) {
  throw new Error("independent verifier did not override a recoverable tool failure");
}

let childRuns = 0;
const defaultAgentState = await run({ id: "default-agent", prompt: "parent", tools: toolSchemas, context: {
  root: toolRoot, verifyTask: async () => ({ score: 1, passed: true, reason: "parent verifier" }),
}, statePath: join(toolRoot, "default-agent.json"), maxTurns: 5, model: {
  complete: async (messages: any[]) => {
    const prompt = messages.find((message) => message.role === "user")?.content;
    if (prompt === "child" || prompt === "again") { childRuns++; return { content: `${prompt} done` }; }
    const results = messages.filter((message) => message.role === "tool").length;
    if (results === 0) return { tool_calls: [
        { id: "spawn", function: { name: "spawn_agent", arguments: JSON.stringify({ task_name: "child", message: "child" }) } },
        { id: "wait", function: { name: "wait_agent", arguments: JSON.stringify({ timeout_ms: 1_000 }) } },
      ] };
    if (results === 2) return { tool_calls: [
      { id: "followup", function: { name: "followup_task", arguments: JSON.stringify({ target: "child", message: "again" }) } },
      { id: "wait-followup", function: { name: "wait_agent", arguments: JSON.stringify({ timeout_ms: 1_000 }) } },
    ] };
    return { content: "parent done" };
  },
} });
if (childRuns !== 2 || defaultAgentState.status !== "done") throw new Error("default subagent runner failed");

await rm(nearRoot, { recursive: true, force: true });
await rm(toolRoot, { recursive: true, force: true });
await rm(root, { recursive: true, force: true });
console.log("harness self-check passed");
