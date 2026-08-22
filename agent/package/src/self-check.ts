import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { run } from "../../runtime/src/runner.ts";
import { parseAction } from "../../runtime/src/parser.ts";
import { execCommand } from "../../runtime/src/tools/exec-command.ts";
import { applyPatch } from "../../runtime/src/tools/apply-patch.ts";
import { toolSchemas } from "../../runtime/src/tools/registry.ts";
import { renderSftRow } from "../../shared/src/renderer.ts";

const root = await mkdtemp(join(tmpdir(), "nanoagent-check-"));
const patch = ["*** Begin Patch", "*** Add File: result.txt", "+ok", "*** End Patch", ""].join("\n");
let calls = 0;
const model = { complete: async (messages: any[]) => {
  calls++;
  return messages.length === 1
    ? { tool_calls: [
      { id: "patch", function: { name: "apply_patch", arguments: JSON.stringify({ patch }) } },
      { id: "read", function: { name: "exec_command", arguments: JSON.stringify({ command: "cat result.txt" }) } },
    ] }
    : { content: JSON.stringify({ message: "completed" }) };
} };
const state = await run({ id: "check", prompt: "make the result file", model, tools: toolSchemas,
  context: { root }, statePath: join(root, "state.json") });
if (state.status !== "done" || calls !== 2 || await readFile(join(root, "result.txt"), "utf8") !== "ok\n") {
  throw new Error(`harness self-check failed: ${JSON.stringify(state)}`);
}
let mismatchRejected = false;
try { await run({ id: "other", prompt: "wrong run", model, tools: toolSchemas, context: { root }, statePath: join(root, "state.json") }); }
catch { mismatchRejected = true; }
if (!mismatchRejected) throw new Error("mismatched run state was accepted");
let promptMismatchRejected = false;
try { await run({ id: "check", prompt: "wrong prompt", model, tools: toolSchemas, context: { root }, statePath: join(root, "state.json") }); }
catch { promptMismatchRejected = true; }
if (!promptMismatchRejected) throw new Error("mismatched run prompt was accepted");
const cwd = await execCommand("pwd", root, ".");
if (cwd.exitCode !== 0 || cwd.output.trim() !== root) throw new Error(`relative cwd failed: ${cwd.output}`);
const nearRoot = await mkdtemp(join(tmpdir(), "nanoagent-near-json-"));
const nearPatch = ["*** Begin Patch", "*** Add File: near.txt", "+near", "*** End Patch", ""].join("\n");
const nearState = await run({ id: "near", prompt: "create near.txt", model: {
  complete: async (messages: any[]) => messages.some((message) => message.role === "tool")
    ? { content: JSON.stringify({ message: "done" }) }
    : { content: `${JSON.stringify({ tool_call: { name: "apply_patch", arguments: { patch: nearPatch } } })}}` },
}, tools: toolSchemas, context: { root: nearRoot }, statePath: join(nearRoot, "state.json") });
if (nearState.status !== "done" || await readFile(join(nearRoot, "near.txt"), "utf8") !== "near\n") throw new Error("text tool-call execution failed");
await writeFile(join(nearRoot, "calc.py"), "return a - b\n");
await applyPatch("--- a/calc.py\n+++ b/calc.py\n@@ -1 +1 @@\n-return a - b\n+return a + b\n", nearRoot);
if (await readFile(join(nearRoot, "calc.py"), "utf8") !== "return a + b\n") throw new Error("standard unified patch failed");
await rm(nearRoot, { recursive: true, force: true });
const rendered = renderSftRow([
  { role: "assistant", kind: "tool_call", tool: "exec_command", toolCallId: "call_1", arguments: { command: "true" } },
  { role: "tool", kind: "tool_result", toolCallId: "call_1", content: "ok", exitCode: 0 },
]);
if (rendered.messages[1]?.role !== "tool" || rendered.messages[1].tool_call_id !== "call_1") throw new Error("tool result was dropped from SFT rendering");
const parallel = renderSftRow([
  { role: "assistant", kind: "tool_call", tool: "a", toolCallId: "a", arguments: {} },
  { role: "assistant", kind: "tool_call", tool: "b", toolCallId: "b", arguments: {} },
  { role: "tool", kind: "tool_result", toolCallId: "a", content: "ok", exitCode: 0 },
  { role: "tool", kind: "tool_result", toolCallId: "b", content: "ok", exitCode: 0 },
]);
if (parallel.messages.length !== 3 || parallel.messages[0]?.tool_calls?.length !== 2) throw new Error("parallel tool calls were split into invalid messages");
if (parseAction('{"name":"exec_command","arguments":{"command":"true"}}}').kind !== "tool_call") throw new Error("near-JSON tool call was not recovered");
await rm(root, { recursive: true, force: true });
console.log("harness self-check passed");
