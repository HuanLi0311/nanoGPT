import { mkdtemp, readFile, rm } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { run } from "../../runtime/src/runner.ts";
import { execCommand } from "../../runtime/src/tools/exec-command.ts";
import { toolSchemas } from "../../runtime/src/tools/registry.ts";
import { renderSftRow } from "../../shared/src/renderer.ts";

const root = await mkdtemp(join(tmpdir(), "nanoagent-check-"));
const patch = ["*** Begin Patch", "*** Add File: result.txt", "+ok", "*** End Patch", ""].join("\n");
let calls = 0;
const model = { complete: async (messages: any[]) => {
  calls++;
  return messages.length === 1
    ? { tool_calls: [{ id: "patch", function: { name: "apply_patch", arguments: JSON.stringify({ patch }) } }] }
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
const cwd = await execCommand("pwd", root, ".");
if (cwd.exitCode !== 0 || cwd.output.trim() !== root) throw new Error(`relative cwd failed: ${cwd.output}`);
const rendered = renderSftRow([
  { role: "assistant", kind: "tool_call", tool: "exec_command", toolCallId: "call_1", arguments: { command: "true" } },
  { role: "tool", kind: "tool_result", toolCallId: "call_1", content: "ok", exitCode: 0 },
]);
if (rendered.messages[1]?.role !== "tool" || rendered.messages[1].tool_call_id !== "call_1") throw new Error("tool result was dropped from SFT rendering");
await rm(root, { recursive: true, force: true });
console.log("harness self-check passed");
