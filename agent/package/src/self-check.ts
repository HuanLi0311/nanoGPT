import { mkdtemp, readFile, rm } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { run } from "../../runtime/src/runner.ts";
import { toolSchemas } from "../../runtime/src/tools/registry.ts";

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
await rm(root, { recursive: true, force: true });
console.log("harness self-check passed");
