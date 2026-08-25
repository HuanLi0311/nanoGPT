import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { tmpdir } from "node:os";
import { deepseek } from "../../runtime/src/deepseek.ts";
import { run, type Model, type RunState } from "../../runtime/src/runtime.ts";
import { execCommand } from "../../runtime/src/tools/exec-command.ts";
import { toolSchemas } from "../../runtime/src/tools/registry.ts";
import type { ChatMessage, ToolSpec } from "../../shared/src/types.ts";

type Response = { content?: string; tool_calls?: { id?: string; function: { name: string; arguments: string } }[] };
type TaskResult = { passed: boolean; score: number; reason: string };
type Task = { id: string; prompt: string; setup(root: string): Promise<void>; verify(root: string): Promise<TaskResult>; scripted(messages: ChatMessage[]): Response };

const call = (id: string, name: string, args: unknown) => ({ id, function: { name, arguments: typeof args === "string" ? args : JSON.stringify(args) } });
const patch = (body: string) => `*** Begin Patch\n${body}\n*** End Patch\n`;
const noSetup = async (_root: string) => {};
const fileCheck = (name: string, expected: string) => async (root: string): Promise<TaskResult> => {
  try {
    const actual = await readFile(join(root, name), "utf8");
    return actual === expected ? { passed: true, score: 1, reason: "external postcondition passed" } : { passed: false, score: 0, reason: `${name} content mismatch` };
  } catch (error) { return { passed: false, score: 0, reason: `${name} missing: ${String(error)}` }; }
};

const tasks: Task[] = [
  {
    id: "write_exact",
    prompt: "In the workspace, create answer.txt containing exactly ALPHA followed by one newline. Verify the file, then reply DONE.",
    setup: noSetup,
    verify: fileCheck("answer.txt", "ALPHA\n"),
    scripted: (messages) => messages.some((m) => m.role === "tool")
      ? { content: JSON.stringify({ message: "DONE" }) }
      : { tool_calls: [call("p1", "apply_patch", patch("*** Add File: answer.txt\n+ALPHA"))] },
  },
  {
    id: "repair_test",
    prompt: "Fix calc.py so add(2, 3) returns 5, run the provided test, and reply DONE only after it passes.",
    setup: async (root) => {
      await writeFile(join(root, "calc.py"), "def add(a, b):\n    return a - b\n", "utf8");
      await writeFile(join(root, "test_calc.py"), "from calc import add\nassert add(2, 3) == 5\n", "utf8");
    },
    verify: async (root) => {
      const result = await execCommand("python3 test_calc.py", root);
      return result.exitCode === 0 ? { passed: true, score: 1, reason: "external test passed" } : { passed: false, score: 0, reason: `external test failed: ${result.output.slice(0, 240)}` };
    },
    scripted: (messages) => messages.some((m) => m.role === "tool")
      ? { content: JSON.stringify({ message: "DONE" }) }
      : { tool_calls: [call("p1", "apply_patch", "--- a/calc.py\n+++ b/calc.py\n@@ -1,2 +1,2 @@\n def add(a, b):\n-    return a - b\n+    return a + b\n"), call("t1", "exec_command", { cmd: "python3 test_calc.py" })] },
  },
  {
    id: "parallel_files",
    prompt: "Create left.txt with LEFT and right.txt with RIGHT, each followed by one newline. Verify both, then reply DONE.",
    setup: noSetup,
    verify: async (root) => {
      const left = await readFile(join(root, "left.txt"), "utf8").catch(() => "");
      const right = await readFile(join(root, "right.txt"), "utf8").catch(() => "");
      return left === "LEFT\n" && right === "RIGHT\n" ? { passed: true, score: 1, reason: "two-file postcondition passed" } : { passed: false, score: 0, reason: "one or both files missing or incorrect" };
    },
    scripted: (messages) => messages.some((m) => m.role === "tool")
      ? { content: JSON.stringify({ message: "DONE" }) }
      : { tool_calls: [call("l", "apply_patch", patch("*** Add File: left.txt\n+LEFT")), call("r", "apply_patch", patch("*** Add File: right.txt\n+RIGHT"))] },
  },
  {
    id: "near_json",
    prompt: "Create near.txt containing NEAR followed by one newline, then reply DONE.",
    setup: noSetup,
    verify: fileCheck("near.txt", "NEAR\n"),
    scripted: (messages) => messages.some((m) => m.role === "tool")
      ? { content: JSON.stringify({ message: "DONE" }) }
      : { content: `${JSON.stringify({ tool_call: { name: "apply_patch", arguments: patch("*** Add File: near.txt\n+NEAR") } })}}` },
  },
];

function wrap(base: Model, trace: Response[]): Model {
  return { complete: async (messages: ChatMessage[], tools: ToolSpec[]) => {
    const response = await base.complete(messages, tools);
    trace.push(response);
    return response;
  } };
}

function scripted(task: Task): Model { return { complete: async (messages) => task.scripted(messages) }; }

function classify(state: RunState | undefined, task: TaskResult, model: string): string {
  if (model === "scripted" && !task.passed) return "harness_or_contract";
  if (!state) return "model_or_transport";
  if (state.verification?.reason === "turn limit exceeded" || state.verification?.reason === "unresolved tool call") return "harness_or_protocol";
  if (!task.passed && state.events.filter((event) => event.kind === "tool_result").every((event) => (event.exitCode ?? 0) === 0)) return "model_or_task";
  return task.passed ? "none" : "undetermined";
}

async function one(task: Task, modelName: string) {
  const root = await mkdtemp(join(tmpdir(), "nanoagent-diagnose-"));
  const trace: Response[] = [];
  try {
    await task.setup(root);
    const base = modelName === "deepseek" ? deepseek() : scripted(task);
    let state: RunState | undefined;
    let error = "";
    try {
      state = await run({ id: `diagnose-${task.id}`, prompt: task.prompt, model: wrap(base, trace), tools: toolSchemas,
        context: { root, verifyTask: async () => {
          const outcome = await task.verify(root);
          return { ...outcome, harnessStatus: "healthy" };
        } }, statePath: join(root, "state.json"),
        maxTurns: Number(process.env.DIAGNOSE_MAX_TURNS ?? 8), retries: Number(process.env.DIAGNOSE_RETRIES ?? 1) });
    } catch (caught) { error = String(caught); }
    const postcondition = await task.verify(root);
    const calls = state?.events.filter((event) => event.kind === "tool_call") ?? [];
    const results = state?.events.filter((event) => event.kind === "tool_result") ?? [];
    const successful = results.filter((event) => (event.exitCode ?? 0) === 0).length;
    return { model: modelName, task: task.id, postcondition, failure_class: classify(state, postcondition, modelName),
      metrics: { generic_verifier_score: state?.verification?.score ?? 0, protocol_valid: calls.length === results.length,
        tool_success_rate: results.length ? successful / results.length : 0, task_success: postcondition.score },
      state: state ? { status: state.status, verification: state.verification, events: state.events } : undefined,
      trace, error: error || undefined };
  } finally { await rm(root, { recursive: true, force: true }); }
}

const mode = process.argv[2] ?? "calibration";
const modelName = mode === "teacher" ? "deepseek" : "scripted";
const offset = Number(process.env.DIAGNOSE_OFFSET ?? 0);
const limit = Number(process.env.DIAGNOSE_LIMIT ?? tasks.length);
const selected = tasks.slice(offset, offset + limit);
const results = [];
for (const task of selected) results.push(await one(task, modelName));
const summary = { mode, model: modelName, tasks: results.length, task_success: results.filter((x) => x.postcondition.passed).length,
  protocol_valid: results.filter((x) => x.metrics.protocol_valid).length, results };
const log = resolve("../../logs", `harness_diagnose_${new Date().toISOString().replace(/[:.]/g, "-")}.json`);
await mkdir(resolve("../../logs"), { recursive: true });
await writeFile(log, JSON.stringify(summary, null, 2), "utf8");
console.log(JSON.stringify({ ...summary, log }, null, 2));
if (modelName === "scripted" && summary.task_success !== summary.tasks) throw new Error("scripted harness calibration failed");
