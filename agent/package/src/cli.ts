import { mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import { basename, dirname, join, resolve } from "node:path";
import type { Event, Trajectory } from "../../shared/src/types.ts";
import { deepseek } from "../../runtime/src/deepseek.ts";
import { run } from "../../runtime/src/runtime.ts";
import { toolSchemas } from "../../runtime/src/tools/registry.ts";

async function files(root: string): Promise<string[]> {
  const out: string[] = [];
  for (const name of await readdir(root, { withFileTypes: true })) {
    const path = join(root, name.name);
    if (name.isDirectory()) out.push(...await files(path));
    else if (name.name.endsWith(".jsonl")) out.push(path);
  }
  return out;
}

async function extract(file: string): Promise<Trajectory> {
  const events: Event[] = [];
  for (const line of (await readFile(file, "utf8")).split(/\r?\n/)) {
    if (!line.trim()) continue;
    let raw: any;
    try { raw = JSON.parse(line); } catch { continue; }
    const item = raw?.payload && typeof raw.payload === "object" ? raw.payload : raw;
    const role = item?.role;
    if (!["system", "user", "assistant", "tool"].includes(role)) continue;
    const kind = item.kind === "tool_call" || item.kind === "tool_result" ? item.kind : "message";
    events.push({
      role,
      kind,
      content: typeof item.content === "string" ? item.content : typeof item.text === "string" ? item.text : undefined,
      tool: typeof item.tool === "string" ? item.tool : typeof item.name === "string" ? item.name : undefined,
      arguments: item.arguments,
      toolCallId: item.tool_call_id ?? item.toolCallId ?? item.id,
      toolResult: item.tool_result ?? item.toolResult,
      exitCode: item.exit_code ?? item.exitCode,
    });
  }
  return { trajectoryId: basename(file), source: "codex", events };
}

const command = process.argv[2] ?? "extract";
if (command === "extract") {
  const root = process.argv[3] ?? `${process.env.HOME}/.codex/sessions`;
  const output = process.argv[4] ?? "../data/canonical/codex-trajectories.jsonl";
  await mkdir(dirname(resolve(output)), { recursive: true });
  const rows: string[] = [];
  for (const file of await files(root)) {
    const trajectory = await extract(file);
    if (trajectory.events.some((event) => event.role === "user")) rows.push(JSON.stringify(trajectory));
  }
  await writeFile(output, rows.length ? `${rows.join("\n")}\n` : "");
} else if (command === "run") {
  const prompt = process.argv.slice(3).join(" ");
  if (!prompt) throw new Error("usage: npm run cli -- run <prompt>");
  const id = (process.env.NANOAGENT_RUN_ID ?? `run-${Date.now()}-${process.pid}`).replace(/[^a-zA-Z0-9._-]/g, "_");
  const state = await run({ id, prompt, model: deepseek(), tools: toolSchemas,
    context: { root: resolve("../..") }, statePath: resolve("../runs", `${id}.json`) });
  console.log(JSON.stringify(state, null, 2));
} else {
  throw new Error(`unknown command: ${command}`);
}
