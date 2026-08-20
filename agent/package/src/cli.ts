import { mkdir, readdir, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { extract } from "../../data/src/codex-jsonl.ts";
import { deepseek } from "../../runtime/src/deepseek.ts";
import { run } from "../../runtime/src/runner.ts";
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
