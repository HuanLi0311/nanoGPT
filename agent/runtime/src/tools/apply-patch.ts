import { spawn } from "node:child_process";
import { mkdir, readFile, unlink, rename, writeFile } from "node:fs/promises";
import { resolve, relative } from "node:path";

function applyGit(diff: string, cwd: string): Promise<string> {
  return new Promise((resolvePromise, reject) => {
    const child = spawn("git", ["apply", "--whitespace=nowarn", "-"], { cwd });
    let stdout = "", stderr = "";
    child.stdout.on("data", (chunk) => { stdout += chunk; });
    child.stderr.on("data", (chunk) => { stderr += chunk; });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code === 0) resolvePromise(stdout || "patch applied");
      else reject(Object.assign(new Error(stderr || `git apply exited ${code}`), { stdout, stderr, code }));
    });
    child.stdin.end(diff);
  });
}

function find(lines: string[], wanted: string[], from: number): number {
  for (let i = from; i <= lines.length - wanted.length; i++) {
    if (wanted.every((line, offset) => lines[i + offset] === line)) return i;
  }
  return -1;
}

async function existing(file: string): Promise<string | undefined> {
  try { return await readFile(file, "utf8"); }
  catch (error: any) { if (error?.code === "ENOENT") return undefined; throw error; }
}

async function applyCodex(diff: string, root: string, paths: string[]): Promise<string> {
  const lines = diff.split(/\r?\n/), base = resolve(root);
  const headers = [...diff.matchAll(/^\*\*\* (Add|Update|Delete) File: (.+)$/gm)];
  const next = new Map<string, string | undefined>();
  const current = async (file: string) => {
    if (!next.has(file)) next.set(file, await existing(file));
    return next.get(file);
  };
  for (let index = 0; index < headers.length; index++) {
    const operation = headers[index][1], path = headers[index][2];
    const file = resolve(base, path);
    const headerLine = diff.slice(0, headers[index].index).split(/\r?\n/).length - 1;
    const endLine = index + 1 < headers.length
      ? diff.slice(0, headers[index + 1].index).split(/\r?\n/).length - 1
      : lines.findIndex((line, i) => i > headerLine && line === "*** End Patch");
    const block = lines.slice(headerLine + 1, endLine < 0 ? lines.length : endLine);
    if (operation === "Add") {
      if (await current(file) !== undefined) throw new Error(`file exists: ${path}`);
      next.set(file, block.filter((line) => line.startsWith("+")).map((line) => line.slice(1)).join("\n") + "\n");
      continue;
    }
    if (operation === "Delete") {
      if (await current(file) === undefined) throw new Error(`file does not exist: ${path}`);
      next.set(file, undefined);
      continue;
    }
    const content = await current(file);
    if (content === undefined) throw new Error(`file does not exist: ${path}`);
    let updated = content.split(/\r?\n/), cursor = 0, hunks: string[][] = [], hunk: string[] = [];
    for (const line of block) {
      if (line.startsWith("@@")) { if (hunk.length) hunks.push(hunk); hunk = []; }
      else if (line) hunk.push(line);
    }
    if (hunk.length) hunks.push(hunk);
    for (const change of hunks) {
      const oldLines = change.filter((line) => line[0] === " " || line[0] === "-").map((line) => line.slice(1));
      const newLines = change.filter((line) => line[0] === " " || line[0] === "+").map((line) => line.slice(1));
      const at = find(updated, oldLines, cursor);
      if (at < 0) throw new Error(`hunk does not match: ${path}`);
      updated.splice(at, oldLines.length, ...newLines); cursor = at + newLines.length;
    }
    next.set(file, updated.join("\n"));
  }
  const writes: { file: string; temp: string }[] = [];
  for (const [file, content] of next) {
    if (content === undefined) continue;
    await mkdir(resolve(file, ".."), { recursive: true });
    const temp = `${file}.tmp-${process.pid}-${writes.length}`;
    await writeFile(temp, content);
    writes.push({ file, temp });
  }
  try {
    for (const { file, temp } of writes) await rename(temp, file);
    for (const [file, content] of next) if (content === undefined) await unlink(file);
  } finally {
    await Promise.all(writes.map(({ temp }) => unlink(temp).catch(() => {})));
  }
  return `applied ${paths.length} file(s)`;
}

export async function applyPatch(diff: string, root: string): Promise<string> {
  if (!diff.trim()) throw new Error("patch is empty");
  const base = resolve(root);
  const codex = diff.startsWith("*** Begin Patch");
  const paths = codex
    ? [...diff.matchAll(/^\*\*\* (?:Add|Update|Delete) File: (.+)$/gm)].map((m) => m[1])
    : [...diff.matchAll(/^(?:---|\+\+\+) [ab]\/([^\t\n]+)/gm)].map((m) => m[1]);
  const unified = /^(?:---|\+\+\+) (?:[ab]\/|\/dev\/null)/m.test(diff);
  if ((!codex && !(/(^|\n)diff --git a\//.test(diff) || (unified && paths.length))) || (codex && !paths.length)) {
    throw new Error("patch must be a Codex Begin Patch or unified git diff");
  }
  for (const path of paths) {
    if (relative(base, resolve(base, path)).startsWith("..")) throw new Error("patch escapes workspace");
  }
  if (codex) return applyCodex(diff, root, paths);
  return applyGit(diff, base);
}
