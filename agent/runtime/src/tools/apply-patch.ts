import { execFile } from "node:child_process";
import { mkdir, readFile, unlink, rename, writeFile } from "node:fs/promises";
import { promisify } from "node:util";
import { resolve, relative } from "node:path";
const apply = promisify(execFile);

function find(lines: string[], wanted: string[], from: number): number {
  for (let i = from; i <= lines.length - wanted.length; i++) {
    if (wanted.every((line, offset) => lines[i + offset] === line)) return i;
  }
  return -1;
}

async function applyCodex(diff: string, root: string, paths: string[]): Promise<string> {
  const lines = diff.split(/\r?\n/), base = resolve(root);
  const headers = [...diff.matchAll(/^\*\*\* (Add|Update|Delete) File: (.+)$/gm)];
  for (let index = 0; index < headers.length; index++) {
    const operation = headers[index][1], path = headers[index][2];
    const file = resolve(base, path);
    const headerLine = diff.slice(0, headers[index].index).split(/\r?\n/).length - 1;
    const endLine = index + 1 < headers.length
      ? diff.slice(0, headers[index + 1].index).split(/\r?\n/).length - 1
      : lines.findIndex((line, i) => i > headerLine && line === "*** End Patch");
    const block = lines.slice(headerLine + 1, endLine < 0 ? lines.length : endLine);
    if (operation === "Add") {
      if (await readFile(file, "utf8").then(() => true, () => false)) throw new Error(`file exists: ${path}`);
      await mkdir(resolve(file, ".."), { recursive: true });
      await writeFile(file, block.filter((line) => line.startsWith("+")).map((line) => line.slice(1)).join("\n") + "\n");
      continue;
    }
    if (operation === "Delete") { await unlink(file); continue; }
    let current = (await readFile(file, "utf8")).split(/\r?\n/), cursor = 0, hunks: string[][] = [], hunk: string[] = [];
    for (const line of block) {
      if (line.startsWith("@@")) { if (hunk.length) hunks.push(hunk); hunk = []; }
      else if (line) hunk.push(line);
    }
    if (hunk.length) hunks.push(hunk);
    for (const change of hunks) {
      const oldLines = change.filter((line) => line[0] === " " || line[0] === "-").map((line) => line.slice(1));
      const newLines = change.filter((line) => line[0] === " " || line[0] === "+").map((line) => line.slice(1));
      const at = find(current, oldLines, cursor);
      if (at < 0) throw new Error(`hunk does not match: ${path}`);
      current.splice(at, oldLines.length, ...newLines); cursor = at + newLines.length;
    }
    const temp = `${file}.tmp-${process.pid}`;
    await writeFile(temp, current.join("\n")); await rename(temp, file);
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
  if ((!codex && !/(^|\n)diff --git a\//.test(diff)) || (codex && !paths.length)) {
    throw new Error("patch must be a Codex Begin Patch or unified git diff");
  }
  for (const path of paths) {
    if (relative(base, resolve(base, path)).startsWith("..")) throw new Error("patch escapes workspace");
  }
  if (codex) return applyCodex(diff, root, paths);
  const result = await apply("git", ["apply", "--whitespace=nowarn", "--"], { cwd: base, input: diff });
  return result.stdout || "patch applied";
}
