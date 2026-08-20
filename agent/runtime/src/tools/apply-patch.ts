import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { resolve, relative } from "node:path";
const apply = promisify(execFile);

export async function applyPatch(diff: string, root: string): Promise<string> {
  if (!diff.trim()) throw new Error("patch is empty");
  if (/(^|\n)diff --git a\//.test(diff) === false) throw new Error("patch must be a unified git diff");
  const base = resolve(root);
  for (const path of [...diff.matchAll(/^(?:---|\+\+\+) [ab]\/([^\t\n]+)/gm)].map((m) => m[1])) {
    if (relative(base, resolve(base, path)).startsWith("..")) throw new Error("patch escapes workspace");
  }
  const result = await apply("git", ["apply", "--whitespace=nowarn", "--"], { cwd: base, input: diff });
  return result.stdout || "patch applied";
}
