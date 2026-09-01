import { lstat, realpath } from "node:fs/promises";
import { dirname, isAbsolute, relative, resolve } from "node:path";

function inside(root: string, path: string): boolean {
  const distance = relative(root, path);
  return distance === "" || (!distance.startsWith("..") && !isAbsolute(distance));
}

async function rejectSymlinkComponents(root: string, path: string): Promise<void> {
  let current = root;
  const distance = relative(root, path);
  for (const part of distance ? distance.split(/[\\/]/) : []) {
    current = resolve(current, part);
    try {
      if ((await lstat(current)).isSymbolicLink()) throw new Error(`symlink path is not allowed: ${path}`);
    } catch (error: any) {
      if (error?.code === "ENOENT") break;
      throw error;
    }
  }
}

async function existingAncestor(path: string): Promise<string> {
  let current = path;
  while (true) {
    try { return await realpath(current); }
    catch (error: any) {
      if (error?.code !== "ENOENT") throw error;
      const parent = dirname(current);
      if (parent === current) throw error;
      current = parent;
    }
  }
}

/** Resolve a tool path under root or the virtual /workspace root. */
export async function workspacePath(root: string, value = ".", mustExist = false): Promise<string> {
  const base = await realpath(resolve(root));
  const raw = String(value || ".");
  const virtual = raw === "/workspace" || raw.startsWith("/workspace/");
  const lexical = virtual
    ? resolve(base, raw.slice("/workspace".length))
    : isAbsolute(raw) ? resolve(raw) : resolve(base, raw);
  if (!inside(base, lexical)) throw new Error("path escapes workspace");
  await rejectSymlinkComponents(base, lexical);
  const ancestor = await existingAncestor(lexical);
  if (!inside(base, ancestor)) throw new Error("path escapes workspace");
  try {
    const existing = await realpath(lexical);
    if (!inside(base, existing)) throw new Error("path escapes workspace");
    return existing;
  } catch (error: any) {
    if (error?.code !== "ENOENT") throw error;
    if (mustExist) throw error;
    return lexical;
  }
}
