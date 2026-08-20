import { exec } from "node:child_process";
import { promisify } from "node:util";
import { resolve, relative } from "node:path";

const run = promisify(exec);
export type CommandResult = { output: string; exitCode: number };

export async function execCommand(command: string, root: string, cwd = root, timeout = 30_000): Promise<CommandResult> {
  if (!command.trim()) throw new Error("command is empty");
  const base = resolve(root), work = resolve(cwd);
  if (relative(base, work).startsWith("..")) throw new Error("cwd escapes workspace");
  try {
    const result = await run(command, { cwd: work, timeout, maxBuffer: 1_000_000, shell: "/bin/bash" });
    return { output: result.stdout + result.stderr, exitCode: 0 };
  } catch (error: any) {
    return { output: `${error.stdout ?? ""}${error.stderr ?? ""}${error.message ?? error}`, exitCode: error.code ?? 1 };
  }
}
