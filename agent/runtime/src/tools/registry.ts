import { execCommand } from "./exec-command.ts";
import { applyPatch } from "./apply-patch.ts";

export type ToolContext = { root: string; cwd?: string };
export async function callTool(name: string, input: any, context: ToolContext): Promise<{ content: string; exitCode: number }> {
  if (name === "exec_command") {
    const result = await execCommand(String(input.command ?? ""), context.root, input.cwd ?? context.cwd, Number(input.timeout ?? 30_000));
    return { content: result.output, exitCode: result.exitCode };
  }
  if (name === "apply_patch") return { content: await applyPatch(String(input.patch ?? ""), context.root), exitCode: 0 };
  throw new Error(`unknown tool: ${name}`);
}

export const toolSchemas = [
  { type: "function" as const, function: { name: "exec_command", description: "Run a command inside the workspace.", parameters: { type: "object", required: ["command"], properties: { command: { type: "string" }, cwd: { type: "string" }, timeout: { type: "integer" } } } } },
  { type: "function" as const, function: { name: "apply_patch", description: "Apply a unified git diff inside the workspace.", parameters: { type: "object", required: ["patch"], properties: { patch: { type: "string" } } } } },
];
