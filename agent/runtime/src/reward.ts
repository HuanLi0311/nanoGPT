import type { Event } from "../../shared/src/types.ts";

export type Verification = { score: number; passed: boolean; reason: string; harnessStatus: string };
export function verify(events: Event[], final: string): Verification {
  const calls = events.filter((event) => event.kind === "tool_call").length;
  const results = events.filter((event) => event.kind === "tool_result");
  const failures = results.filter((event) => (event.exitCode ?? 0) !== 0).length;
  if (!final.trim()) return { score: 0, passed: false, reason: "empty final answer", harnessStatus: "protocol" };
  if (results.length < calls) return { score: 0, passed: false, reason: "unresolved tool call", harnessStatus: "protocol" };
  if (failures) return { score: 0, passed: false, reason: `${failures} tool failure(s)`, harnessStatus: "healthy" };
  return { score: 0, passed: false, reason: "missing independent verifier", harnessStatus: "unscored" };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  console.assert(!verify([], "done").passed);
  console.assert(!verify([], "").passed);
}
