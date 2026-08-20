import type { Event } from "../../shared/src/types.ts";

export type Verification = { score: number; passed: boolean; reason: string };
export function verify(events: Event[], final: string): Verification {
  const calls = events.filter((event) => event.kind === "tool_call").length;
  const results = events.filter((event) => event.kind === "tool_result");
  const failures = results.filter((event) => (event.exitCode ?? 0) !== 0).length;
  if (!final.trim()) return { score: 0, passed: false, reason: "empty final answer" };
  if (results.length < calls) return { score: 0, passed: false, reason: "unresolved tool call" };
  if (failures) return { score: Math.max(0, 0.5 - failures * 0.1), passed: false, reason: `${failures} tool failure(s)` };
  return { score: Math.min(1, 0.5 + (calls ? 0.25 : 0) + 0.25), passed: true, reason: "completed" };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  console.assert(verify([], "done").passed);
  console.assert(!verify([], "").passed);
}
