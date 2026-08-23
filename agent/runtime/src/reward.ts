import type { Event } from "../../shared/src/types.ts";

export type Verification = { score: number; passed: boolean; reason: string; harnessStatus: string };
export function verify(events: Event[], final: string): Verification {
  const calls = events.filter((event) => event.kind === "tool_call");
  const results = events.filter((event) => event.kind === "tool_result");
  const failures = results.filter((event) => (event.exitCode ?? 0) !== 0).length;
  if (!final.trim()) return { score: 0, passed: false, reason: "empty final answer", harnessStatus: "protocol" };
  const callIds = calls.map((event) => event.toolCallId).filter((id): id is string => Boolean(id));
  const resultIds = results.map((event) => event.toolCallId).filter((id): id is string => Boolean(id));
  if (callIds.length !== calls.length || resultIds.length !== results.length) {
    return { score: 0, passed: false, reason: "tool call/result missing tool_call_id", harnessStatus: "protocol" };
  }
  if (new Set(callIds).size !== callIds.length) {
    return { score: 0, passed: false, reason: "duplicate tool_call_id", harnessStatus: "protocol" };
  }
  if (new Set(resultIds).size !== resultIds.length || resultIds.length !== callIds.length || resultIds.some((id) => !callIds.includes(id))) {
    return { score: 0, passed: false, reason: "tool result IDs do not match tool calls", harnessStatus: "protocol" };
  }
  if (failures) return { score: 0, passed: false, reason: `${failures} tool failure(s)`, harnessStatus: "tool_failure" };
  return { score: 0, passed: false, reason: "missing independent verifier", harnessStatus: "unscored" };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  console.assert(!verify([], "done").passed);
  console.assert(!verify([], "").passed);
  console.assert(verify([
    { role: "assistant", kind: "tool_call", toolCallId: "call_1" },
    { role: "tool", kind: "tool_result", toolCallId: "call_1", exitCode: 0 },
  ], "done").harnessStatus === "unscored");
  console.assert(verify([
    { role: "assistant", kind: "tool_call", toolCallId: "call_1" },
    { role: "assistant", kind: "tool_call", toolCallId: "call_1" },
  ], "done").harnessStatus === "protocol");
}
