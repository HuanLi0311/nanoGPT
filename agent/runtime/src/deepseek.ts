import type { ChatMessage, ToolSpec } from "../../shared/src/types.ts";
import type { Model } from "./loop.ts";

export function deepseek(apiKey = process.env.DEEPSEEK_API_KEY, model = process.env.DEEPSEEK_MODEL ?? "deepseek-chat"): Model {
  if (!apiKey) throw new Error("DEEPSEEK_API_KEY is required");
  return { async complete(messages: ChatMessage[], tools: ToolSpec[]) {
    const timeout = Number(process.env.DEEPSEEK_TIMEOUT_MS ?? 120_000);
    const response = await fetch(process.env.DEEPSEEK_BASE_URL ?? "https://api.deepseek.com/chat/completions", {
      method: "POST", headers: { "content-type": "application/json", authorization: `Bearer ${apiKey}` },
      body: JSON.stringify({ model, messages, tools: tools.length ? tools : undefined, temperature: 0.2 }),
      signal: AbortSignal.timeout(Number.isFinite(timeout) && timeout > 0 ? timeout : 120_000),
    });
    if (!response.ok) throw new Error(`DeepSeek HTTP ${response.status}: ${(await response.text()).slice(0, 500)}`);
    const choice = (await response.json() as any).choices?.[0]?.message;
    if (!choice) throw new Error("DeepSeek returned no message");
    return choice;
  } };
}
