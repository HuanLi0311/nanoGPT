import { spawn, type ChildProcess } from "node:child_process";
import { workspacePath } from "../../../workspace/boundary.ts";

export type ExecCommandInput = {
  cmd: string;
  workdir?: string;
  tty?: boolean;
  yield_time_ms?: number;
  max_output_tokens?: number;
  shell?: string;
  login?: boolean;
};

export type WriteStdinInput = {
  session_id: number;
  chars?: string;
  yield_time_ms?: number;
  max_output_tokens?: number;
};

export type ExecResult = {
  output: string;
  wall_time_seconds: number;
  exit_code?: number;
  session_id?: number;
  chunk_id?: string;
  original_token_count?: number;
};

export type CommandResult = { output: string; exitCode: number };

type Session = {
  id: number;
  child: ChildProcess;
  startedAt: number;
  output: string;
  cursor: number;
  exitCode?: number;
  done: Promise<void>;
  finish: () => void;
};

const DEFAULT_YIELD_MS = 10_000;
const MAX_YIELD_MS = 30_000;
const MAX_CAPTURE_CHARS = 4_000_000;

const delay = (milliseconds: number) => new Promise<void>((resolveDelay) => setTimeout(resolveDelay, milliseconds));

function yieldMilliseconds(value: unknown, fallback = DEFAULT_YIELD_MS): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed >= 0 ? Math.min(Math.floor(parsed), MAX_YIELD_MS) : fallback;
}

function outputLimit(value: unknown): number | undefined {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? Math.floor(parsed) * 4 : undefined;
}

function shellQuote(value: string): string {
  return `'${value.replace(/'/g, `'"'"'`)}'`;
}

function commandProcess(input: ExecCommandInput, cwd: string): ChildProcess {
  const shell = input.shell || process.env.SHELL || "/bin/bash";
  const login = input.login !== false;
  if (input.tty && process.platform !== "win32") {
    const command = `${shellQuote(shell)} ${login ? "-lc" : "-c"} ${shellQuote(input.cmd)}`;
    return spawn("script", ["-qefc", command, "/dev/null"], { cwd, stdio: "pipe" });
  }
  return spawn(shell, [login ? "-lc" : "-c", input.cmd], { cwd, stdio: "pipe" });
}

/** Keeps Codex-like shell sessions alive between exec_command and write_stdin. */
export class ShellManager {
  private nextId = 1;
  private nextChunk = 1;
  private readonly sessions = new Map<number, Session>();

  async exec(input: ExecCommandInput, root: string, defaultCwd?: string): Promise<ExecResult> {
    if (!input.cmd?.trim()) throw new Error("cmd is required");
    const cwd = await workspacePath(root, input.workdir ?? defaultCwd, true);
    const child = commandProcess(input, cwd);
    let finish = () => {};
    const done = new Promise<void>((resolveDone) => { finish = resolveDone; });
    const session: Session = {
      id: this.nextId++, child, startedAt: Date.now(), output: "", cursor: 0, done, finish,
    };
    const append = (value: unknown) => {
      session.output += String(value);
      if (session.output.length > MAX_CAPTURE_CHARS) {
        const removed = session.output.length - MAX_CAPTURE_CHARS;
        session.output = session.output.slice(-MAX_CAPTURE_CHARS);
        session.cursor = Math.max(0, session.cursor - removed);
      }
    };
    child.stdout?.on("data", append);
    child.stderr?.on("data", append);
    child.on("error", (error) => append(`${error.message}\n`));
    child.on("close", (code) => { session.exitCode = code ?? 1; session.finish(); });
    this.sessions.set(session.id, session);
    await Promise.race([session.done, delay(yieldMilliseconds(input.yield_time_ms))]);
    return this.result(session, input.max_output_tokens, true);
  }

  async write(input: WriteStdinInput): Promise<ExecResult> {
    const session = this.sessions.get(Number(input.session_id));
    if (!session) throw new Error(`unknown or completed session_id: ${input.session_id}`);
    if (session.exitCode === undefined && input.chars) session.child.stdin?.write(input.chars);
    await Promise.race([session.done, delay(yieldMilliseconds(input.yield_time_ms, input.chars ? 250 : DEFAULT_YIELD_MS))]);
    return this.result(session, input.max_output_tokens, true);
  }

  terminate(sessionId: number): boolean {
    const session = this.sessions.get(sessionId);
    if (!session || session.exitCode !== undefined) return false;
    session.child.kill("SIGTERM");
    return true;
  }

  private result(session: Session, maxOutputTokens: unknown, consumeTerminal: boolean): ExecResult {
    const fresh = session.output.slice(session.cursor);
    session.cursor = session.output.length;
    const originalTokenCount = Math.ceil(fresh.length / 4);
    const limit = outputLimit(maxOutputTokens);
    const output = limit && fresh.length > limit ? fresh.slice(-limit) : fresh;
    const result: ExecResult = {
      output,
      wall_time_seconds: (Date.now() - session.startedAt) / 1_000,
      chunk_id: `chunk_${this.nextChunk++}`,
      ...(limit && fresh.length > limit ? { original_token_count: originalTokenCount } : {}),
    };
    if (session.exitCode === undefined) result.session_id = session.id;
    else {
      result.exit_code = session.exitCode;
      if (consumeTerminal) this.sessions.delete(session.id);
    }
    return result;
  }
}

/** Backwards-compatible helper used by existing diagnostics and tests. */
export async function execCommand(command: string, root: string, cwd = ".", timeout = 30_000): Promise<CommandResult> {
  const manager = new ShellManager();
  const deadline = Date.now() + Math.max(1, timeout);
  let result = await manager.exec({ cmd: command, workdir: cwd, yield_time_ms: Math.min(timeout, MAX_YIELD_MS) }, root);
  while (result.session_id !== undefined && Date.now() < deadline) {
    result = await manager.write({ session_id: result.session_id, yield_time_ms: Math.min(MAX_YIELD_MS, deadline - Date.now()) });
  }
  if (result.session_id !== undefined) manager.terminate(result.session_id);
  return { output: result.output, exitCode: result.exit_code ?? 124 };
}
