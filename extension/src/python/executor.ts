import { spawn, ChildProcess } from "child_process";
import * as fs from "fs";
import * as path from "path";
import * as vscode from "vscode";
import { config, log, workspaceRoot } from "../utils/config";
import { loadErrorPatterns } from "../data/content";
import { matchAgainstPatterns } from "./errorParser";
import { filterNoise, flushNoise, initialState, NoiseFilterState } from "./noiseFilter";
import type { ErrorPattern } from "../types";

export interface RunResult {
  exitCode: number | null;
  stdout: string;
  stderr: string;
  durationMs: number;
  timedOut: boolean;
  newFigures: string[];
}

export interface RunHandlers {
  onStdout?: (text: string) => void;
  onStderr?: (text: string) => void;
  onStatus?: (running: boolean) => void;
}

let running: ChildProcess | undefined;

/** 是否有正在运行的脚本 */
export function isRunning(): boolean {
  return !!running;
}

/** 停止当前运行（SIGTERM，2 秒后未退出则 SIGKILL） */
export function stopRun(): void {
  if (running) {
    log("stop requested");
    running.kill("SIGTERM");
    setTimeout(() => {
      try {
        if (running && running.exitCode === null) {
          running.kill("SIGKILL");
        }
      } catch {
        /* already gone */
      }
    }, 2000);
  }
}

/** 列出目录下 mtime 晚于 since 的新图片（图表自动捕获） */
export function newFiguresSince(figuresDir: string, since: number): string[] {
  try {
    return fs.readdirSync(figuresDir)
      .filter((f) => /\.(png|jpg|gif|svg)$/i.test(f))
      .map((f) => path.join(figuresDir, f))
      .filter((f) => fs.statSync(f).mtimeMs > since)
      .sort((a, b) => fs.statSync(a).mtimeMs - fs.statSync(b).mtimeMs);
  } catch {
    return [];
  }
}

/**
 * 运行一段 Python 代码。
 * 方式：child_process.spawn（设计要点 3.4 方式一），工作区根目录为 cwd，
 * PYTHONPATH 追加 <root>/src 以复用 tf2tutorial 包。
 */
export async function runPython(
  code: string,
  handlers: RunHandlers
): Promise<RunResult> {
  const root = workspaceRoot();
  const tmp = path.join(root ?? process.cwd(), ".tfTutorTmp");
  fs.mkdirSync(tmp, { recursive: true });
  const script = path.join(tmp, `cell_${Date.now()}.py`);
  fs.writeFileSync(script, code, "utf8");
  try {
    return await spawnAndCollect([script], handlers);
  } finally {
    fs.rmSync(script, { force: true });
  }
}

/** 直接运行一个脚本文件（章节入口的"▶ 运行"走这里，不依赖编辑器焦点） */
export async function runPythonFile(
  scriptAbsPath: string,
  handlers: RunHandlers
): Promise<RunResult> {
  return spawnAndCollect([scriptAbsPath], handlers);
}

/** 执行环境（可测试）：强制 UTF-8 防 Windows 中文乱码；TF 降噪 */
export function buildExecutorEnv(root?: string): NodeJS.ProcessEnv {
  return {
    ...process.env,
    PYTHONPATH: root ? path.join(root, "src") : process.env.PYTHONPATH ?? "",
    PYTHONIOENCODING: "utf-8",
    PYTHONUTF8: "1",
    TF_CPP_MIN_LOG_LEVEL: "3",
  };
}

function spawnAndCollect(args: string[], handlers: RunHandlers): Promise<RunResult> {
  const root = workspaceRoot();
  const py = resolveInterpreterSync();
  const timeoutMs = config.timeoutSec * 1000;

  const figuresDir = root ? path.join(root, "figures") : "";
  const since = Date.now();

  return new Promise<RunResult>((resolve) => {
    handlers.onStatus?.(true);
    const proc = spawn(py, args, {
      cwd: root ?? process.cwd(),
      env: buildExecutorEnv(root),
    });
    running = proc;

    let stdout = "";
    let stderr = ""; // 完整保留（供报错匹配），显示层会过滤无害噪音
    let timedOut = false;
    const nstate: NoiseFilterState = initialState();

    const timer = setTimeout(() => {
      timedOut = true;
      log(`timeout after ${config.timeoutSec}s, killing`);
      handlers.onStderr?.(`\n[TF 学习伴侣] 超过 ${config.timeoutSec}s 自动终止（可在设置中调整 tfTutor.timeoutSec）\n`);
      proc.kill("SIGTERM");
    }, timeoutMs);

    proc.stdout.on("data", (d: Buffer) => {
      const text = d.toString("utf8");
      stdout += text;
      handlers.onStdout?.(text);
    });
    proc.stderr.on("data", (d: Buffer) => {
      const text = d.toString("utf8");
      stderr += text;
      // 逐行过滤无害日志，其余按到达顺序即时上屏
      const r = filterNoise(nstate, text);
      for (const line of r.shown) {
        handlers.onStderr?.(line + "\n");
      }
    });
    proc.on("error", (err) => {
      clearTimeout(timer);
      handlers.onStatus?.(false);
      stderr += `\n[TF 学习伴侣] 无法启动 Python (${py})：${err.message}\n` +
        "请检查 tfTutor.pythonPath 设置，或运行「环境自检与一键配置」。\n";
      resolve({ exitCode: -1, stdout, stderr, durationMs: 0, timedOut, newFigures: [] });
    });

    proc.on("close", (code) => {
      clearTimeout(timer);
      running = undefined;
      const durationMs = Date.now() - since;
      handlers.onStatus?.(false);
      const tail = flushNoise(nstate);
      for (const line of tail.shown) {
        handlers.onStderr?.(line + "\n");
      }
      if (nstate.total > 1) {
        handlers.onStderr?.(`[TF 学习伴侣] 共隐藏 ${nstate.total} 条无害的 TF 初始化日志\n`);
      }
      const figs = newFiguresSince(figuresDir, since);
      log(`run finished: code=${code} ${durationMs}ms figures=${figs.length} timedOut=${timedOut}`);
      resolve({ exitCode: code, stdout, stderr, durationMs, timedOut, newFigures: figs });
    });
  });
}

/** 同步版解释器探测（spawn 前调用） */
export function resolveInterpreterSync(): string {
  const configured = config.pythonPath;
  if (configured && fs.existsSync(configured)) {
    return configured;
  }
  const root = workspaceRoot();
  const candidates: string[] = [];
  if (root) {
    candidates.push(path.join(root, ".venv", "Scripts", "python.exe"));
    candidates.push(path.join(root, ".venv", "bin", "python"));
  }
  candidates.push("F:\\tf2-env\\Scripts\\python.exe");
  for (const c of candidates) {
    if (fs.existsSync(c)) {
      return c;
    }
  }
  return process.platform === "win32" ? "python" : "python3";
}

/** 解释当前报错：逐条匹配报错库的正则（匹配逻辑见 ./errorParser，可单元测试） */
export function matchError(context: vscode.ExtensionContext,
  stderr: string): { pattern: ErrorPattern; matched: string } | null {
  return matchAgainstPatterns(stderr, loadErrorPatterns(context));
}

/**
 * 解释器探测顺序（设计要点 2.10 / 3.7 跨平台）：
 * 设置 tfTutor.pythonPath → 工作区 .venv → F:\tf2-env（本项目约定）→ PATH python/python3
 */
export async function resolveInterpreter(): Promise<string> {
  const configured = config.pythonPath;
  if (configured && fs.existsSync(configured)) {
    return configured;
  }
  const root = workspaceRoot();
  const candidates: string[] = [];
  if (root) {
    candidates.push(path.join(root, ".venv", "Scripts", "python.exe"));
    candidates.push(path.join(root, ".venv", "bin", "python"));
  }
  candidates.push("F:\\tf2-env\\Scripts\\python.exe"); // tf2-tutorial 项目约定
  candidates.push("python3");
  candidates.push("python");
  for (const c of candidates) {
    if (c.includes("/") || c.includes("\\")) {
      if (fs.existsSync(c)) {
        return c;
      }
    } else if (await onPath(c)) {
      return c;
    }
  }
  return "python";
}

function onPath(cmd: string): Promise<boolean> {
  const probe = process.platform === "win32" ? "where" : "which";
  return new Promise((resolve) => {
    const p = spawn(probe, [cmd]);
    p.on("close", (code) => resolve(code === 0));
    p.on("error", () => resolve(false));
  });
}
