import * as vscode from "vscode";
import * as path from "path";
import { spawn } from "child_process";
import { config, log, workspaceRoot } from "../utils/config";
import { resolveInterpreter } from "./executor";

/**
 * 环境自检与一键配置（设计要点 2.10）。
 * 自检项与 scripts/env_check.py 一致（单一事实来源：Python 端已实现详细检查，
 * 扩展负责调度 + 把结果解析成可操作的提示）。
 */

export interface EnvCheckResult {
  ok: boolean;
  lines: string[];
}

/** 运行项目自检脚本 scripts/env_check.py（若存在），返回解析后的结论 */
export async function runProjectEnvCheck(): Promise<EnvCheckResult | null> {
  const root = workspaceRoot();
  if (!root) {
    return null;
  }
  const script = path.join(root, "scripts", "env_check.py");
  if (!require("fs").existsSync(script)) {
    return null;
  }
  const py = await resolveInterpreter();
  log(`env_check via ${py}`);
  const lines: string[] = [];
  const code = await new Promise<number>((resolve) => {
    const p = spawn(py, [script], { cwd: root, env: { ...process.env, PYTHONIOENCODING: "utf-8" } });
    p.stdout.on("data", (d: Buffer) => lines.push(...d.toString("utf8").split("\n")));
    p.stderr.on("data", (d: Buffer) => lines.push(...d.toString("utf8").split("\n")));
    p.on("error", () => resolve(-1));
    p.on("close", (c) => resolve(c ?? -1));
  });
  const joined = lines.join("\n");
  return { ok: code === 0 && joined.includes("✅"), lines };
}

/** 依赖是否齐备（轻量探测：能否 import tensorflow） */
export async function probeTensorFlow(): Promise<{ installed: boolean; version?: string }> {
  const py = await resolveInterpreter();
  const out = await new Promise<string>((resolve) => {
    const p = spawn(py, ["-c", "import tensorflow as tf; print(tf.__version__)"], {
      env: { ...process.env, TF_CPP_MIN_LOG_LEVEL: "3" },
    });
    let acc = "";
    p.stdout.on("data", (d: Buffer) => (acc += d.toString()));
    p.stderr.on("data", () => undefined);
    p.on("close", () => resolve(acc.trim()));
    p.on("error", () => resolve(""));
  });
  return { installed: !!out, version: out || undefined };
}

/** 一键配置：缺 venv 则创建 → 安装依赖 → 提示选择解释器 → 复跑自检 */
export async function setupEnvironment(): Promise<string> {
  const root = workspaceRoot();
  if (!root) {
    void vscode.window.showErrorMessage("请先打开 tf2-tutorial 工作区文件夹。");
    return "no-workspace";
  }
  const fs = require("fs") as typeof import("fs");
  const venvDir = path.join(root, ".venv");
  const py = await resolveInterpreter();

  // 1. venv
  if (!fs.existsSync(path.join(venvDir, "Scripts", "python.exe")) &&
      !fs.existsSync(path.join(venvDir, "bin", "python"))) {
    const doCreate = await vscode.window.showInformationMessage(
      "未检测到工作区虚拟环境（.venv）。是否现在创建？（约 10 秒）", "创建", "跳过");
    if (doCreate !== "创建") {
      return "cancelled";
    }
    await vscode.window.withProgress(
      { location: vscode.ProgressLocation.Notification, title: "创建虚拟环境..." },
      () => new Promise<void>((resolve, reject) => {
        const p = spawn(py, ["-m", "venv", venvDir], { cwd: root });
        p.on("close", (c) => (c === 0 ? resolve() : reject(new Error(`venv 退出码 ${c}`))));
        p.on("error", reject);
      }));
  }

  // 2. 依赖
  const req = path.join(root, "requirements.txt");
  const venvPython = process.platform === "win32"
    ? path.join(venvDir, "Scripts", "python.exe")
    : path.join(venvDir, "bin", "python");
  if (fs.existsSync(req)) {
    const doInstall = await vscode.window.showInformationMessage(
      "是否在 .venv 中安装 requirements.txt 依赖？（TensorFlow 较大，可能需要几分钟）",
      "安装", "跳过");
    if (doInstall === "安装") {
      const terminal = vscode.window.createTerminal({ name: "TF 学习伴侣 · 配置环境", cwd: root });
      terminal.show();
      terminal.sendText(`"${venvPython}" -m pip install -r requirements.txt`);
      void vscode.window.showInformationMessage(
        "依赖安装已在终端中启动，完成后回到这里继续。");
      return "installing";
    }
  }

  // 3. 自检
  const result = await runProjectEnvCheck();
  if (result) {
    result.lines.forEach((l) => log(l));
    if (result.ok) {
      void vscode.window.showInformationMessage("✅ 环境自检通过：TF/Keras 版本与本项目全部章节兼容。");
    } else {
      void vscode.window.showWarningMessage(
        "⚠ 环境自检未通过——详见「TF 学习伴侣」输出面板，对照手册附录 D 处理。");
    }
  }
  return "done";
}

/** 轻量模式提示（需求分析：低配机器自动建议 EPOCHS=1） */
export function lightModeHint(): string {
  return config.lightMode
    ? "\n# [TF 学习伴侣·轻量模式] 已启用：建议把 EPOCHS 改为 1 先跑通流程。\n"
    : "";
}
