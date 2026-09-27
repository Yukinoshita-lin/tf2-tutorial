import * as vscode from "vscode";
import * as fs from "fs";
import * as path from "path";
import { findRepoRoot } from "./repoRoot";

let channel: vscode.OutputChannel | undefined;

export function getLogger(): vscode.OutputChannel {
  if (!channel) {
    channel = vscode.window.createOutputChannel("TF 学习伴侣");
  }
  return channel;
}

export function log(msg: string): void {
  getLogger().appendLine(`[${new Date().toLocaleTimeString()}] ${msg}`);
}

/**
 * 项目根探测（缓存）：
 * 1) 工作区文件夹里像仓库根的（含 chapters/ 或手册）；
 * 2) 从已打开文件的路径向上找仓库根——支持"只打开一个 .py 文件、未开工作区"的单文件模式；
 * 3) 回退到第一个工作区文件夹。
 */
let cachedRoot: string | undefined;

function isRepoRoot(dir: string): boolean {
  try {
    return fs.existsSync(path.join(dir, "chapters")) ||
      fs.existsSync(path.join(dir, "docs", "learning_handbook_zh.md"));
  } catch {
    return false;
  }
}

export function projectRoot(): string | undefined {
  if (cachedRoot) {
    return cachedRoot;
  }
  // 1) 工作区文件夹
  for (const f of vscode.workspace.workspaceFolders ?? []) {
    const p = f.uri.fsPath;
    if (isRepoRoot(p)) {
      cachedRoot = p;
      return cachedRoot;
    }
  }
  // 2) 从打开的文件向上找
  const files = [
    ...(vscode.window.activeTextEditor
      ? [vscode.window.activeTextEditor.document.uri.fsPath]
      : []),
    ...vscode.window.visibleTextEditors.map((e) => e.document.uri.fsPath),
  ];
  for (const f of files) {
    const root = findRepoRoot(path.dirname(f), isRepoRoot);
    if (root) {
      cachedRoot = root;
      log(`project root detected from open file: ${root}`);
      return cachedRoot;
    }
  }
  // 3) 回退：第一个工作区文件夹
  const wf = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
  if (wf) {
    cachedRoot = wf;
  }
  return cachedRoot;
}

/** 兼容旧调用点：语义即"项目根" */
export function workspaceRoot(): string | undefined {
  return projectRoot();
}

/** 读取插件配置 */
export const config = {
  get pythonPath(): string {
    return vscode.workspace.getConfiguration("tfTutor").get<string>("pythonPath", "");
  },
  get timeoutSec(): number {
    return vscode.workspace.getConfiguration("tfTutor").get<number>("timeoutSec", 300);
  },
  get autoSave(): boolean {
    return vscode.workspace.getConfiguration("tfTutor").get<boolean>("autoSaveBeforeRun", true);
  },
  get lightMode(): boolean {
    return vscode.workspace.getConfiguration("tfTutor").get<boolean>("lightMode", false);
  },
  get manualPath(): string {
    return vscode.workspace.getConfiguration("tfTutor").get<string>(
      "manualPath", "docs/learning_handbook_zh.md");
  },
};
