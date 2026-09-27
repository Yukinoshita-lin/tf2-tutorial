import * as vscode from "vscode";
import type { Chapter, ErrorPattern, ProgressData } from "../types";
import { workspaceRoot } from "../utils/config";

/**
 * 内容层：优先读取**工作区**里的教学仓库内容（手册生成的 chapters/*.json、报错库），
 * 工作区没有时回退到插件自带内容 —— 实现"手册与插件用同一份源文件"（设计要点·风险表）。
 */

function firstExisting(candidates: string[]): string | undefined {
  for (const c of candidates) {
    if (c && require("fs").existsSync(c)) {
      return c;
    }
  }
  return undefined;
}

export function extensionContentDir(context: vscode.ExtensionContext): string {
  return context.asAbsolutePath("content");
}

function chaptersDir(context: vscode.ExtensionContext): string {
  const root = workspaceRoot();
  return firstExisting([
    root ? require("path").join(root, "extension", "content", "chapters") : "",
    root ? require("path").join(root, "content", "chapters") : "",
    require("path").join(extensionContentDir(context), "chapters"),
  ]) ?? require("path").join(extensionContentDir(context), "chapters");
}

/** 章节地图（含依赖关系，供知识地图使用） */
export interface ChapterIndexEntry {
  id: string;
  num: string;
  title: string;
  entry: string;
  deps: string[];
}

export function loadChapterIndex(context: vscode.ExtensionContext): ChapterIndexEntry[] {
  const root = workspaceRoot();
  const idx = firstExisting([
    root ? require("path").join(root, "extension", "content", "chapters.json") : "",
    require("path").join(extensionContentDir(context), "chapters.json"),
  ]);
  if (idx) {
    try {
      return JSON.parse(require("fs").readFileSync(idx, "utf8"));
    } catch {
      /* fallthrough */
    }
  }
  return [];
}

export function loadChapter(context: vscode.ExtensionContext, id: string): Chapter | undefined {
  const file = require("path").join(chaptersDir(context), `${id}.json`);
  try {
    return JSON.parse(require("fs").readFileSync(file, "utf8"));
  } catch {
    return undefined;
  }
}

/** 读取手册 Markdown 的指定行区间（供 Webview 渲染"完整讲解"） */
export function loadManualRange(context: vscode.ExtensionContext,
  range?: [number, number]): string {
  const root = workspaceRoot();
  const manual = firstExisting([
    root ? require("path").join(root, ...config_manual().split("/")) : "",
    require("path").join(context.extensionPath, "content", "learning_handbook_zh.md"),
  ]);
  if (!manual) {
    return "（未找到学习手册：请打开 tf2-tutorial 工作区，或在设置中指定 tfTutor.manualPath）";
  }
  const lines = require("fs").readFileSync(manual, "utf8").split("\n");
  if (!range) {
    return lines.join("\n");
  }
  return lines.slice(range[0] - 1, range[1] - 1).join("\n");
}

function config_manual(): string {
  return vscode.workspace.getConfiguration("tfTutor").get<string>(
    "manualPath", "docs/learning_handbook_zh.md");
}

/** 报错库（工作区可覆盖插件自带版本，支持热更新） */
export function loadErrorPatterns(context: vscode.ExtensionContext): ErrorPattern[] {
  const root = workspaceRoot();
  const file = firstExisting([
    root ? require("path").join(root, "extension", "content", "errors", "errorPatterns.json") : "",
    root ? require("path").join(root, "content", "errors", "errorPatterns.json") : "",
    require("path").join(extensionContentDir(context), "errors", "errorPatterns.json"),
  ]);
  if (!file) {
    return [];
  }
  try {
    const parsed = JSON.parse(require("fs").readFileSync(file, "utf8"));
    return Array.isArray(parsed) ? parsed : parsed.patterns ?? [];
  } catch {
    return [];
  }
}

/** 学习进度（globalState 持久化，跨窗口保存） */
const PROGRESS_KEY = "tfTutor.progress";

export function loadProgress(context: vscode.ExtensionContext): ProgressData {
  const raw = context.globalState.get<ProgressData>(PROGRESS_KEY);
  if (raw && raw.version === 1) {
    return raw;
  }
  return { version: 1, chapters: {}, quizHistory: {} };
}

export function saveProgress(context: vscode.ExtensionContext, data: ProgressData): void {
  void context.globalState.update(PROGRESS_KEY, data);
}

export function defaultProgress(): { read: boolean; ran: boolean; modified: boolean; passed: boolean } {
  return { read: false, ran: false, modified: false, passed: false };
}
