import * as vscode from "vscode";
import { LessonPanel } from "./panels/lessonPanel";
import { KnowledgeMapProvider, ProgressProvider } from "./tree/knowledgeMap";
import { initCommands, registerAll } from "./commands";
import { log } from "./utils/config";
import type { WebToExt } from "./types";

/**
 * TF 学习伴侣 —— 入口。
 * 职责：注册命令 / 视图 / Webview 面板；把 Webview 消息路由到命令层。
 */

export function activate(context: vscode.ExtensionContext): void {
  log("activate");
  initCommands(context);

  const panel = new LessonPanel(context);
  const km = new KnowledgeMapProvider(context);
  const pg = new ProgressProvider(context);

  context.subscriptions.push(
    vscode.window.registerTreeDataProvider("tfTutor.knowledgeMap", km),
    vscode.window.registerTreeDataProvider("tfTutor.progress", pg),
    panel,
  );

  registerAll(context, panel, km, pg);

  // Webview 消息里有 openLesson 等，需要命令已注册——handler 在 registerAll 内挂接。
  // 监听章节脚本保存：若为章节文件则刷新知识地图（进度"已改"提示）。
  context.subscriptions.push(vscode.workspace.onDidSaveTextDocument((doc) => {
    if (doc.uri.fsPath.includes("chapters") && doc.uri.fsPath.endsWith(".py")) {
      km.refresh();
      pg.refresh();
    }
  }));

  // 面板关闭后再打开时需要重新挂 handler：LessonPanel 内部管理。
  // 这里兜底：激活完成打一条日志，方便排障。
  const handler: (msg: WebToExt) => void = () => undefined;
  void handler;

  void vscode.window.showInformationMessage(
    "TF 学习伴侣已就绪：命令面板运行「TF 学习伴侣: 打开当前章节讲解」开始学习。");
}

export function deactivate(): void {
  log("deactivate");
}
