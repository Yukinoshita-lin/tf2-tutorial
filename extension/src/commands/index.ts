import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import * as vscode from "vscode";
import type { WebToExt } from "../types";
import { LessonPanel } from "../panels/lessonPanel";
import { KnowledgeMapProvider, ProgressProvider } from "../tree/knowledgeMap";
import {
  isRunning, matchError, runPython, runPythonFile, stopRun,
} from "../python/executor";
import { lightModeHint } from "../python/env";
import { extractCellRange, extractMetrics } from "../utils/extract";
import {
  loadChapter, loadChapterIndex, loadManualRange, loadProgress,
} from "../data/content";
import { config, getLogger, log, workspaceRoot } from "../utils/config";

/** 从编辑器提取代码：整文件 / # %% 单元格。
 *  优先活动编辑器；面板抢焦点时回退到任何可见的 .py 编辑器。 */
function activePythonEditor(): vscode.TextEditor | undefined {
  const active = vscode.window.activeTextEditor;
  if (active && active.document.languageId === "python") {
    return active;
  }
  return vscode.window.visibleTextEditors.find(
    (e) => e.document.languageId === "python");
}

export function extractCode(kind: "file" | "cell"): { code: string; file: string } | undefined {
  const editor = activePythonEditor();
  if (!editor) {
    return undefined;
  }
  const file = editor.document.uri.fsPath;
  if (kind === "file") {
    return { code: editor.document.getText(), file };
  }
  if (kind === "cell") {
    const cursor = editor.selection.active.line;
    const lines = editor.document.getText().split("\n");
    const [start, end] = extractCellRange(lines, cursor);
    const cellCode = [
      "# [TF 学习伴侣] 单元格自动附带：项目路径与 tf2tutorial 导入",
      "import sys, os",
      `os.chdir(${JSON.stringify(workspaceRoot() ?? path.dirname(file))})`,
      `sys.path.insert(0, ${JSON.stringify(path.join(workspaceRoot() ?? "", "src"))})`,
      ...lines.slice(start, end),
    ].join("\n");
    return { code: cellCode, file };
  }
  return undefined;
}

/** 选中代码提取（带同样的前导，保证 import tf2tutorial 可用） */
export function extractSelection(): { code: string; file: string } | undefined {
  const editor = vscode.window.activeTextEditor;
  if (!editor || editor.document.languageId !== "python" || editor.selection.isEmpty) {
    void vscode.window.showWarningMessage("请先选中要运行的代码。");
    return undefined;
  }
  const prologue = [
    "import sys, os",
    `os.chdir(${JSON.stringify(workspaceRoot() ?? path.dirname(editor.document.uri.fsPath))})`,
    `sys.path.insert(0, ${JSON.stringify(path.join(workspaceRoot() ?? "", "src"))})`,
  ].join("\n");
  return { code: prologue + "\n" + editor.document.getText(editor.selection), file: editor.document.uri.fsPath };
}

/** 统一的"运行"入口：输出到 Webview，结束后做报错翻译 + 图表捕获。
 *  文件运行优先取编辑器内容；面板抢焦点/无编辑器时回退到当前章节入口脚本。 */
export async function runCode(
  panel: LessonPanel, kind: "file" | "cell" | "selection"): Promise<void> {
  if (isRunning()) {
    void vscode.window.showWarningMessage("已有脚本在运行，请先停止（面板顶部 ■ 按钮）。");
    return;
  }
  panel.reveal();
  panel.setTab("output");
  panel.setRunning(true);
  panel.appendOutput("stdout", `$ python … ${kind} · ${new Date().toLocaleTimeString()}\n${lightModeHint()}`);
  // 降级保障：webview 加载失败时，输出仍可在「输出」通道查看
  const chan = getLogger();
  chan.appendLine(`---- run ${kind} @ ${new Date().toLocaleTimeString()} ----`);
  const tee = (stream: "stdout" | "stderr") => (t: string) => {
    chan.appendLine(t.endsWith("\n") ? t : t + "\n");
    panel.appendOutput(stream, t);
  };

  const outTee = tee("stdout");
  const errTee = tee("stderr");

  let result: import("../python/executor").RunResult;
  if (kind === "selection") {
    const sel = extractSelection();
    if (!sel) {
      panel.setRunning(false);
      return;
    }
    if (config.autoSave) {
      await vscode.window.activeTextEditor?.document.save();
    }
    result = await runPython(sel.code, {
      onStdout: outTee, onStderr: errTee,
    });
  } else {
    const extracted = extractCode(kind === "file" ? "file" : "cell");
    const root = workspaceRoot();
    if (extracted) {
      if (config.autoSave) {
        await activePythonEditor()?.document.save();
      }
      result = await runPython(extracted.code, {
        onStdout: outTee, onStderr: errTee,
      });
    } else {
      // 回退：直接运行当前章节的入口脚本（无需编辑器焦点）
      const id = panel.currentChapterId;
      const entry = id
        ? loadChapterIndex(globalContext).find((e) => e.id === id)?.entry
        : undefined;
      const entryAbs = entry && root ? path.join(root, entry) : undefined;
      if (!entryAbs || !fs.existsSync(entryAbs)) {
        panel.setRunning(false);
        void vscode.window.showWarningMessage(
          "未找到可运行的代码：请打开 chapters/ 下的 .py 文件，或从知识地图先选择一章。");
        return;
      }
      panel.appendOutput("stdout", `$ python ${entry}   （当前章节入口脚本）\n`);
      result = await runPythonFile(entryAbs, {
        onStdout: (t) => panel.appendOutput("stdout", t),
        onStderr: (t) => panel.appendOutput("stderr", t),
      });
    }
  }
  panel.setRunning(false, result.exitCode ?? -1, result.durationMs);
  chan.appendLine(`---- exit ${result.exitCode ?? "?"} · ${result.durationMs}ms ----`);

  // 图表捕获（需求分析：figures/ 新图自动展示）
  const root = workspaceRoot();
  if (result.newFigures.length && root) {
    panel.showImages(result.newFigures.map((f) => ({
      name: path.basename(f),
      uri: String(vscode.Uri.file(f)).replace(/^file:\/\//, ""),
    })));
  }

  // 报错翻译（设计要点 2.4：命中→翻译卡片；未命中→搜索按钮）
  if (result.exitCode !== 0 && !result.timedOut && result.stderr) {
    panel.showErrorCard(matchError(globalContext, result.stderr));
  } else {
    panel.showErrorCard(null);
  }

  // 进度标记"已跑"
  if (result.exitCode === 0) {
    const file = activePythonEditor()?.document.uri.fsPath
      ?? (panel.currentChapterId
        ? path.join(workspaceRoot() ?? "",
            loadChapterIndex(globalContext).find((e) => e.id === panel.currentChapterId)?.entry ?? "")
        : "");
    if (file && fs.existsSync(file)) {
      markProgressFlag(file, "ran");
    }
  }
}

let globalContext: vscode.ExtensionContext;

export function initCommands(context: vscode.ExtensionContext): void {
  globalContext = context;
}

/** 根据运行文件路径推断章节 id（chapters/NN_xxx.py → NN_xxx） */
function chapterIdFromFile(file: string): string | undefined {
  const base = path.basename(file, ".py");
  return loadChapterIndex(globalContext).some((e) => e.id === base) ? base : undefined;
}

/** 进度标记工具 */
export function markProgressFlag(file: string, flag: "ran" | "modified"): void {
  const id = chapterIdFromFile(file);
  if (!id) {
    return;
  }
  const data = loadProgress(globalContext);
  const cur = data.chapters[id] ?? { read: false, ran: false, modified: false, passed: false };
  cur[flag] = true;
  data.chapters[id] = cur;
  void globalContext.globalState.update("tfTutor.progress", data);
  log(`progress: ${id}.${flag}=true`);
}

/** 注册全部命令 */
export function registerAll(
  context: vscode.ExtensionContext,
  panel: LessonPanel,
  km: KnowledgeMapProvider,
  pg: ProgressProvider,
): void {
  const reg = (id: string, fn: (...args: unknown[]) => unknown) =>
    context.subscriptions.push(vscode.commands.registerCommand(id, fn));

  reg("tfTutor.runFile", () => runCode(panel, "file"));
  reg("tfTutor.runCell", () => runCode(panel, "cell"));
  reg("tfTutor.runSelection", () => runCode(panel, "selection"));
  reg("tfTutor.stopRun", () => stopRun());

  // 打开讲解：无参 → 按当前文件推断章节；有参 → 指定章节 id
  reg("tfTutor.openLesson", async (...args: unknown[]) => {
    const chapterId = args[0] as string | undefined;
    const id = chapterId ?? chapterIdFromFile(
      vscode.window.activeTextEditor?.document.uri.fsPath ?? "") ??
      loadChapterIndex(context)[1]?.id; // 默认第 1 章
    if (!id) {
      void vscode.window.showWarningMessage("未找到章节内容（content/chapters/*.json）。");
      return;
    }
    const chapter = loadChapter(context, id);
    if (!chapter) {
      void vscode.window.showWarningMessage(`章节内容缺失：${id}（运行 scripts/generate_content.py 重新生成）`);
      return;
    }
    await panel.showChapter(chapter, loadManualRange(context, chapter.mdRange));
    km.refresh();
    pg.refresh();
  });

  reg("tfTutor.showOutput", () => panel.reveal());

  // 参数扫描（设计要点 2.5 的 MVP 版：文本替换 + 逐值运行 + 指标提取）
  reg("tfTutor.scanParams", async () => {
    const editor = vscode.window.activeTextEditor;
    if (!editor || editor.selection.isEmpty) {
      void vscode.window.showWarningMessage("请先选中一行参数赋值，例如：learning_rate = 0.1");
      return;
    }
    const selText = editor.document.getText(editor.selection).trim();
    const m = selText.match(/^([A-Za-z_]\w*)\s*=\s*([-+0-9.eE]+)$/);
    if (!m) {
      void vscode.window.showWarningMessage("选中的内容不是形如 `name = 数值` 的赋值。");
      return;
    }
    const valuesInput = await vscode.window.showInputBox({
      prompt: `参数扫描：${m[1]} 的取值列表（逗号分隔）`,
      value: "0.001,0.01,0.1,1.0",
    });
    if (!valuesInput) {
      return;
    }
    const values = valuesInput.split(",").map((v) => v.trim()).filter(Boolean);
    const full = editor.document.getText();
    const results: { label: string; value: string; metrics: Record<string, number> }[] = [];
    await vscode.window.withProgress(
      { location: vscode.ProgressLocation.Notification,
        title: `参数扫描：${m[1]} × ${values.length}` },
      async (progress) => {
        for (let i = 0; i < values.length; i++) {
          progress.report({ message: `${m[1]}=${values[i]} (${i + 1}/${values.length})` });
          const variant = full.replace(
            new RegExp(`(${m[1]}\\s*=\\s*)([-+0-9.eE]+)`), `$1${values[i]}`);
          const tmp = path.join(os.tmpdir(), `tftutor_scan_${Date.now()}_${i}.py`);
          fs.writeFileSync(tmp, variant, "utf8");
          const r = await runPython(fs.readFileSync(tmp, "utf8"), {});
          fs.rmSync(tmp, { force: true });
          const metrics = extractMetrics(r.stdout);
          results.push({ label: `${m[1]}=${values[i]}`, value: values[i], metrics });
        }
      });
    panel.reveal();
    panel.showScanResults(results);
    // 扫描也会推进当前章节的"已改"
    markProgressFlag(editor.document.uri.fsPath, "modified");
  });

  reg("tfTutor.explainError", async () => {
    const clip = await vscode.env.clipboard.readText();
    const card = clip ? matchError(context, clip) : null;
    if (card) {
      panel.reveal();
      panel.showErrorCard(card);
    } else {
      void vscode.window.showInformationMessage(
        "剪贴板中没有可识别的报错。运行脚本报错后会自动弹出翻译卡片（未命中时也提供搜索按钮）。");
    }
  });

  reg("tfTutor.takeQuiz", () => vscode.commands.executeCommand("tfTutor.openLesson")
    .then(() => panel.reveal()));

  reg("tfTutor.showProgress", () => {
    panel.reveal();
    void panel.updateProgress();
    pg.refresh();
  });

  reg("tfTutor.setupEnv", async () => {
    const { setupEnvironment } = await import("../python/env");
    await setupEnvironment();
  });

  reg("tfTutor.openManual", async () => {
    const root = workspaceRoot();
    if (!root) {
      return;
    }
    const candidates = [config.manualPath, "docs/learning_handbook_zh.pdf"];
    for (const c of candidates) {
      const f = path.join(root, c);
      if (fs.existsSync(f)) {
        await vscode.commands.executeCommand("vscode.open", vscode.Uri.file(f),
          { viewColumn: vscode.ViewColumn.One });
        return;
      }
    }
    void vscode.window.showWarningMessage("未找到学习手册（tfTutor.manualPath）。");
  });

  reg("tfTutor.resetChapter", async () => {
    const id = panel.currentChapterId;
    if (!id) {
      return;
    }
    const data = loadProgress(context);
    delete data.chapters[id];
    void context.globalState.update("tfTutor.progress", data);
    km.refresh();
    pg.refresh();
    void vscode.window.showInformationMessage(`已重置 ${id} 的学习进度。`);
  });

  // Webview → 扩展 的消息
  panel.setHandler(async (msg: WebToExt) => {
    switch (msg.type) {
      case "run":
        await vscode.commands.executeCommand(
          msg.payload.kind === "file" ? "tfTutor.runFile" : "tfTutor.runCell");
        break;
      case "stop":
        stopRun();
        break;
      case "experiment": {
        // "动手试一试"：把建议参数作为注释插入光标处，鼓励读者照着改
        const editor = vscode.window.activeTextEditor;
        if (editor) {
          await editor.edit((eb) =>
            eb.insert(editor.selection.end, `\n# [动手试一试] ${msg.payload.text}\n`));
          void vscode.window.showTextDocument(editor.document);
        }
        markProgressFlag(editor?.document.uri.fsPath ?? "", "modified");
        break;
      }
      case "quizAnswer": {
        const data = loadProgress(context);
        const { chapterId, index, selfCorrect } = msg.payload;
        const hist = data.quizHistory[chapterId] ??
          { correct: 0, total: 0, at: "" };
        if (index === 0) {
          hist.correct = 0;
          hist.total = 0;
        }
        hist.total += 1;
        hist.correct += selfCorrect ? 1 : 0;
        hist.at = new Date().toISOString();
        data.quizHistory[chapterId] = hist;
        const cur = data.chapters[chapterId] ??
          { read: false, ran: false, modified: false, passed: false };
        cur.modified = true;
        const chapter = loadChapter(context, chapterId);
        if (chapter && index + 1 >= chapter.quiz.length && hist.correct === hist.total) {
          cur.passed = true;
          void vscode.window.showInformationMessage(`🎉 ${chapterId} 已过关！`);
        }
        data.chapters[chapterId] = cur;
        void context.globalState.update("tfTutor.progress", data);
        km.refresh();
        pg.refresh();
        void panel.updateProgress();
        break;
      }
      case "openEntry": {
        // 讲解面板「打开」按钮：真正打开入口脚本文件
        const root = workspaceRoot();
        const f = path.join(root ?? "", msg.payload.entry);
        if (fs.existsSync(f)) {
          await vscode.window.showTextDocument(vscode.Uri.file(f),
            { viewColumn: vscode.ViewColumn.One });
        } else {
          void vscode.window.showWarningMessage(
            `入口脚本不存在：${msg.payload.entry}。请用「文件 > 打开文件夹」打开 tf2-tutorial ` +
            `项目根（含 chapters/ 的目录），或先打开 chapters/ 下的任一脚本后再试。`);
        }
        break;
      }
      case "markRead":
      case "setTab":
      case "openChapter":
      case "searchError":
        if (msg.type === "searchError") {
          await vscode.env.openExternal(vscode.Uri.parse(
            `https://www.google.com/search?q=${encodeURIComponent(
              "tensorflow " + msg.payload.text.slice(0, 200))}`));
        } else if (msg.type === "openChapter") {
          await vscode.commands.executeCommand("tfTutor.openLesson", msg.payload.chapterId);
        }
        break;
    }
  });
}
