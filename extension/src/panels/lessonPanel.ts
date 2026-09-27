import * as vscode from "vscode";
import type { Chapter, ChapterProgress, ExtToWeb, WebToExt } from "../types";
import { defaultProgress, loadProgress, saveProgress } from "../data/content";
import { log } from "../utils/config";
import { renderMarkdown } from "../utils/miniMd";

/**
 * Webview 教学面板（设计要点 2.2）：五个 Tab —— 讲解 / 输出 / 图表 / 自测 / 进度。
 * 决策记录（DESIGN.md）：MVP 用 vanilla JS + 内置迷你 Markdown 渲染器，
 * 不引入 React/构建链，保证零依赖可编译；面板成长后再升级（V2 路线）。
 */

export class LessonPanel {
  private panel?: vscode.WebviewPanel;
  private chapter?: Chapter;
  private chapterMd = "";
  private progressMap = new Map<string, ChapterProgress>();
  private onMessage?: (msg: WebToExt) => void;

  constructor(private readonly context: vscode.ExtensionContext) {}

  setHandler(handler: (msg: WebToExt) => void): void {
    this.onMessage = handler;
  }

  get currentChapterId(): string | undefined {
    return this.chapter?.id;
  }

  isReady(): boolean {
    return !!this.panel;
  }

  dispose(): void {
    this.panel?.dispose();
    this.panel = undefined;
  }

  reveal(): void {
    if (this.panel) {
      this.panel.reveal(vscode.ViewColumn.Beside);
    }
  }

  /** 打开（或复用）面板并加载一章 */
  async showChapter(chapter: Chapter, markdown: string): Promise<void> {
    this.chapter = chapter;
    this.chapterMd = markdown;
    const progress = loadProgress(this.context);
    this.progressMap = new Map(Object.entries(progress.chapters));

    if (!this.panel) {
      this.panel = vscode.window.createWebviewPanel(
        "tfTutor.lesson", "TF 学习伴侣", vscode.ViewColumn.Beside,
        { enableScripts: true, localResourceRoots: [this.context.extensionUri] });
      this.panel.webview.html = this.html();
      this.panel.webview.onDidReceiveMessage((msg: WebToExt) => this.onMessage?.(msg));
      this.panel.onDidDispose(() => (this.panel = undefined));
      void this.post({ type: "ready" });
    }
    this.panel.reveal(vscode.ViewColumn.Beside);
    await this.post({
      type: "chapter",
      payload: {
        chapter, markdown,
        progress: this.progressMap.get(chapter.id) ?? defaultProgress(),
      },
    });
    this.markRead(chapter.id);
  }

  /** 输出流式追加（stdout/stderr） */
  appendOutput(stream: "stdout" | "stderr", text: string): void {
    void this.post({ type: "output", payload: { stream, text } });
  }

  setRunning(running: boolean, exitCode?: number, durationMs?: number): void {
    void this.post({ type: "status", payload: { running, exitCode, durationMs } });
  }

  showErrorCard(card: { pattern: import("../types").ErrorPattern; matched: string } | null): void {
    void this.post({ type: "errorCard", payload: card });
  }

  showImages(uris: { uri: string; name: string }[]): void {
    void this.post({ type: "images", payload: uris });
  }

  /** 让面板切换 Tab（例如运行后自动切到"输出"） */
  setTab(tab: string): void {
    void this.post({ type: "setTab", payload: { tab } });
  }

  /** 参数扫描结果表（图表 Tab 内追加显示） */
  showScanResults(results: { label: string; value: string; metrics: Record<string, number> }[]): void {
    void this.post({ type: "scanResult", payload: results });
    this.reveal();
  }

  async updateProgress(): Promise<void> {
    await this.post({ type: "progress", payload: loadProgress(this.context) });
  }

  private markRead(chapterId: string): void {
    const data = loadProgress(this.context);
    const cur = data.chapters[chapterId] ?? defaultProgress();
    if (!cur.read) {
      cur.read = true;
      data.chapters[chapterId] = cur;
      saveProgress(this.context, data);
    }
  }

  private async post(msg: ExtToWeb): Promise<void> {
    if (this.panel) {
      await this.panel.webview.postMessage(msg);
    } else {
      log(`panel closed, drop message: ${msg.type}`);
    }
  }

  // ------------------------------------------------------------------ HTML

  private html(): string {
    return LessonPanel.buildHtml(this.panel!.webview.cspSource);
  }

  /** 静态构建（可单元测试）：CSP/nonce/Tab 结构一次验证 */
  static buildHtml(cspSource: string): string {
    const nonce = Array.from({ length: 16 }, () =>
      Math.floor(Math.random() * 256).toString(16).padStart(2, "0")).join("");
    const css = LessonPanel.STYLES();
    const js = LessonPanel.FRONTEND_JS();
    const csp = `default-src 'none'; img-src ${cspSource} data:; style-src 'unsafe-inline'; script-src 'nonce-${nonce}';`;
    return `<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy" content="${csp}">
<title>TF 学习伴侣</title>
<style>${css}</style>
</head>
<body>
  <nav id="tabs">
    <button data-tab="lesson" class="active">📖 讲解</button>
    <button data-tab="output">🖥 输出</button>
    <button data-tab="charts">📈 图表</button>
    <button data-tab="quiz">✅ 自测</button>
    <button data-tab="progress">🗺 进度</button>
    <span id="runState"></span>
    <button id="stopBtn" title="停止运行">■</button>
  </nav>
  <main id="views"></main>
  <script nonce="${nonce}">${js}</script>
</body>
</html>`;
  }

  /** 前端脚本：Tab 切换 + 消息渲染 + 交互回传（vanilla，无外部依赖） */
  private static FRONTEND_JS(): string {
    return `
'use strict';
const vscode = acquireVsCodeApi();
const md = ${renderMarkdown.toString()};
globalThis.__tfTutorMd = md;
const views = document.getElementById('views');
const runState = document.getElementById('runState');
const stopBtn = document.getElementById('stopBtn');
let currentChapter = null;
let currentMd = '';
let progress = null;

/* ---------- Tab 管理 ---------- */
const viewState = {};
function setTab(tab) {
  document.querySelectorAll('#tabs button[data-tab]').forEach((b) =>
    b.classList.toggle('active', b.dataset.tab === tab));
  render(tab);
}
function render(tab) {
  views.innerHTML = '';
  const fn = { lesson: renderLesson, output: renderOutput, charts: renderCharts,
               quiz: renderQuiz, progress: renderProgress }[tab] || renderLesson;
  fn(views);
}
document.querySelectorAll('#tabs button[data-tab]').forEach((b) =>
  b.addEventListener('click', () => setTab(b.dataset.tab)));

/* ---------- 讲解 Tab ---------- */
function renderLesson(root) {
  if (!currentChapter) {
    root.innerHTML = '<p class="muted">请运行命令「打开当前章节讲解」，或从左侧知识地图选择一章。</p>';
    return;
  }
  const p = progress && progress.chapters[currentChapter.id] || {read:1,ran:0,modified:0,passed:0};
  root.innerHTML =
    '<div class="chipbar">' +
      chip(p.read, '已读') + chip(p.ran, '已跑') + chip(p.modified, '已改') + chip(p.passed, '已过关') +
    '</div>' +
    '<h2>' + esc(currentChapter.title) + '</h2>' +
    (currentChapter.entry ? '<p class="entry">入口脚本：<code>' + esc(currentChapter.entry) +
      '</code> <button class="mini" data-act="openEntry">打开</button>' +
      ' <button class="mini primary" data-act="runEntry">▶ 运行</button></p>' : '') +
    '<div class="box goal"><b>🎯 本节目标</b><p>' + esc(currentChapter.goals || '见下文') + '</p></div>' +
    '<div id="mdBody">' + md(currentMd) + '</div>' +
    '<div class="box try"><b>✋ 动手试一试（点击自动填入建议参数到当前编辑器）</b>' +
      '<ul>' + (currentChapter.experiments || []).map((t, i) =>
        '<li><button class="mini" data-exp="' + i + '">填入</button> ' + esc(t) + '</li>').join('') +
      '</ul></div>' +
    '<div class="box pit"><b>⚠️ 常见坑</b><ul>' +
      (currentChapter.pitfalls || []).map((t) => '<li>' + esc(t) + '</li>').join('') + '</ul></div>' +
    '<div class="box pass"><b>🏁 过关标准</b><p>' + esc(currentChapter.passCriteria || '') + '</p></div>' +
    '<button class="mini primary" data-act="quiz">✅ 开始本章过关自测</button>';
  root.querySelectorAll('[data-exp]').forEach((b) =>
    b.addEventListener('click', () =>
      vscode.postMessage({ type: 'experiment', payload: { text: currentChapter.experiments[+b.dataset.exp] } })));
  root.querySelectorAll('[data-act]').forEach((b) =>
    b.addEventListener('click', () => {
      if (b.dataset.act === 'openEntry') {
        vscode.postMessage({ type: 'openEntry', payload: { entry: currentChapter.entry } });
      } else if (b.dataset.act === 'runEntry') {
        vscode.postMessage({ type: 'run', payload: { kind: 'file' } });
        setTab('output');
      } else if (b.dataset.act === 'quiz') {
        setTab('quiz');
      }
    }));
}
function chip(on, label) {
  return '<span class="chip ' + (on ? 'on' : '') + '">' + (on ? '● ' : '○ ') + label + '</span>';
}

/* ---------- 输出 Tab ---------- */
const outBuf = []; // {s:'out'|'err', t} —— 按到达顺序交错渲染
function renderOutput(root) {
  root.innerHTML =
    '<div class="toolbar"><span id="statusText"></span></div>' +
    '<pre id="outPre" class="console"></pre>';
  refreshOutput();
}
function refreshOutput() {
  const pre = document.getElementById('outPre');
  if (!pre) { return; }
  const st = document.getElementById('statusText');
  if (st) { st.textContent = runState.textContent; }
  pre.innerHTML = outBuf.map((e) => e.s === 'err'
    ? '<span class="err">' + esc(e.t) + '</span>' : esc(e.t)).join('');
  pre.scrollTop = pre.scrollHeight;
}

/* ---------- 图表 Tab ---------- */
const imageUris = [];
function renderCharts(root) {
  if (!imageUris.length) {
    root.innerHTML = '<p class="muted">运行章节脚本后，figures/ 下新生成的图片会自动出现在这里。</p>';
    return;
  }
  root.innerHTML = imageUris.map((im) =>
    '<figure><img src="' + im.uri + '"><figcaption>' + esc(im.name) + '</figcaption></figure>').join('');
}

/* ---------- 自测 Tab ---------- */
function renderQuiz(root) {
  if (!currentChapter) { root.innerHTML = '<p class="muted">先选择一章。</p>'; return; }
  const quiz = currentChapter.quiz || [];
  if (!quiz.length) {
    root.innerHTML = '<p class="muted">本章暂无可判分的自测题（见手册过关自测的挑战题）。</p>';
    return;
  }
  root.innerHTML = '<h3>过关自测 · 基础题（先作答，再对答案自评）</h3>' +
    quiz.map((q, i) =>
      '<div class="quiz"><p><b>Q' + (i + 1) + '.</b> ' + esc(q.question) + '</p>' +
      '<button class="mini" data-q="' + i + '">显示参考答案</button>' +
      '<div class="answer" id="ans' + i + '" style="display:none">' + esc(q.answer) + '</div>' +
      '<div class="selfgrade" id="grade' + i + '" style="display:none">' +
      '<span>自评：</span><button class="mini ok" data-q="' + i + '" data-ok="1">✔ 我答对了</button>' +
      '<button class="mini bad" data-q="' + i + '" data-ok="0">✘ 需要重学</button></div></div>').join('') +
    '<p class="muted">全部自评"答对"即标记本章「已过关」，计入学习进度。</p>';
  root.querySelectorAll('[data-q]').forEach((b) =>
    b.addEventListener('click', () => {
      const i = +b.dataset.q;
      if (b.textContent.indexOf('显示') === 0) {
        document.getElementById('ans' + i).style.display = 'block';
        document.getElementById('grade' + i).style.display = 'block';
      } else {
        vscode.postMessage({ type: 'quizAnswer', payload:
          { chapterId: currentChapter.id, index: i, selfCorrect: b.dataset.ok === '1' } });
        b.disabled = true;
      }
    }));
}

/* ---------- 进度 Tab ---------- */
function renderProgress(root) {
  if (!progress) { root.innerHTML = '<p class="muted">暂无进度。</p>'; return; }
  const rows = Object.keys(progress.chapters).sort().map((id) => {
    const p = progress.chapters[id];
    const done = p.read + p.ran + p.modified + p.passed;
    return '<tr><td>' + esc(id) + '</td><td>' +
      ['read', 'ran', 'modified', 'passed'].map((k) =>
        '<span class="chip ' + (p[k] ? 'on' : '') + '">' + (p[k] ? '●' : '○') + '</span>').join('') +
      '</td><td>' + done + '/4</td></tr>';
  }).join('');
  root.innerHTML = '<h3>学习进度</h3><table class="ptable"><tr><th>章节</th><th>已读/已跑/已改/已过关</th><th>进度</th></tr>' +
    (rows || '<tr><td colspan="3" class="muted">从知识地图选择一章开始学习</td></tr>') + '</table>';
}

/* ---------- 消息处理 ---------- */
window.addEventListener('message', (e) => {
  const msg = e.data;
  switch (msg.type) {
    case 'chapter': {
      currentChapter = msg.payload.chapter;
      currentMd = msg.payload.markdown || '';
      progress = msg.payload.progress ? { chapters: { [currentChapter.id]: msg.payload.progress } } : progress;
      outBuf.length = 0; imageUris.length = 0;
      const active = document.querySelector('#tabs button.active');
      render(active ? active.dataset.tab : 'lesson');
      break;
    }
    case 'output': {
      outBuf.push({ s: msg.payload.stream === 'stderr' ? 'err' : 'out', t: msg.payload.text });
      refreshOutput();
      break;
    }
    case 'status': {
      runState.textContent = msg.payload.running
        ? '▶ 运行中…'
        : (msg.payload.exitCode !== undefined
          ? (msg.payload.exitCode === 0 ? '✔ 完成 (' + Math.round(msg.payload.durationMs / 1000) + 's)'
                                         : '✘ 退出码 ' + msg.payload.exitCode)
          : '');
      runState.className = msg.payload.running ? 'running' : (msg.payload.exitCode === 0 ? 'ok' : 'bad');
      stopBtn.style.display = msg.payload.running ? 'inline-block' : 'none';
      break;
    }
    case 'errorCard': {
      const c = msg.payload;
      if (!c) { break; }
      const el = document.createElement('div');
      el.className = 'errorCard';
      el.innerHTML =
        '<b>❌ ' + esc(c.pattern.title) + '</b>' +
        '<p>📖 ' + esc(c.pattern.explanation) + '</p>' +
        '<p><b>🔍 最可能的原因</b></p><ul>' + c.pattern.causes.map((x) => '<li>' + esc(x) + '</li>').join('') + '</ul>' +
        '<p><b>💡 修复建议</b></p><ul>' + c.pattern.fixes.map((x) => '<li>' + esc(x) + '</li>').join('') + '</ul>' +
        '<p>📚 ' + esc(c.pattern.relatedChapter) + '</p>' +
        '<button class="mini" id="searchBtn">🔎 搜索此报错</button>';
      const active = document.querySelector('#tabs button.active');
      if (active && active.dataset.tab !== 'output') { setTab('output'); }
      const pre = document.getElementById('outPre');
      if (pre) { pre.insertAdjacentElement('afterend', el); }
      el.querySelector('#searchBtn').addEventListener('click', () =>
        vscode.postMessage({ type: 'searchError', payload: { text: c.matched } }));
      break;
    }
    case 'images': {
      for (const im of msg.payload) { imageUris.push(im); }
      break;
    }
    case 'scanResult': {
      const rows = msg.payload.map((r) =>
        '<tr><td>' + esc(r.label) + '</td><td>' +
        Object.keys(r.metrics).map((k) =>
          k + '=' + r.metrics[k]).join('　') + '</td></tr>').join('');
      imageUris.length = 0; // 切到图表 Tab
      const active = document.querySelector('#tabs button.active');
      if (active) { setTab('charts'); }
      views.innerHTML = '<h3>参数扫描结果</h3>' +
        '<table class="ptable"><tr><th>取值</th><th>最终指标（来自脚本输出）</th></tr>' +
        (rows || '') + '</table>' +
        '<p class="muted">指标自动从脚本输出解析（val_acc / accuracy / loss / F1 的最后一次取值）。</p>';
      break;
    }
    case 'setTab': {
      setTab(msg.payload.tab);
      break;
    }
    case 'progress': {
      progress = msg.payload;
      const active = document.querySelector('#tabs button.active');
      if (active && active.dataset.tab === 'progress') { render('progress'); }
      break;
    }
  }
});

stopBtn.addEventListener('click', () => vscode.postMessage({ type: 'stop' }));
vscode.postMessage({ type: 'ready' });
`;
  }

  /** 样式：跟随 VS Code 主题变量，深浅色自适应（设计要点测试项） */
  private static STYLES(): string {
    return `
    body { font-family: var(--vscode-font-family); color: var(--vscode-foreground);
           background: var(--vscode-editor-background); margin: 0; font-size: 13px; }
    #tabs { position: sticky; top: 0; background: var(--vscode-editor-background);
            border-bottom: 1px solid var(--vscode-panel-border); padding: 6px 8px; z-index: 5; }
    #tabs button { background: transparent; color: var(--vscode-foreground);
                   border: none; padding: 4px 10px; cursor: pointer; border-radius: 4px; }
    #tabs button.active { background: var(--vscode-button-background); color: var(--vscode-button-foreground); }
    #runState { margin-left: 10px; font-weight: bold; }
    #runState.running { color: var(--vscode-charts-yellow); }
    #runState.ok { color: var(--vscode-charts-green); }
    #runState.bad { color: var(--vscode-charts-red); }
    #stopBtn { display: none; color: var(--vscode-charts-red); font-weight: bold; }
    main { padding: 10px 14px; line-height: 1.55; }
    h2, h3, h4 { color: var(--vscode-symbolIcon-classForeground); }
    pre { background: var(--vscode-textCodeBlock-background); padding: 8px 10px;
          border-radius: 6px; overflow-x: auto; max-height: 420px; overflow-y: auto;
          font-size: 12px; }
    code { background: var(--vscode-textCodeBlock-background); padding: 1px 5px; border-radius: 3px; }
    blockquote { border-left: 3px solid var(--vscode-charts-purple); margin: 6px 0;
                 padding: 2px 10px; color: var(--vscode-descriptionForeground); }
    .box { border: 1px solid var(--vscode-panel-border); border-radius: 8px;
           padding: 8px 12px; margin: 10px 0; }
    .box.goal { border-color: var(--vscode-charts-blue); }
    .box.try { border-color: var(--vscode-charts-green); }
    .box.pit { border-color: var(--vscode-charts-orange); }
    .box.pass { border-color: var(--vscode-charts-purple); }
    .box p { margin: 4px 0; }
    .chip { display: inline-block; border: 1px solid var(--vscode-panel-border);
            border-radius: 999px; padding: 1px 10px; margin-right: 6px; font-size: 12px;
            color: var(--vscode-descriptionForeground); }
    .chip.on { background: var(--vscode-charts-green); color: #fff; border-color: transparent; }
    .mini { background: var(--vscode-button-secondaryBackground); color: var(--vscode-button-secondaryForeground);
            border: 1px solid var(--vscode-panel-border); border-radius: 4px;
            padding: 2px 10px; cursor: pointer; margin: 2px 4px 2px 0; font-size: 12px; }
    .mini.primary { background: var(--vscode-button-background); color: var(--vscode-button-foreground); }
    .mini.ok { border-color: var(--vscode-charts-green); }
    .mini.bad { border-color: var(--vscode-charts-red); }
    .entry { background: var(--vscode-textCodeBlock-background); border-radius: 6px; padding: 6px 10px; }
    .errorCard { border: 2px solid var(--vscode-charts-red); border-radius: 8px;
                 padding: 10px 14px; margin: 10px 0; background: rgba(220,38,38,0.06); }
    .errorCard ul { margin: 4px 0; }
    .console { max-height: 60vh; }
    .err { color: var(--vscode-charts-red); }
    figure { margin: 10px 0; text-align: center; }
    figure img { max-width: 100%; border-radius: 6px; }
    figcaption { color: var(--vscode-descriptionForeground); font-size: 12px; }
    .ptable { border-collapse: collapse; width: 100%; }
    .ptable td, .ptable th { border: 1px solid var(--vscode-panel-border); padding: 4px 10px; }
    .trow { display: flex; border-bottom: 1px solid var(--vscode-panel-border); }
    .tcell { flex: 1; padding: 4px 8px; border-right: 1px solid var(--vscode-panel-border); }
    .quiz { border: 1px solid var(--vscode-panel-border); border-radius: 8px; padding: 8px 12px; margin: 8px 0; }
    .answer { background: var(--vscode-textCodeBlock-background); border-radius: 6px; padding: 6px 10px; margin: 6px 0; }
    .muted { color: var(--vscode-descriptionForeground); }
    .link { color: var(--vscode-textLink-foreground); }`;
  }
}
