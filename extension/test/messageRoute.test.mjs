import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);

/**
 * Webview → 扩展 消息路由回归测试（stub `vscode` 模块，无需启动 VS Code）。
 * 背景：0.1.1 曾漏掉 openEntry 的 case，导致"打开"按钮点击后毫无反应——
 * 本测试保证 WebToEx 协议里的每个消息类型都有路由行为。
 *
 * 注意：模块级单次 hook + 单次 require（模块缓存共享实例），
 * 各测试只重置计数器，避免多 stub 错位。
 */

const here = dirname(fileURLToPath(import.meta.url));
const EXT = join(here, "..");
const ROOT = join(EXT, "..");

const calls = {
  commands: new Map(),
  showTextDocument: [],
  warnings: [],
  infos: [],
  executed: [],
};
const vscodeStub = {
  commands: {
    registerCommand: (id, fn) => { calls.commands.set(id, fn); },
    executeCommand: async (id, ...args) => { calls.executed.push({ id, args }); return undefined; },
  },
  window: {
    showTextDocument: async (uri, opts) => { calls.showTextDocument.push({ uri, opts }); return {}; },
    showWarningMessage: async (m) => { calls.warnings.push(m); return undefined; },
    showInformationMessage: async (m) => { calls.infos.push(m); return undefined; },
    createWebviewPanel: () => { throw new Error("panel must be injected as stub"); },
    createOutputChannel: () => ({ appendLine() { }, show() { } }),
    activeTextEditor: undefined,
    visibleTextEditors: [],
  },
  workspace: {
    workspaceFolders: [{ uri: { fsPath: ROOT } }],
    getConfiguration: () => ({ get: (_k, d) => d }),
    onDidSaveTextDocument: () => ({ dispose() { } }),
  },
  extensions: { getExtension: () => undefined },
  env: {
    openExternal: async () => undefined,
    clipboard: { readText: async () => "" },
  },
  Uri: { file: (p) => ({ fsPath: p, path: p, toString: () => p }) },
  ViewColumn: { One: 1, Beside: 2, Active: -1 },
  ThemeIcon: class { constructor(name) { this.name = name; } },
  ThemeColor: class { constructor(c) { this.color = c; } },
  MarkdownString: class { constructor(v) { this.value = v; } },
  TreeItem: class { constructor(label) { this.label = label; } },
  TreeItemCollapsibleState: { None: 0, Collapsed: 1 },
  ProgressLocation: { Notification: 15 },
};

const Module = require("node:module");
const origLoad = Module._load;
Module._load = function (request, parent, isMain) {
  if (request === "vscode") { return vscodeStub; }
  return origLoad.apply(this, arguments);
};

const store = new Map();
const context = {
  extensionPath: EXT,
  asAbsolutePath: (p) => join(EXT, p),
  globalState: { get: (k) => store.get(k), update: async (k, v) => { store.set(k, v); } },
  workspaceState: { get: () => undefined, update: async () => { } },
  subscriptions: { push: () => { } },
};
const panelStub = {
  handler: null,
  setHandler(fn) { this.handler = fn; },
  reveal() { }, setRunning() { }, appendOutput() { }, setTab() { },
  showErrorCard() { }, showImages() { }, updateProgress: async () => { },
  currentChapterId: "01_tensors_autograd",
};
const kmStub = { refresh() { } };
const pgStub = { refresh() { } };

const cmd = require(join(EXT, "out", "commands", "index.js"));
cmd.initCommands(context);
cmd.registerAll(context, panelStub, kmStub, pgStub);
Module._load = origLoad; // 模块已加载完毕，还原 hook

function resetCounters() {
  calls.showTextDocument.length = 0;
  calls.warnings.length = 0;
  calls.infos.length = 0;
}

test("全部 13 条命令均已注册", () => {
  const pkg = JSON.parse(readFileSync(join(EXT, "package.json"), "utf8"));
  for (const c of pkg.contributes.commands) {
    assert.ok(calls.commands.has(c.command), `命令未注册: ${c.command}`);
  }
  assert.ok(typeof panelStub.handler === "function", "消息 handler 未挂接");
});

test("openEntry：打开按钮 → showTextDocument 打开入口脚本（0.1.1 回归）", async () => {
  resetCounters();
  await panelStub.handler({
    type: "openEntry", payload: { entry: "chapters/01_tensors_autograd.py" },
  });
  assert.equal(calls.showTextDocument.length, 1, "showTextDocument 未被调用");
  assert.match(calls.showTextDocument[0].uri.fsPath,
    /chapters[\\/]01_tensors_autograd\.py$/);
});

test("openEntry：入口不存在 → 警告而非崩溃", async () => {
  resetCounters();
  await panelStub.handler({
    type: "openEntry", payload: { entry: "chapters/__no_such__.py" },
  });
  assert.equal(calls.showTextDocument.length, 0);
  assert.ok(calls.warnings.some((w) => w.includes("入口脚本不存在")),
    `应有警告，实际: ${JSON.stringify(calls.warnings)}`);
});

test("WebToExt 协议的每个消息类型都有路由（不抛异常、不静默崩）", async () => {
  const types = readFileSync(join(EXT, "src", "types.ts"), "utf8");
  const block = types.slice(types.indexOf("export type WebToExt"));
  const literals = [...block.matchAll(/type: "([a-zA-Z]+)"/g)].map((m) => m[1]);
  assert.ok(literals.length >= 8, `协议消息类型解析异常: ${literals}`);
  for (const t of literals) {
    // 有副作用的类型跳过；openEntry 用不存在的入口以免重复打开文件
    if (t === "ready" || t === "run" || t === "stop" || t === "searchError") {
      continue;
    }
    const payload = t === "openEntry"
      ? { entry: "chapters/__no_such__.py" }
      : t === "quizAnswer"
        ? { chapterId: "01_tensors_autograd", index: 0, selfCorrect: false }
        : { chapterId: "01_tensors_autograd", text: "x", tab: "lesson" };
    await panelStub.handler({ type: t, payload });
  }
});

test("run/stop 有副作用消息的路由函数存在", () => {
  assert.equal(typeof calls.commands.get("tfTutor.runFile"), "function");
  assert.equal(typeof calls.commands.get("tfTutor.stopRun"), "function");
});
