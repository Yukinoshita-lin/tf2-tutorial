/**
 * 测试专用：以 stub 替换 `vscode` 模块（进程级单例，幂等安装）。
 * 所有需要加载 out/*.js 的测试共用同一个 stub，避免模块缓存导致的实例错位。
 */

import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { dirname } from "node:path";

const require = createRequire(import.meta.url);
const here = dirname(fileURLToPath(import.meta.url));

export const calls = {
  commands: new Map(),
  showTextDocument: [],
  warnings: [],
  infos: [],
  executed: [],
};

function resetCalls() {
  calls.showTextDocument.length = 0;
  calls.warnings.length = 0;
  calls.infos.length = 0;
  calls.executed.length = 0;
}

let installed = null;

/** 安装（幂等）并返回 { vscode, calls }。root = 教学仓库根目录。 */
export function getVscodeStub(root) {
  if (!installed) {
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
        workspaceFolders: [{ uri: { fsPath: root } }],
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
    installed = { vscode: vscodeStub };
  }
  resetCalls();
  return installed;
}

/** 创建插件扩展上下文 stub（globalState 用内存 Map 持久化，可断言） */
export function makeContext(EXT) {
  const store = new Map();
  return {
    context: {
      extensionPath: EXT,
      asAbsolutePath: (p) => join(EXT, p),
      globalState: { get: (k) => store.get(k), update: async (k, v) => { store.set(k, v); } },
      workspaceState: { get: () => undefined, update: async () => { } },
      subscriptions: { push: () => { } },
    },
    store,
  };
}
