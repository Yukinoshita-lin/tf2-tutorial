import { createRequire } from "node:module";
import { existsSync, readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const here = dirname(fileURLToPath(import.meta.url));
const EXT = resolve(here, "..");
const ROOT = resolve(EXT, "..");

const calls = { showTextDocument: [], warnings: [], infos: [] };
const vscodeStub = {
  commands: { registerCommand: (id, fn) => calls.commands?.set(id, fn) ?? (calls.commands = new Map([[id, fn]])) },
  window: {
    showTextDocument: async (uri, o) => { calls.showTextDocument.push(uri); return {}; },
    showWarningMessage: async (m) => { calls.warnings.push(m); return undefined; },
    showInformationMessage: async (m) => { calls.infos.push(m); return undefined; },
    createWebviewPanel: () => { throw new Error("not used"); },
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
  env: { openExternal: async () => undefined, clipboard: { readText: async () => "" } },
  Uri: { file: (p) => ({ fsPath: p, path: p }) },
  ViewColumn: { One: 1, Beside: 2 },
  ThemeIcon: class { },
  ThemeColor: class { },
  MarkdownString: class { },
  TreeItem: class { },
  TreeItemCollapsibleState: { None: 0 },
  ProgressLocation: { Notification: 15 },
};

const Module = require("node:module");
const origLoad = Module._load;
Module._load = function (request, parent, isMain) {
  if (request === "vscode") { return vscodeStub; }
  return origLoad.apply(this, arguments);
};

const cmd = require(join(EXT, "out", "commands", "index.js"));
Module._load = origLoad;

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
  setHandler(fn) { console.log("setHandler called:", typeof fn); this.handler = fn; },
  reveal() { }, setRunning() { }, appendOutput() { }, setTab() { },
  showErrorCard() { }, showImages() { }, updateProgress: async () => { },
  currentChapterId: "01_tensors_autograd",
};
const kmStub = { refresh() { } };
const pgStub = { refresh() { } };

cmd.initCommands(context);
console.log("before registerAll, handler:", typeof panelStub.handler);
cmd.registerAll(context, panelStub, kmStub, pgStub);
console.log("after registerAll, handler:", typeof panelStub.handler);
console.log("registered commands:", calls.commands ? calls.commands.size : "n/a");

try {
  await panelStub.handler({ type: "openEntry", payload: { entry: "chapters/01_tensors_autograd.py" } });
  console.log("dispatch ok");
} catch (e) {
  console.log("dispatch THREW:", e && e.message);
}
console.log("showTextDocument calls:", calls.showTextDocument.length, calls.showTextDocument);
console.log("warnings:", calls.warnings);
