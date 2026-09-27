import * as vscode from "vscode";
import { ChapterIndexEntry, loadChapterIndex, loadProgress } from "../data/content";

/**
 * 知识地图 / 进度 树视图（设计要点 2.8）。
 * 已过关 🟢 · 已跑 🔵 · 已读 ⚪ · 未学 ⚪ —— 点击章节打开讲解面板。
 */

type Icon = "passed" | "ran" | "read" | "fresh";

export class KnowledgeMapProvider implements vscode.TreeDataProvider<Node> {
  private _onDidChange = new vscode.EventEmitter<void>();
  readonly onDidChangeTreeData = this._onDidChange.event;

  constructor(private readonly context: vscode.ExtensionContext) {}

  refresh(): void {
    this._onDidChange.fire();
  }

  getTreeItem(el: Node): vscode.TreeItem {
    return el.item;
  }

  getChildren(): Node[] {
    const index = loadChapterIndex(this.context);
    const progress = loadProgress(this.context);
    return index.map((e: ChapterIndexEntry) => {
      const p = progress.chapters[e.id];
      const score = p ? [p.read, p.ran, p.modified, p.passed].filter(Boolean).length : 0;
      const icon: Icon =
        score === 4 ? "passed" : p?.ran ? "ran" : p?.read ? "read" : "fresh";
      const item = new vscode.TreeItem(
        `${e.num}  ${titleOf(e)}`,
        vscode.TreeItemCollapsibleState.None);
      item.description = stateLabel(icon);
      item.iconPath = new vscode.ThemeIcon(iconName(icon), iconColor(icon));
      item.tooltip = new vscode.MarkdownString(
        `**${e.title}**\n\n- 入口：\`${e.entry}\`\n- 前置：${e.deps.length ? e.deps.join(", ") : "无"}\n` +
        `- 进度：${score}/4`);
      item.command = {
        command: "tfTutor.openLesson", title: "打开讲解",
        arguments: [e.id],
      };
      return { item, entry: e };
    });
  }
}

export class ProgressProvider implements vscode.TreeDataProvider<vscode.TreeItem> {
  private _onDidChange = new vscode.EventEmitter<void>();
  readonly onDidChangeTreeData = this._onDidChange.event;

  constructor(private readonly context: vscode.ExtensionContext) {}

  refresh(): void {
    this._onDidChange.fire();
  }

  getTreeItem(el: vscode.TreeItem): vscode.TreeItem {
    return el;
  }

  getChildren(): vscode.TreeItem[] {
    const progress = loadProgress(this.context);
    const index = loadChapterIndex(this.context);
    const total = index.length || 1;
    let sum = 0;
    const counts = { read: 0, ran: 0, modified: 0, passed: 0 };
    for (const e of index) {
      const p = progress.chapters[e.id];
      if (p) {
        sum += [p.read, p.ran, p.modified, p.passed].filter(Boolean).length;
        counts.read += p.read ? 1 : 0;
        counts.ran += p.ran ? 1 : 0;
        counts.modified += p.modified ? 1 : 0;
        counts.passed += p.passed ? 1 : 0;
      }
    }
    const pct = Math.round((sum / (total * 4)) * 100);
    const bar = "█".repeat(Math.round(pct / 5)) + "░".repeat(20 - Math.round(pct / 5));
    const overall = new vscode.TreeItem(`总进度 ${pct}%  ${bar}`);
    overall.description = `${counts.passed}/${total} 章过关`;
    const items = [overall];
    const mk = (label: string, value: number, icon: string) => {
      const it = new vscode.TreeItem(label, vscode.TreeItemCollapsibleState.None);
      it.description = `${value} 章`;
      it.iconPath = new vscode.ThemeIcon(icon);
      return it;
    };
    items.push(mk("已读", counts.read, "eye"));
    items.push(mk("已跑", counts.ran, "play"));
    items.push(mk("已改", counts.modified, "edit"));
    items.push(mk("已过关", counts.passed, "check"));
    return items;
  }
}

interface Node {
  item: vscode.TreeItem;
  entry: ChapterIndexEntry;
}

function titleOf(e: ChapterIndexEntry): string {
  const m = e.title.match(/——\s*(.+)$/);
  return m ? m[1] : e.title;
}

function stateLabel(icon: Icon): string {
  return { passed: "已过关", ran: "已跑", read: "已读", fresh: "未开始" }[icon];
}

function iconName(icon: Icon): string {
  return { passed: "check-all", ran: "play-circle", read: "circle-large-outline", fresh: "circle-outline" }[icon];
}

function iconColor(icon: Icon): vscode.ThemeColor {
  switch (icon) {
    case "passed":
      return new vscode.ThemeColor("charts.green");
    case "ran":
      return new vscode.ThemeColor("charts.blue");
    case "read":
      return new vscode.ThemeColor("charts.yellow");
    default:
      return new vscode.ThemeColor("foreground");
  }
}
