/**
 * 共享类型定义：章节内容、报错模式、学习进度、Webview 消息协议。
 */

/** 一章的结构化教学内容（由 scripts/generate_content.py 从手册生成） */
export interface Chapter {
  id: string;              // 例如 "01_tensors_autograd"
  num: string;             // "01"
  title: string;           // "第 1 章 —— 张量与自动求导"
  entry: string;           // 入口脚本（相对工作区根），可为空
  goals: string;           // 本节目标（纯文本）
  concepts: string[];      // 核心概念要点
  experiments: string[];   // 动手试一试（可一键执行的实验建议）
  pitfalls: string[];      // 常见坑
  counterExamples: { title: string; lesson: string }[]; // 反例：标题+一句话教训
  passCriteria: string;    // 过关标准
  quiz: { question: string; answer: string }[]; // 过关自测·基础题
  reading: string[];       // 延伸阅读
  deps: string[];          // 前置章节 id（知识地图用）
  mdRange?: [number, number]; // 手册 md 中本章的起止行（1-based，含头不含尾）
}

/** 报错模式（报错翻译库的一条） */
export interface ErrorPattern {
  id: string;
  regex: string;
  title: string;
  explanation: string;
  causes: string[];
  fixes: string[];
  relatedChapter: string;
}

/** 每章学习进度（四状态，对应需求分析"已读/已跑/已改/已过关"） */
export interface ChapterProgress {
  read: boolean;
  ran: boolean;
  modified: boolean;
  passed: boolean;         // 自测过关（全部基础题自评通过）
}

export interface ProgressData {
  version: 1;
  chapters: Record<string, ChapterProgress>;
  quizHistory: Record<string, { correct: number; total: number; at: string }>;
}

/** 扩展 → Webview 的消息 */
export type ExtToWeb =
  | { type: "ready" }
  | { type: "chapter"; payload: { chapter: Chapter; progress: ChapterProgress; markdown: string } }
  | { type: "output"; payload: { stream: "stdout" | "stderr"; text: string } }
  | { type: "status"; payload: { running: boolean; exitCode?: number; durationMs?: number } }
  | { type: "errorCard"; payload: { pattern: ErrorPattern; matched: string } | null }
  | { type: "images"; payload: { uri: string; name: string }[] }
  | { type: "progress"; payload: ProgressData }
  | { type: "scanResult"; payload: { label: string; value: string; metrics: Record<string, number> }[] }
  | { type: "setTab"; payload: { tab: string } };

/** Webview → 扩展 的消息 */
export type WebToExt =
  | { type: "ready" }
  | { type: "setTab"; payload: { tab: string } }
  | { type: "run"; payload: { kind: "file" | "cell"; cellCode?: string } }
  | { type: "stop" }
  | { type: "experiment"; payload: { text: string } }   // 动手试一试：把建议写进编辑器
  | { type: "quizAnswer"; payload: { chapterId: string; index: number; selfCorrect: boolean } }
  | { type: "markRead"; payload: { chapterId: string } }
  | { type: "openChapter"; payload: { chapterId: string } }
  | { type: "openEntry"; payload: { entry: string } }
  | { type: "searchError"; payload: { text: string } };
