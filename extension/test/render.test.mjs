import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const { getVscodeStub } = await import(pathToFileURL(join(here, "vscodeStub.mjs")).href);
const ROOT = join(here, "..", "..");
getVscodeStub(ROOT);
const { LessonPanel } = await import(pathToFileURL(join(here, "..", "out", "panels", "lessonPanel.js")).href);
const { matchAgainstPatterns } = await import(
  pathToFileURL(join(here, "..", "out", "python", "errorParser.js")).href);
const patterns = JSON.parse(
  readFileSync(join(here, "..", "content", "errors", "errorPatterns.json"), "utf8")
).patterns;

/** 从构建出的 HTML 中提取 md 渲染器源码并编译为函数（不触 DOM） */
function compileFrontend() {
  const html = LessonPanel.buildHtml("https://t.example");
  const re = new RegExp("const md = (function renderMarkdown[\\s\\S]*?\\n});");
  const m = html.match(re);
  assert.ok(m, "md 渲染器源码未注入前端");
  return new Function("return (" + m[1] + ")")();
}

test("前端 JS 可被引擎完整编译（语法级验证）", () => {
  const html = LessonPanel.buildHtml("https://test-csp.example");
  const m = html.match(/<script nonce="[0-9a-f]+">([\s\S]*?)<\/script>/);
  assert.ok(m, "未找到内联脚本");
  assert.doesNotThrow(() => { new Function(m[1]); }, "前端 JS 存在语法错误");
});

test("迷你 Markdown 渲染器：核心语法子集正确转译", () => {
  const mdFn = compileFrontend();
  const out = mdFn([
    "## 标题",
    "**粗体** 和 `代码`",
    "- 项目一",
    "1. 步骤",
    "> 引用",
    "```python",
    "a @ b",
    "```",
    "",
    "正文段落",
  ].join("\n"));
  assert.match(out, /<h3>标题<\/h3>/);
  assert.match(out, /<b>粗体<\/b>/);
  assert.match(out, /<code>代码<\/code>/);
  assert.match(out, /<ul>\s*<li>项目一<\/li>\s*<\/ul>/);
  assert.match(out, /<ol>\s*<li>步骤<\/li>\s*<\/ol>/);
  assert.match(out, /<blockquote>引用<\/blockquote>/);
  assert.match(out, /<pre>a @ b<\/pre>/);
  assert.match(out, /<p>正文段落<\/p>/);
  // HTML 转义：<script> 注入必须被转义
  const xss = mdFn("<script>alert(1)</script>");
  assert.ok(!xss.includes("<script>"), "未转义的 HTML 注入！");
});

test("迷你 Markdown 渲染器：表格行转译", () => {
  const mdFn = compileFrontend();
  const out = mdFn("| a | b |\n| 1 | 2 |");
  assert.match(out, /class="trow"/);
  assert.match(out, /class="tcell"/);
});

test("报错库对正常训练输出零误报（真实样本）", () => {
  const realOutputs = [
    "Chapter 01 — tensors & autograd (NumPy-only)\nstep=100 loss=0.0578 w=2.537 b=0.009\n[timer] chapter01 total: 0.098s",
    "epoch 1: loss=0.5041 val_acc=0.8562\nepoch 2: loss=0.2586 val_acc=0.8792\n测试集: accuracy=0.8542  F1=0.8521",
    "saved -> F:\\Tensorflow\\figures\\chapter01_loss_surface.png\nfigure saved",
    "step  1000: loss_g=6.634 loss_d=0.333\nGAN training in progress",
    "Downloading data: 100%|██████████| 170M/170M",
    "shape a = (2, 3) dtype a = float32\na + b (broadcast) =",
  ];
  for (const out of realOutputs) {
    const hit = matchAgainstPatterns(out, patterns);
    assert.equal(hit, null, `对正常输出误报: ${hit?.pattern.id}`);
  }
});
