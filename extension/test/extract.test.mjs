import { test } from "node:test";
import assert from "node:assert/strict";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const { extractCellRange, extractMetrics } = await import(
  pathToFileURL(join(here, "..", "out", "utils", "extract.js")).href);

// ---------------- 单元格提取 ----------------

const SRC = [
  "# %% A",
  "a = 1",
  "# %% B",
  "b = 2",
  "c = 3",
  "# %% C",
  "d = 4",
];

test("单元格：光标在首格 → 从 0 到下一标记", () => {
  assert.deepEqual(extractCellRange(SRC, 1), [0, 2]);
});

test("单元格：光标在中间格（含所属标记）", () => {
  assert.deepEqual(extractCellRange(SRC, 4), [2, 5]);
});

test("单元格：光标在最后一格 → 到文件末尾（含所属标记）", () => {
  assert.deepEqual(extractCellRange(SRC, 6), [5, 7]);
});

test("单元格：光标正好在标记行 → 该标记归属当前格", () => {
  const [s, e] = extractCellRange(SRC, 3);
  assert.deepEqual([s, e], [2, 5]);
  assert.match(SRC[s], /# %% B/);
});

test("无标记文件 → 整个文件", () => {
  assert.deepEqual(extractCellRange(["a=1", "b=2"], 0), [0, 2]);
});

test("变体写法 #%% 与缩进兼容", () => {
  const src = ["  #%%", "x=1"];
  assert.deepEqual(extractCellRange(src, 1), [0, 2]);
});

// ---------------- 指标提取 ----------------

test("指标：同名取最后一次（训练结束值）", () => {
  const out = "epoch1 acc=0.70\nepoch2 acc=0.85\nval_acc=0.83";
  const m = extractMetrics(out);
  assert.equal(m.acc, 0.85);
  assert.equal(m.val_acc, 0.83);
});

test("指标：支持冒号与 F1/loss", () => {
  const m = extractMetrics("测试集: accuracy=0.8542  F1=0.8521\nloss: 0.1745");
  assert.equal(m.accuracy, 0.8542);
  assert.equal(m.f1, 0.8521);
  assert.equal(m.loss, 0.1745);
});

test("指标：无匹配返回空对象", () => {
  assert.deepEqual(extractMetrics("hello world"), {});
});
