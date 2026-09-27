import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

/**
 * 内容完整性测试（单一事实来源校验）：
 * 章节 JSON / 报错库 / 片段 / 清单 必须结构完好且与仓库实际文件对齐。
 * 手册更新后跑 generate_content.py，再跑本测试即可发现漂移。
 */

const here = dirname(fileURLToPath(import.meta.url));
const EXT = join(here, "..");
const ROOT = join(EXT, "..");

const index = JSON.parse(readFileSync(join(EXT, "content", "chapters.json"), "utf8"));

test("章节索引：17 章、id 唯一、编号有序", () => {
  assert.equal(index.length, 17);
  const ids = index.map((e) => e.id);
  assert.equal(new Set(ids).size, 17);
  const nums = index.map((e) => Number(e.num));
  assert.deepEqual(nums, [...nums].sort((a, b) => a - b));
});

test("每章 JSON 存在且字段完整", () => {
  for (const e of index) {
    const file = join(EXT, "content", "chapters", `${e.id}.json`);
    assert.ok(existsSync(file), `缺少 ${e.id}.json`);
    const ch = JSON.parse(readFileSync(file, "utf8"));
    assert.equal(ch.id, e.id);
    assert.ok(ch.title.length > 0);
    assert.ok(ch.goals.length > 0, `${e.id} 缺 goals`);
    assert.ok(Array.isArray(ch.concepts));
    assert.ok(Array.isArray(ch.experiments));
    assert.ok(Array.isArray(ch.pitfalls));
    assert.ok(Array.isArray(ch.counterExamples));
    for (const ce of ch.counterExamples) {
      assert.ok(ce.title.length > 0 && ce.lesson.length > 0, `${e.id} 反例字段不完整`);
    }
    assert.ok(Array.isArray(ch.quiz));
    for (const q of ch.quiz) {
      assert.ok(q.question.length > 0 && q.answer.length > 0, `${e.id} 自测题字段不完整`);
    }
    assert.ok(Array.isArray(ch.reading));
  }
});

test("依赖关系：deps 都指向真实章节（知识地图数据可信）", () => {
  const ids = new Set(index.map((e) => e.id));
  for (const e of index) {
    for (const d of e.deps) {
      assert.ok(ids.has(d), `${e.id} 依赖了不存在的章节 ${d}`);
    }
  }
  // 主线依赖的关键链路抽查
  const depOf = Object.fromEntries(index.map((e) => [e.id, e.deps]));
  assert.ok(depOf["02_linear_regression"].includes("01_tensors_autograd"));
  assert.ok(depOf["16_text_capstone"].includes("13_attention_transformer"));
});

test("入口脚本与 mdRange 指向真实文件/行区间", () => {
  const mdLines = readFileSync(join(ROOT, "docs", "learning_handbook_zh.md"), "utf8")
    .split("\n").length;
  for (const e of index) {
    const entryAbs = join(ROOT, e.entry);
    assert.ok(existsSync(entryAbs), `${e.id} 入口脚本不存在: ${e.entry}`);
    const ch = JSON.parse(readFileSync(join(EXT, "content", "chapters", `${e.id}.json`), "utf8"));
    const [a, b] = ch.mdRange;
    assert.ok(a >= 1 && b <= mdLines && a < b, `${e.id} mdRange 越界: ${a}-${b}`);
  }
});

test("章节 JSON 的 mdRange 切片包含各自章节标识（定位正确性）", () => {
  const mdLines = readFileSync(join(ROOT, "docs", "learning_handbook_zh.md"), "utf8").split("\n");
  for (const e of index) {
    const ch = JSON.parse(readFileSync(join(EXT, "content", "chapters", `${e.id}.json`), "utf8"));
    const slice = mdLines.slice(ch.mdRange[0] - 1, ch.mdRange[1] - 1).join("\n");
    // 标识 token：路线图 / 第 0X 章 / 实战 NN / 进阶篇 NN / 进阶篇毕业项目
    const token = e.id === "00_ml_basics" ? "路线图"
      : e.id === "16_text_capstone" ? "进阶篇毕业项目"
      : e.num >= "09" && e.num <= "10" ? `实战 ${e.num}`
      : Number(e.num) >= 11 ? `进阶篇 ${e.num}`
      : `第 ${e.num} 章`;
    assert.ok(slice.includes(token),
      `${e.id} mdRange 切片未包含章节标识 "${token}"`);
  }
});

// ---------------- 报错库 ----------------

const errLib = JSON.parse(
  readFileSync(join(EXT, "content", "errors", "errorPatterns.json"), "utf8")).patterns;

test("报错库：id 唯一、正则可编译、字段完整", () => {
  const ids = new Set();
  for (const p of errLib) {
    assert.ok(!ids.has(p.id), `重复 id: ${p.id}`);
    ids.add(p.id);
    assert.doesNotThrow(() => new RegExp(p.regex, "im"), `非法正则: ${p.id}`);
    assert.ok(p.title.length > 0 && p.explanation.length > 0, `${p.id} 缺解释`);
    assert.ok(p.causes.length > 0 && p.fixes.length > 0, `${p.id} 缺原因/修复`);
    assert.ok(p.relatedChapter.length > 0, `${p.id} 缺相关章节`);
  }
  assert.ok(errLib.length >= 15, "报错库覆盖不足");
});
