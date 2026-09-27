import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const { getVscodeStub } = await import(pathToFileURL(join(here, "vscodeStub.mjs")).href);
const ROOT = join(here, "..", "..");
getVscodeStub(ROOT);
const noise = await import(pathToFileURL(join(here, "..", "out", "python", "noiseFilter.js")).href);
const { filterNoise, flushNoise, initialState } = noise;
const extract = await import(pathToFileURL(join(here, "..", "out", "utils", "extract.js")).href);
const { extractCellRange, extractMetrics } = extract;
const { matchAgainstPatterns } = await import(
  pathToFileURL(join(here, "..", "out", "python", "errorParser.js")).href);
const patterns = JSON.parse(
  readFileSync(join(here, "..", "content", "errors", "errorPatterns.json"), "utf8")
).patterns;

/** 可复现的伪随机（mulberry32）——同一 seed 永远同一序列，失败可重放 */
function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const NOISE_LINES = [
  "WARNING: All log messages before absl::InitializeLog() is called are written to STDERR",
  "I0000 00:00:1790516493.935771   15776 port.cc:153] oneDNN custom operations are on.",
  "To enable the following instructions: SSE3 SSE4.1, rebuild TensorFlow.",
  "2026-09-27 21:41:33 cpu_feature_guard.cc:182] This CPU binary is optimized.",
];

test("noiseFilter：200 组随机切分/混合，不变式全保持", () => {
  for (let seed = 1; seed <= 200; seed++) {
    const rand = rng(seed);
    // 随机构造输出行：40% 噪音，60% 真实内容
    const n = 1 + Math.floor(rand() * 12);
    const lines = [];
    for (let i = 0; i < n; i++) {
      lines.push(rand() < 0.4
        ? NOISE_LINES[Math.floor(rand() * NOISE_LINES.length)]
        : `real output line ${i}: loss=${(rand()).toFixed(4)}`);
    }
    const expectedReal = lines.filter((l) =>
      !/absl::InitializeLog|oneDNN custom operations|cpu_feature_guard|To enable the following instructions/.test(l));

    // 随机切成 1-5 个 chunk（模拟流式到达），含随机的缺尾换行
    const st = initialState();
    const shown = [];
    let pos = 0;
    const text = lines.join("\n") + (rand() < 0.5 ? "\n" : "");
    while (pos < text.length) {
      const size = 1 + Math.floor(rand() * 40);
      const chunk = text.slice(pos, pos + size);
      pos += size;
      const r = filterNoise(st, chunk);
      shown.push(...r.shown);
    }
    shown.push(...flushNoise(st).shown);

    // 不变式 1：显示的行（去掉一次性提示）必须与"非噪音行"逐行一致
    const shownReal = shown.filter((l) => !l.includes("已隐藏 TensorFlow 初始化的无害日志"));
    assert.deepEqual(shownReal, expectedReal, `seed=${seed} 显示行与预期不符`);
    // 不变式 2：噪音计数 === 噪音行总数
    assert.equal(st.total, lines.length - expectedReal.length, `seed=${seed} 计数不符`);
    // 不变式 3：提示至多出现一次
    assert.ok(shown.filter((l) => l.includes("已隐藏")).length <= 1, `seed=${seed} 提示重复`);
  }
});

test("extractCellRange：300 组随机文档，覆盖完整/互不重叠/每格恰一标记", () => {
  for (let seed = 1; seed <= 300; seed++) {
    const rand = rng(seed);
    const markerVariants = ["# %%", "#%%", "  # %%", "\t#%%"];
    const nCells = 1 + Math.floor(rand() * 5);
    const lines = [];
    for (let c = 0; c < nCells; c++) {
      lines.push(markerVariants[Math.floor(rand() * markerVariants.length)] + ` C${c}`);
      const bodyN = Math.floor(rand() * 3);
      for (let b = 0; b < bodyN; b++) {
        lines.push(`  body ${c}.${b}`);
      }
    }
    // 不变式 A：每个光标位置的区间都包含该行
    for (let cur = 0; cur < lines.length; cur++) {
      const [s, e] = extractCellRange(lines, cur);
      assert.ok(s <= cur && cur < e, `seed=${seed} cur=${cur} 区间[${s},${e}) 不含光标行`);
    }
    // 不变式 B：所有行按区间分块，连续覆盖且不重叠
    const ranges = [];
    for (let cur = 0; cur < lines.length; cur++) {
      ranges.push(extractCellRange(lines, cur).join(":"));
    }
    const unique = [...new Set(ranges)];
    let cover = 0;
    for (const r of unique) {
      const [s, e] = r.split(":").map(Number);
      assert.equal(s, cover, `seed=${seed} 区间不连续: ${r} (期望起点 ${cover})`);
      cover = e;
    }
    assert.equal(cover, lines.length, `seed=${seed} 区间未覆盖到文件末尾`);
    // 不变式 C：每格至多含一个标记行
    for (const r of unique) {
      const [s, e] = r.split(":").map(Number);
      const markers = lines.slice(s, e).filter((l) => /^\s*#\s*%%/.test(l)).length;
      assert.ok(markers <= 1, `seed=${seed} 区间含 ${markers} 个标记`);
    }
  }
});

test("extractMetrics：500 组随机键值，末值语义与归一化", () => {
  const keys = ["val_acc", "accuracy", "acc", "loss", "F1"];
  for (let seed = 1; seed <= 500; seed++) {
    const rand = rng(seed);
    const n = 1 + Math.floor(rand() * 8);
    const lines = [];
    const expected = {};
    for (let i = 0; i < n; i++) {
      const k = keys[Math.floor(rand() * keys.length)];
      const v = rand();
      const sep = rand() < 0.5 ? "=" : ":";
      lines.push(`${k}${sep}${v.toFixed(6)}`);
      expected[k.toLowerCase()] = parseFloat(v.toFixed(6)); // 末值覆盖
    }
    const m = extractMetrics(lines.join("\n"));
    assert.deepEqual(m, expected, `seed=${seed}`);
  }
});

test("matchAgainstPatterns：随机拼接真实报错片段不漏不串", () => {
  const fragments = [
    ["shapes (2,3) and (2,3) not aligned", "matmul_shape_mismatch"],
    ["You must compile your model before", "must_compile"],
    ["Unknown layer: 'X'", "unknown_layer"],
    ["OOM when allocating tensor", "resource_exhausted"],
    ["ModuleNotFoundError: No module named 'requests'", "module_not_found"],
  ];
  for (let seed = 1; seed <= 200; seed++) {
    const rand = rng(seed);
    const chosen = fragments.filter(() => rand() < 0.5);
    const order = chosen.sort(() => rand() - 0.5);
    const text = order.map(([f]) => f).join("\n" + "-".repeat(20) + "\n");
    const hit = matchAgainstPatterns(text, patterns);
    if (order.length === 0) {
      assert.equal(hit, null);
    } else {
      assert.ok(hit, `seed=${seed} 应命中`);
      // 命中的必须是"文本里出现过的"某个模式（顺序优先 → 第一个出现的合法模式）
      const ids = new Set(order.map(([, id]) => id));
      assert.ok(ids.has(hit.pattern.id), `seed=${seed} 命中了文本外的模式`);
    }
  }
});
