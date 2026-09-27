import { test } from "node:test";
import assert from "node:assert/strict";
import { filterNoise, flushNoise, initialState } from "../out/python/noiseFilter.js";

const NOISE1 = "WARNING: All log messages before absl::InitializeLog() is called are written to STDERR";
const NOISE2 = "I0000 00:00:1790516493.935771   15776 port.cc:153] oneDNN custom operations are on.";
const REAL_ERR = "ValueError: shapes (2,3) and (2,3) not aligned: 3 (dim 1) != 2 (dim 0)";

test("噪音行被隐藏，首次提示一次", () => {
  const st = initialState();
  const r1 = filterNoise(st, NOISE1 + "\n");
  assert.equal(r1.shown.length, 1);
  assert.match(r1.shown[0], /已隐藏 TensorFlow 初始化的无害日志/);
  const r2 = filterNoise(st, NOISE2 + "\n");
  assert.equal(r2.shown.length, 0);
  assert.equal(st.total, 2);
});

test("真实报错行原样保留", () => {
  const st = initialState();
  const r = filterNoise(st, REAL_ERR + "\n");
  assert.deepEqual(r.shown, [REAL_ERR]);
  assert.equal(r.noise, 0);
});

test("半行跨批拼接（chunk 边界）", () => {
  const st = initialState();
  const a = filterNoise(st, "ValueError: sha");
  assert.deepEqual(a.shown, []);           // 半行不显示
  const b = filterNoise(st, "pes (2,3) and (2,3) not aligned\n");
  assert.deepEqual(b.shown, ["ValueError: shapes (2,3) and (2,3) not aligned"]);
  assert.equal(st.carry, "");
});

test("半行若为噪音尾巴，flush 时吞掉", () => {
  const st = initialState();
  filterNoise(st, "I0000 port.cc:153] oneDNN custom operations are on.");
  const f = flushNoise(st);
  assert.equal(f.shown.length, 0);
  assert.equal(f.noise, 1);
});

test("flush 对真实报错尾巴不吞", () => {
  const st = initialState();
  filterNoise(st, "ValueError: shapes (2,3) and (2,3) not aligned");
  const f = flushNoise(st);
  assert.equal(f.shown.length, 1);
  assert.equal(f.noise, 0);
});

test("混合输出：噪音隐藏、报错保留、计数正确", () => {
  const st = initialState();
  const mixed = [NOISE1, REAL_ERR, NOISE2, "Traceback (most recent call last):"].join("\n") + "\n";
  const r = filterNoise(st, mixed);
  assert.equal(r.noise, 2);
  assert.deepEqual(r.shown, [
    "[TF 学习伴侣] （已隐藏 TensorFlow 初始化的无害日志：absl / oneDNN）",
    REAL_ERR,
    "Traceback (most recent call last):",
  ]);
  assert.equal(st.total, 2);
});

test("无噪音的普通输出不受影响", () => {
  const st = initialState();
  const r = filterNoise(st, "epoch 1: loss=0.50\nepoch 2: loss=0.26\n");
  assert.equal(r.shown.length, 2);
  assert.equal(r.noise, 0);
  assert.equal(st.total, 0);
});
