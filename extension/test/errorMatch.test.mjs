import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { matchAgainstPatterns } from "../out/python/errorParser.js";

const here = dirname(fileURLToPath(import.meta.url));
const patterns = JSON.parse(
  readFileSync(join(here, "..", "content", "errors", "errorPatterns.json"), "utf8")).patterns;

/** 真实报错样本 → 期望命中的模式 id（与手册附录 C 反例对应） */
const POSITIVE = [
  ["ValueError: shapes (2,3) and (2,3) not aligned: 3 (dim 1) != 2 (dim 0)", "matmul_shape_mismatch"],
  ['ValueError: Input 0 of layer "dense" is incompatible with the layer: expected shape=(None, 1), found shape=(None, 100, 1)', "input_shape_wrong"],
  ["ValueError: Negative dimension size caused by subtracting 2 from 1", "negative_dimension"],
  ["RuntimeError: You must compile your model before using it.", "must_compile"],
  ["ValueError: A KerasTensor cannot be used as input to a TensorFlow function", "keras_tensor_wrong_fn"],
  ["ValueError: Exception encountered when calling SelfAttention.call(). Shapes used to initialize variables must be fully-defined", "shape_fully_defined"],
  ["ValueError: Unknown layer: 'MyLayer'. Please pass custom_objects", "unknown_layer"],
  ["tensorflow.python.framework.errors_impl.InvalidArgumentError: indices out of range", "indices_out_of_range"],
  ["ValueError: setting an array element with a sequence. inhomogeneous shape", "ragged_sequence"],
  ["ValueError: Exception input shapes ... expected ndims=3", "rnn_ndims"],
  ["tensorflow.python.framework.errors_impl.ResourceExhaustedError: OOM when allocating tensor", "resource_exhausted"],
  ["ModuleNotFoundError: No module named 'tensorflow'", "module_not_found"],
  ["FileNotFoundError: [Errno 2] No such file or directory: 'figures/x.png'", "file_not_found"],
  ["SyntaxError: unexpected character after line continuation character", "syntax_error"],
  ["step=  5 loss=nan w=nan b=0.000", "nan_loss"],
  ["numpy.core._exceptions._UFuncNoLoopError: cannot compute MatMul due to input types", "dtype_mismatch_cast"],
];

test("真实报错样本逐条命中预期模式", () => {
  for (const [err, expectedId] of POSITIVE) {
    const hit = matchAgainstPatterns(err, patterns);
    assert.ok(hit, `未命中: ${err.slice(0, 60)}`);
    assert.equal(hit.pattern.id, expectedId, `应命中 ${expectedId}，实际 ${hit.pattern.id}`);
    assert.ok(hit.matched.length > 0);
  }
});

test("无关文本不误报", () => {
  for (const text of ["hello world", "epoch 1: loss=0.50", ""]) {
    assert.equal(matchAgainstPatterns(text, patterns), null, `误报: ${text}`);
  }
});

test("多行堆栈中命中（im 标志）", () => {
  const stack = [
    "Traceback (most recent call last):",
    '  File "ch02.py", line 8, in <module>',
    "    model.fit(x, y)",
    'ValueError: Input 0 of layer "dense" is incompatible with the layer',
  ].join("\n");
  const hit = matchAgainstPatterns(stack, patterns);
  assert.equal(hit?.pattern.id, "input_shape_wrong");
});

test("第一条命中优先（顺序即优先级）", () => {
  // 同时含两种模式 → 排在前面的 shape_mismatch 胜出
  const both = "ValueError: shapes (2,3) and (2,3) not aligned\n"
    + "ResourceExhaustedError: OOM";
  assert.equal(matchAgainstPatterns(both, patterns)?.pattern.id, "matmul_shape_mismatch");
});

test("非法正则不炸插件（跳过该条，后续模式仍生效）", () => {
  const err = "ValueError: shapes (2,3) and (2,3) not aligned";
  const bad = [{ id: "bad", regex: "([unclosed", title: "", causes: [], fixes: [], relatedChapter: "" }];
  assert.equal(matchAgainstPatterns(err, bad), null);
  const mixed = [bad[0], ...patterns];
  assert.equal(matchAgainstPatterns(err, mixed)?.pattern.id, "matmul_shape_mismatch");
});
