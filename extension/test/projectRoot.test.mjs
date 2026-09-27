import { test } from "node:test";
import assert from "node:assert/strict";
import { findRepoRoot } from "../out/utils/repoRoot.js";

/** 用内存 Set 充当"目录里有什么"的判定，纯逻辑测试向上探测 */
function makePredicate(roots) {
  return (dir) => roots.has(dir.replace(/[\\/]+$/, ""));
}

test("从章节脚本向上找到仓库根", () => {
  const isRoot = makePredicate(new Set(["F:\\Tensorflow"]));
  const r = findRepoRoot("F:\\Tensorflow\\chapters\\01_tensors_autograd.py", isRoot);
  assert.equal(r, "F:\\Tensorflow");
});

test("从子目录（figures/）向上也能找到", () => {
  const isRoot = makePredicate(new Set(["F:\\Tensorflow"]));
  const r = findRepoRoot("F:\\Tensorflow\\figures\\handbook\\hb_x.png", isRoot);
  assert.equal(r, "F:\\Tensorflow");
});

test("起点即仓库根", () => {
  const isRoot = makePredicate(new Set(["F:\\Tensorflow"]));
  assert.equal(findRepoRoot("F:\\Tensorflow", isRoot), "F:\\Tensorflow");
});

test("无仓库根 → undefined（不会越找越远）", () => {
  const isRoot = makePredicate(new Set(["F:\\elsewhere"]));
  assert.equal(
    findRepoRoot("C:\\Users\\me\\Desktop\\x.py", isRoot), undefined);
});

test("正反斜杠混合路径都能处理", () => {
  const isRoot = makePredicate(new Set(["F:/Tensorflow"]));
  const r = findRepoRoot("F:/Tensorflow\\chapters/a.py", isRoot);
  assert.equal(r, "F:/Tensorflow");
});
