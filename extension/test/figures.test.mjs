import { test } from "node:test";
import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { mkdtempSync, writeFileSync, utimesSync, existsSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { tmpdir } from "node:os";

const here = dirname(fileURLToPath(import.meta.url));
const { getVscodeStub } = await import(pathToFileURL(join(here, "vscodeStub.mjs")).href);
const ROOT = join(here, "..", "..");
getVscodeStub(ROOT); // executor 依赖 vscode，先装 stub
const { newFiguresSince } = await import(pathToFileURL(join(here, "..", "out", "python", "executor.js")).href);

const before = Date.now() - 60_000; // 一分钟前

test("figures：新图捕获、旧图忽略；目录不存在时安全返回空", async () => {
  const dir = mkdtempSync(join(tmpdir(), "tftutor-fig-"));
  writeFileSync(join(dir, "old.png"), "x");
  utimesSync(join(dir, "old.png"), new Date(before), new Date(before));
  await new Promise((r) => setTimeout(r, 30));
  writeFileSync(join(dir, "new.png"), "y");
  writeFileSync(join(dir, "notes.txt"), "z"); // 非图片应被忽略

  const figs = newFiguresSince(dir, Date.now() - 1000);
  assert.deepEqual(figs.map((f) => f.split(/[\\/]/).pop()), ["new.png"]);

  // since 早于所有文件 → 全捕获
  const all = newFiguresSince(dir, before - 1000);
  assert.equal(all.length, 2);

  assert.deepEqual(newFiguresSince(dir, Date.now() + 60_000), []);
});

test("figures：目录不存在 → 空数组不抛异常", () => {
  assert.deepEqual(newFiguresSince(join(tmpdir(), "no-such-dir-xyz"), 0), []);
});
