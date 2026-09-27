import { test } from "node:test";
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { pathToFileURL } from "node:url";

/**
 * TF 真实噪音端到端：在真机导入 TensorFlow（会产生 absl/oneDNN stderr 日志），
 * 用与 executor 相同的 env + noiseFilter 管线，验证：
 *   ① 真实输出（MARKER）按原样显示；② 无害日志被隐藏；③ stderr 仍完整供匹配。
 * 无 TF 环境自动跳过。
 */

const here = dirname(fileURLToPath(import.meta.url));
const ROOT = join(here, "..", "..");
const { getVscodeStub } = await import(pathToFileURL(join(here, "vscodeStub.mjs")).href);
getVscodeStub(ROOT);
const { buildExecutorEnv } = await import(pathToFileURL(join(here, "..", "out", "python", "executor.js")).href);
const { filterNoise, flushNoise, initialState } = await import(
  pathToFileURL(join(here, "..", "out", "python", "noiseFilter.js")).href);

const PY = ["F:/tf2-env/Scripts/python.exe", join(ROOT, ".venv", "Scripts", "python.exe")]
  .find((p) => existsSync(p));

test("TF 导入：真实 stderr 过噪音后真实输出保留、噪声计数>0", { skip: !PY && "无 TF 解释器" }, async () => {
  const code = [
    "import sys",
    "import tensorflow as tf",
    "print('MARKER_REAL_OUTPUT', tf.__version__)",
  ].join("; ");
  const fullStderr = await new Promise((resolve) => {
    const p = spawn(PY, ["-c", code], {
      cwd: ROOT, env: { ...buildExecutorEnv(ROOT) },
    });
    let err = "";
    p.stderr.on("data", (d) => (err += d.toString("utf8")));
    p.stdout.on("data", () => undefined);
    p.on("close", () => resolve(err));
  });

  assert.ok(fullStderr.length > 0, "应有无害初始化日志");
  assert.match(fullStderr, /absl::InitializeLog|oneDNN custom operations/);

  // 走与 executor 相同的过滤管线
  const st = initialState();
  const shown = [];
  let carry = fullStderr;
  for (let i = 0; i < 50 && carry; i++) {
    const chunk = carry.slice(0, 256);
    carry = carry.slice(256);
    const r = filterNoise(st, chunk);
    shown.push(...r.shown);
  }
  shown.push(...flushNoise(st).shown);
  const shownText = shown.join("\n");

  assert.ok(shownText.includes("已隐藏 TensorFlow 初始化的无害日志"), "应提示一次隐藏");
  assert.ok(!/absl::InitializeLog/.test(shownText), "absl 日志应被隐藏");
  assert.ok(!/oneDNN custom operations/.test(shownText), "oneDNN 日志应被隐藏");
  assert.ok(st.total > 0, "应统计到噪音行");
});
