import { test } from "node:test";
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

/**
 * Python 执行层冒烟（设计要点 4.3"Python 脚本报错时"链路 + 需求分析"一键运行"）：
 * 用与 executor.ts 完全相同的环境变量跑真实章节脚本，验证
 * 退出码 0 / 中文输出无乱码 / 正常耗时。
 * 解释器不存在时跳过（CI 无 TF 环境时不应红）。
 */

const here = dirname(fileURLToPath(import.meta.url));
const ROOT = join(here, "..", "..");
const PY_CANDIDATES = [
  "F:/tf2-env/Scripts/python.exe",
  join(ROOT, ".venv", "Scripts", "python.exe"),
];
const py = PY_CANDIDATES.find((p) => existsSync(p));

function runScript(script, timeoutMs = 180_000) {
  return new Promise((resolve) => {
    const started = Date.now();
    const p = spawn(py, [script], {
      cwd: ROOT,
      env: {
        ...process.env,
        PYTHONPATH: join(ROOT, "src"),
        PYTHONIOENCODING: "utf-8",
        PYTHONUTF8: "1",
        TF_CPP_MIN_LOG_LEVEL: "3",
      },
    });
    let out = "";
    p.stdout.on("data", (d) => (out += d.toString("utf8")));
    p.stderr.on("data", (d) => (out += d.toString("utf8")));
    const t = setTimeout(() => { p.kill("SIGKILL"); }, timeoutMs);
    p.on("close", (code) => { clearTimeout(t); resolve({ code, out, ms: Date.now() - started }); });
    p.on("error", () => { clearTimeout(t); resolve({ code: -1, out, ms: 0 }); });
  });
}

test("第 0 章：真实执行、中文无乱码、正常耗时", { skip: !py && "未找到 Python 解释器" }, async () => {
  const script = join(ROOT, "chapters", "00_ml_basics.py");
  assert.ok(existsSync(script));
  const { code, out, ms } = await runScript(script);
  assert.equal(code, 0, `脚本失败: ${out.slice(-500)}`);
  assert.ok(out.includes("第 0 章"), "缺少章节标题输出");
  assert.ok(!out.includes("\uFFFD"), "输出含替换符——编码又坏了");
  assert.ok(ms < 120_000, `第 0 章耗时异常: ${ms}ms`);
});

test("报错脚本 → 非零退出码 + stderr 可被翻译（executor 前置链路）", { skip: !py && "未找到 Python 解释器" }, async () => {
  const patterns = JSON.parse(
    readFileSync(join(ROOT, "extension", "content", "errors", "errorPatterns.json"), "utf8")
  ).patterns;
  const { code, out } = await new Promise((resolve) => {
    const p = spawn(py, ["-c", "import numpy as np\na=np.ones((2,3)); b=np.ones((2,3))\na@b"], {
      cwd: ROOT,
      env: { ...process.env, PYTHONIOENCODING: "utf-8", PYTHONUTF8: "1" },
    });
    let out = "";
    p.stderr.on("data", (d) => (out += d.toString("utf8")));
    p.on("close", (code) => resolve({ code, out }));
  });
  assert.notEqual(code, 0, "故意报错的脚本应非零退出");
  const hit = patterns.find((x) => new RegExp(x.regex, "im").test(out));
  assert.equal(hit?.id, "matmul_shape_mismatch", "真实 stderr 未命中翻译库");
});
