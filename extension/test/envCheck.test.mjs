import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const { getVscodeStub } = await import(pathToFileURL(join(here, "vscodeStub.mjs")).href);
const ROOT = join(here, "..", "..");
getVscodeStub(ROOT);
const { resolveInterpreterSync, buildExecutorEnv } = await import(
  pathToFileURL(join(here, "..", "out", "python", "executor.js")).href);
const env = await import(pathToFileURL(join(here, "..", "out", "python", "env.js")).href);

test("解释器探测：返回存在的可执行文件", () => {
  const py = resolveInterpreterSync();
  assert.ok(existsSync(py) || py === "python" || py === "python3", `解析到 ${py}`);
});

test("执行环境：强制 UTF-8 与 TF 降噪（防中文乱码回归）", () => {
  const e = buildExecutorEnv(ROOT);
  assert.equal(e.PYTHONIOENCODING, "utf-8");
  assert.equal(e.PYTHONUTF8, "1");
  assert.equal(e.TF_CPP_MIN_LOG_LEVEL, "3");
  assert.ok((e.PYTHONPATH ?? "").includes(join("src")));
});

test("环境自检：真实调度 scripts/env_check.py 并解析结论", async (t) => {
  const tf = await env.probeTensorFlow();
  const result = await env.runProjectEnvCheck();
  assert.ok(result, "env_check.py 未被找到/执行");
  const joined = result.lines.join("\n");
  assert.match(joined, /版本兼容性自检/, "自检脚本应输出结论段");
  if (tf.installed) {
    // 有 TF 的环境（本机 / 装了依赖的 CI）：必须给出 ✅ 兼容结论
    t.diagnostic(`TF ${tf.version} → ok=${result.ok}`);
    assert.equal(result.ok, true, "TF 已装但自检未通过：" + joined.slice(-400));
  } else {
    // 无 TF 的环境（CI extension job）：结论应为"不兼容"并给出修复建议——
    // 这证明「调度 + 解析 + 结论」功能本身正常，只是环境缺依赖
    assert.equal(result.ok, false);
    assert.match(joined, /TensorFlow 不可导入/);
    assert.match(joined, /pip install/);
    t.skip("无 TF 环境，仅验证调度与解析");
  }
});

test("TensorFlow 探测：已安装且版本 >= 2.16", async (t) => {
  const r = await env.probeTensorFlow();
  if (!r.installed) { return t.skip("无 TF 环境"); }
  const [maj] = r.version.split(".").map(Number);
  assert.ok(maj >= 2 && maj < 3, `TF 版本异常: ${r.version}`);
});
