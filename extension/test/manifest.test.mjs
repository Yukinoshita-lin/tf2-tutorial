import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, existsSync, statSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

/** 清单与资源一致性（发布前检查的自动化版，设计要点 4.4） */

const here = dirname(fileURLToPath(import.meta.url));
const EXT = join(here, "..");
const pkg = JSON.parse(readFileSync(join(EXT, "package.json"), "utf8"));

test("package.json：命令唯一、main/图标/片段存在", () => {
  const cmds = pkg.contributes.commands.map((c) => c.command);
  assert.equal(new Set(cmds).size, cmds.length, "命令 id 重复");
  assert.ok(cmds.length >= 12, "命令数量少于设计清单");
  for (const kb of pkg.contributes.keybindings) {
    assert.ok(cmds.includes(kb.command), `键位绑定了未注册命令: ${kb.command}`);
  }
  assert.ok(existsSync(join(EXT, pkg.main)), `main 不存在: ${pkg.main}`);
  assert.ok(existsSync(join(EXT, pkg.icon)), "icon.png 缺失");
  for (const s of pkg.contributes.snippets) {
    assert.ok(existsSync(join(EXT, s.path)), `片段文件缺失: ${s.path}`);
  }
  for (const vc of Object.values(pkg.contributes.views)) {
    for (const v of vc) {
      assert.ok(["tfTutor.knowledgeMap", "tfTutor.progress"].includes(v.id),
        `未实现的视图: ${v.id}`);
    }
  }
});

test("片段：JSON 合法、prefix 唯一、body 非空", () => {
  const snip = JSON.parse(readFileSync(join(EXT, "snippets", "tf-snippets.json"), "utf8"));
  const prefixes = new Set();
  let count = 0;
  for (const [name, s] of Object.entries(snip)) {
    assert.ok(!prefixes.has(s.prefix), `prefix 重复: ${s.prefix}`);
    prefixes.add(s.prefix);
    assert.ok(Array.isArray(s.body) && s.body.length > 0, `${name} body 为空`);
    assert.ok(s.description.length > 0, `${name} 缺 description`);
    count++;
  }
  assert.ok(count >= 10, "片段数量少于 10");
});

test("编译产物存在且为最新（sourceMap 同步）", () => {
  const js = join(EXT, "out", "extension.js");
  const ts = join(EXT, "src", "extension.ts");
  assert.ok(existsSync(js), "先运行 npm run compile");
  assert.ok(existsSync(js + ".map"));
  // 产物不应比源码旧（迭代更新时忘记编译的典型症状）
  const jsM = statSync(js).mtimeMs;
  const tsM = statSync(ts).mtimeMs;
  assert.ok(jsM >= tsM - 1000,
    `out/ (${jsM}) 比 src/ (${tsM}) 旧——请重新 npm run compile`);
});

test("CHANGELOG 记录当前版本", () => {
  const log = readFileSync(join(EXT, "CHANGELOG.md"), "utf8");
  assert.ok(log.includes(pkg.version), `CHANGELOG 缺少 ${pkg.version} 的记录`);
});
