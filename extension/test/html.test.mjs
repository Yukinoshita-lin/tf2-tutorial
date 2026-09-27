import { test } from "node:test";
import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const { getVscodeStub } = await import(pathToFileURL(join(here, "vscodeStub.mjs")).href);
const ROOT = join(here, "..", "..");
getVscodeStub(ROOT);
const { LessonPanel } = await import(pathToFileURL(join(here, "..", "out", "panels", "lessonPanel.js")).href);

test("Webview HTML：CSP + nonce + 五 Tab 齐备", () => {
  const html = LessonPanel.buildHtml("https://test-csp.example");
  assert.ok(html.startsWith("<!DOCTYPE html>"));
  // CSP：script 仅允许 nonce，img 允许 vscode 资源
  assert.match(html, /Content-Security-Policy"/);
  assert.match(html, /default-src 'none'/);
  assert.match(html, /script-src 'nonce-[0-9a-f]{32}'/);
  assert.match(html, /img-src https:\/\/test-csp\.example data:/);
  // 五个 Tab
  for (const tab of ["讲解", "输出", "图表", "自测", "进度"]) {
    assert.ok(html.includes(tab), `缺 Tab: ${tab}`);
  }
  // 关键交互元素
  for (const id of ["stopBtn", "runState", "views", "openEntry"]) {
    assert.ok(html.includes(id), `缺元素: ${id}`);
  }
});

test("Webview HTML：内联脚本无 '</script>' 破坏", () => {
  const html = LessonPanel.buildHtml("https://test-csp.example");
  // 除最后收尾标签外，不应出现多余的 </script>
  const count = html.split("</script>").length - 1;
  assert.equal(count, 1, `出现 ${count} 次 </script>，内联 JS 可能在中途终止文档`);
});

test("Webview HTML：中文 UI 文案与样式主题变量", () => {
  const html = LessonPanel.buildHtml("https://test-csp.example");
  assert.ok(html.includes("--vscode-editor-background"));
  assert.ok(html.includes("动手试一试"));
  assert.ok(html.includes("过关自测"));
  assert.ok(html.includes("errorCard"));
  assert.ok(html.includes("已隐藏") === false); // 该提示来自执行器侧，不在前端 JS
});
