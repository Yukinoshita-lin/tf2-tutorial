/**
 * 迷你 Markdown 渲染器（自包含纯函数，无 DOM/无依赖）。
 *
 * 由 lessonPanel 的前端脚本通过 `renderMarkdown.toString()` 注入到 Webview，
 * 同时可被单元测试直接编译调用——教学内容是受控子集，覆盖：
 * 标题/粗体/斜体/行内代码/围栏代码块/无序有序列表/引用/表格/水平线/段落。
 * 所有文本先 HTML 转义，杜绝注入。
 */
export function renderMarkdown(src: string): string {
  function esc(s: string): string {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }
  function inline(s: string): string {
    s = esc(s);
    s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
    s = s.replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>");
    s = s.replace(/\*([^*]+)\*/g, "<i>$1</i>");
    s = s.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<span class="link">$1</span>');
    return s;
  }
  const out: string[] = [];
  let inCode = false;
  let code: string[] = [];
  let inList = false;
  let listTag = "ul";
  const closeList = () => {
    if (inList) { out.push("</" + listTag + ">"); inList = false; }
  };
  for (const raw of src.split("\n")) {
    const line = raw.replace(/\t/g, "    ");
    if (line.startsWith("```")) {
      closeList();
      if (inCode) { out.push("<pre>" + esc(code.join("\n")) + "</pre>"); inCode = false; code = []; }
      else { inCode = true; }
      continue;
    }
    if (inCode) { code.push(line); continue; }
    if (/^#{1,4} /.test(line)) {
      closeList();
      const level = line.match(/^#+/)![0].length;
      out.push("<h" + (level + 1) + ">" + inline(line.replace(/^#+ /, "")) + "</h" + (level + 1) + ">");
      continue;
    }
    if (/^\s*[-*] /.test(line)) {
      if (!inList || listTag !== "ul") { closeList(); out.push("<ul>"); inList = true; listTag = "ul"; }
      out.push("<li>" + inline(line.replace(/^\s*[-*] /, "")) + "</li>");
      continue;
    }
    if (/^\s*\d+\. /.test(line)) {
      if (!inList || listTag !== "ol") { closeList(); out.push("<ol>"); inList = true; listTag = "ol"; }
      out.push("<li>" + inline(line.replace(/^\s*\d+\. /, "")) + "</li>");
      continue;
    }
    if (/^> /.test(line)) {
      closeList();
      out.push("<blockquote>" + inline(line.slice(2)) + "</blockquote>");
      continue;
    }
    if (/^---+$/.test(line.trim())) {
      closeList();
      out.push("<hr>");
      continue;
    }
    if (line.trim() === "") { closeList(); continue; }
    if (/\|/.test(line) && !/^\|[-| ]+\|$/.test(line)) {
      closeList();
      const cells = line.split("|")
        .filter((_c, i, a) => !(i === 0 && a[0].trim() === ""))
        .map((c) => inline(c.trim()));
      out.push('<div class="trow">' +
        cells.map((c) => '<span class="tcell">' + c + "</span>").join("") + "</div>");
      continue;
    }
    closeList();
    out.push("<p>" + inline(line) + "</p>");
  }
  closeList();
  if (inCode) { out.push("<pre>" + esc(code.join("\n")) + "</pre>"); }
  return out.join("\n");
}
