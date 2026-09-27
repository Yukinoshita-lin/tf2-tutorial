import type { ErrorPattern } from "../types";

/**
 * 报错翻译匹配（纯逻辑，可单元测试）。
 * 按 patterns 顺序逐条用正则匹配 stderr，命中即返回；
 * `im` 标志：大小写不敏感 + 多行（对齐设计要点 2.4 的匹配流程）。
 */
export function matchAgainstPatterns(
  stderr: string,
  patterns: ErrorPattern[],
): { pattern: ErrorPattern; matched: string } | null {
  for (const p of patterns) {
    let re: RegExp;
    try {
      re = new RegExp(p.regex, "im");
    } catch {
      continue; // 库里写了非法正则不能让插件崩
    }
    const m = stderr.match(re);
    if (m) {
      return { pattern: p, matched: m[0] };
    }
  }
  return null;
}
