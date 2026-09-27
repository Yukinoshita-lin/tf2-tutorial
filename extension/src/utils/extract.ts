/**
 * 单元格/指标提取的纯逻辑（可单元测试，不依赖 vscode）。
 */

/** 判断某行是否为单元格分隔标记（# %% / #%%，允许首尾空白） */
export function isCellMarker(line: string | undefined): boolean {
  return /^\s*#\s*%%/.test(line ?? "");
}

/**
 * 给定文件所有行与光标行（0-based），返回当前单元格的 [start, end) 行区间。
 * 语义（与 VS Code Jupyter 一致）：**标记行属于它下面的格**——
 *   - 光标在标记行上 → 本格从该标记行开始；
 *   - 光标在体行 → 向上并入最近所属的标记行；
 * 无任何标记时返回整个文件 [0, lines.length)。
 */
export function extractCellRange(lines: string[], cursorLine: number): [number, number] {
  const hasMarker = lines.some((l) => isCellMarker(l));
  if (!hasMarker) {
    return [0, lines.length];
  }
  let start = Math.min(Math.max(cursorLine, 0), lines.length - 1);
  let end = start;
  if (!isCellMarker(lines[start])) {
    while (start > 0 && !isCellMarker(lines[start - 1])) {
      start--;
    }
    if (start > 0 && isCellMarker(lines[start - 1])) {
      start--; // 并入所属标记行
    }
  }
  while (end < lines.length - 1 && !isCellMarker(lines[end + 1])) {
    end++;
  }
  return [start, end + 1];
}

const METRIC_RE = /(val_acc|accuracy|acc|loss|F1)\s*[=:]\s*([0-9]+(?:\.[0-9]+)?)/gi;

/**
 * 从脚本输出中提取最终指标：同名指标取**最后一次**出现（=训练结束值）。
 * 返回小写指标名 → 数值。
 */
export function extractMetrics(stdout: string): Record<string, number> {
  const metrics: Record<string, number> = {};
  let m: RegExpExecArray | null;
  METRIC_RE.lastIndex = 0;
  while ((m = METRIC_RE.exec(stdout)) !== null) {
    metrics[m[1].toLowerCase()] = parseFloat(m[2]);
  }
  return metrics;
}
