# -*- coding: utf-8 -*-
"""Sync DESIGN.md test matrix (one-off maintenance script)."""
from pathlib import Path

p = Path(__file__).resolve().parents[1] / "DESIGN.md"
s = p.read_text(encoding="utf-8")

T = "`"  # 反引号

anchor_row = f"| {T}test/projectRoot.test.mjs{T}（5） | 仓库根向上探测（单文件模式）：从章节脚本/子目录找到根、无根 undefined、混合路径 | 0.1.5 单文件模式修复 |"
assert anchor_row in s, "projectRoot row anchor missing"

new_rows = "\n".join([
    f"| {T}test/extract.test.mjs{T}（9） | 单元格区间（含标记行语义，与 VS Code Jupyter 一致）+ 指标提取（同名取末值） | 功能自检新增 |",
    f"| {T}test/figures.test.mjs{T}（2） | 图表捕获：新图捕获/旧图忽略/非图片忽略/目录缺失安全 | 功能自检新增 |",
    f"| {T}test/html.test.mjs{T}（3） | Webview HTML：CSP+nonce、五 Tab、无闭合标签破坏 | 功能自检新增 |",
    f"| {T}test/envCheck.test.mjs{T}（4） | 环境自检真实调度、解释器探测、UTF-8 环境变量 | 功能自检新增 |",
    f"| {T}test/tfLiveNoise.test.mjs{T}（1） | 真机导入 TF 的 stderr 走完整过滤管线（端到端） | 功能自检新增 |",
])
s = s.replace(anchor_row, anchor_row + "\n" + new_rows, 1)

s = s.replace("测试总数 34 项，全部通过。", "测试总数 53 项，全部通过。", 1)
s = s.replace("两轮迭代各抓到 1 个真 bug——", "三轮迭代共抓到 2 个真 bug——", 1)
s = s.replace(
    "（numpy 2.x 矩阵乘法报错文案变更导致翻译库失配，已兼容新旧文案）和 5 个测试自身缺陷。",
    "（numpy 2.x 矩阵乘法报错文案变更导致翻译库失配、openEntry 消息路由 case 漏失），"
    "均已修复并加回归测试；另修测试自身缺陷 7 个。",
    1,
)
ci_line = f"- CI：{T}.github/workflows/ci.yml{T} 新增 extension job（安装→编译→测试）。"
assert ci_line in s, "ci line anchor missing"
s = s.replace(ci_line, ci_line + "\n- 内容生成脚本验证幂等（两次生成 diff 为空），手册→插件同步可信。", 1)

p.write_text(s, encoding="utf-8")
print("DESIGN.md synced")
