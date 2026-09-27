# -*- coding: utf-8 -*-
"""Generate extension chapter content (content/chapters/*.json) from the handbook.

单一事实来源原则（设计要点·风险表）：手册 docs/learning_handbook_zh.md 是唯一内容源，
本脚本把每章解析成结构化 JSON，供 VS Code 插件的 Webview 讲解面板 / 自测 / 知识地图使用。
手册更新后重跑：python extension/scripts/generate_content.py

解析目标（与手册的五板块教学结构一一对应）：
    目标(🎯/**目标：**) → goals
    📌 核心概念          → concepts
    ✋ 动手试一试        → experiments
    ⚠️ 常见坑            → pitfalls
    ❌ 反例教室          → counterExamples（标题 + 一句话教训）
    ✅ 过关自测·基础题   → quiz（含【答】的题拆成 question/answer）
    **过关标准**         → passCriteria
    📚 延伸阅读          → reading
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MD = ROOT / "docs" / "learning_handbook_zh.md"
OUT = Path(__file__).resolve().parents[1] / "content"

# 章 id → (手册 section 标题前缀, 章号, 显示标题)
SECTIONS = [
    ("00_ml_basics",            "## 0. 路线图", "00", "第 0 章 · 路线图与环境"),
    ("01_tensors_autograd",     "## 1. 第 01 章", "01", "第 1 章 · 张量与自动求导"),
    ("02_linear_regression",    "## 2. 第 02 章", "02", "第 2 章 · 用 Keras 写第一个模型"),
    ("03_mlp_mnist",            "## 3. 第 03 章", "03", "第 3 章 · MLP 分类 MNIST"),
    ("04_cnn_cifar10",          "## 4. 第 04 章", "04", "第 4 章 · CNN 入门 (CIFAR-10)"),
    ("05_text_imdb",            "## 5. 第 05 章", "05", "第 5 章 · 文本分类 (IMDB)"),
    ("06_transfer_learning",    "## 6. 第 06 章", "06", "第 6 章 · 迁移学习 (tf_flowers)"),
    ("07_callbacks_tensorboard","## 7. 第 07 章", "07", "第 7 章 · 回调函数与 TensorBoard"),
    ("08_save_and_export",      "## 8. 第 08 章", "08", "第 8 章 · 保存与导出"),
    ("09_capstone",             "## 实战 09", "09", "实战 09 · 综合项目"),
    ("10_edge_raspberry_pi",    "## 实战 10", "10", "实战 10 · 边缘部署"),
    ("11_tfdata_pipeline",      "## 进阶篇 11", "11", "进阶篇 11 · tf.data 数据管道"),
    ("12_rnn_timeseries",       "## 进阶篇 12", "12", "进阶篇 12 · 序列建模"),
    ("13_attention_transformer","## 进阶篇 13", "13", "进阶篇 13 · 手写注意力"),
    ("14_autoencoder_gan",      "## 进阶篇 14", "14", "进阶篇 14 · 生成模型"),
    ("15_custom_training",      "## 进阶篇 15", "15", "进阶篇 15 · 自定义训练与性能"),
    ("16_text_capstone",        "## 进阶篇毕业项目", "16", "第 16 章 · 进阶毕业项目"),
]

# 知识地图依赖（来自 docs/knowledge_map.md 的章节依赖图）
DEPS = {
    "00_ml_basics": [],
    "01_tensors_autograd": ["00_ml_basics"],
    "02_linear_regression": ["01_tensors_autograd"],
    "03_mlp_mnist": ["02_linear_regression"],
    "04_cnn_cifar10": ["03_mlp_mnist"],
    "05_text_imdb": ["03_mlp_mnist"],
    "06_transfer_learning": ["04_cnn_cifar10"],
    "07_callbacks_tensorboard": ["02_linear_regression"],
    "08_save_and_export": ["06_transfer_learning", "07_callbacks_tensorboard"],
    "09_capstone": ["01_tensors_autograd", "02_linear_regression", "03_mlp_mnist",
                    "04_cnn_cifar10", "05_text_imdb", "06_transfer_learning",
                    "07_callbacks_tensorboard", "08_save_and_export"],
    "10_edge_raspberry_pi": ["08_save_and_export"],
    "11_tfdata_pipeline": [],
    "12_rnn_timeseries": ["02_linear_regression"],
    "13_attention_transformer": ["12_rnn_timeseries"],
    "14_autoencoder_gan": ["03_mlp_mnist"],
    "15_custom_training": ["01_tensors_autograd", "11_tfdata_pipeline"],
    "16_text_capstone": ["11_tfdata_pipeline", "12_rnn_timeseries",
                         "13_attention_transformer", "15_custom_training"],
}

ENTRY_FALLBACK = {
    "00_ml_basics": "chapters/00_ml_basics.py",
    "01_tensors_autograd": "chapters/01_tensors_autograd.py",
    "02_linear_regression": "chapters/02_linear_regression.py",
    "03_mlp_mnist": "chapters/03_mlp_mnist.py",
    "04_cnn_cifar10": "chapters/04_cnn_cifar10.py",
    "05_text_imdb": "chapters/05_text_imdb.py",
    "06_transfer_learning": "chapters/06_transfer_learning.py",
    "07_callbacks_tensorboard": "chapters/07_callbacks_tensorboard.py",
    "08_save_and_export": "chapters/08_save_and_export.py",
    "09_capstone": "chapters/09_capstone_image_classifier.py",
    "10_edge_raspberry_pi": "chapters/10_edge_raspberry_pi.py",
    "11_tfdata_pipeline": "chapters/11_tfdata_pipeline.py",
    "12_rnn_timeseries": "chapters/12_rnn_timeseries.py",
    "13_attention_transformer": "chapters/13_attention_transformer.py",
    "14_autoencoder_gan": "chapters/14_autoencoder_gan.py",
    "15_custom_training": "chapters/15_custom_training.py",
    "16_text_capstone": "chapters/16_text_capstone.py",
}


def section_range(lines: list[str], header_prefix: str) -> tuple[int, int]:
    """返回 [start, end) 行号（1-based，含头不含尾）。"""
    start = next(i for i, l in enumerate(lines) if l.startswith(header_prefix))
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("## "):
            end = j
            break
    return start + 1, end + 1  # 1-based


def block(lines: list[str], header: str) -> list[str]:
    """取 '### ... header ...' 到下一个 '### ' 之间的行（header 为子串匹配，
    兼容 '### 📌 核心概念' 这类带 emoji 的标题）。"""
    out: list[str] = []
    on = False
    for l in lines:
        if l.startswith("### "):
            on = header in l
            continue
        if on:
            out.append(l)
    return out


def bullets(rows: list[str]) -> list[str]:
    """收集列表项（含续行），去掉 markdown 粗体标记。"""
    items: list[str] = []
    for l in rows:
        t = l.rstrip()
        if re.match(r"^\s*([-*]|\d+\.)\s+", t):
            items.append(re.sub(r"^\s*([-*]|\d+\.)\s+", "", t).strip())
        elif items and t.startswith(("  ", "\t")) and t.strip():
            items[-1] += " " + t.strip()
    return [re.sub(r"\*\*([^*]+)\*\*", r"\1", x) for x in items if x]


def parse_section(text: str, chapter_id: str) -> dict:
    lines = text.split("\n")
    ch: dict = {}

    # 代码入口
    m = re.search(r"代码入口：`([^`]+)`", text)
    ch["entry"] = m.group(1) if m else ENTRY_FALLBACK.get(chapter_id, "")

    # 目标：🎯 块第一段，或 **目标：** 行
    m = re.search(r"### 🎯 本章目标\n\n(.+?)(?:\n\n```|\n\n### )", text, re.S)
    if m:
        ch["goals"] = re.sub(r"\n +", " ", m.group(1)).strip().split("\n")[0][:200]
    else:
        m2 = re.search(r"\*\*目标：\*\*\s*(.+)", text)
        ch["goals"] = m2.group(1).strip() if m2 else "通读本章并跑通入口脚本。"

    ch["concepts"] = bullets(block(lines, "核心概念"))[:8]

    exp = bullets(block(lines, "动手试一试"))
    ch["experiments"] = [re.sub(r"\*\*([^*]+)\*\*", r"\1", e)[:160] for e in exp][:6]

    ch["pitfalls"] = bullets(block(lines, "常见坑"))[:8]

    # 反例：**反例 N：标题** … 取"一句话教训"，退化为 "> 修复/教训" 引用行
    ce = []
    cur_title = None
    fallback = None
    for l in lines:
        m = re.match(r"\*\*反例 (\d+)：(.+?)\*\*", l.strip())
        if m:
            if cur_title and fallback:
                ce.append({"title": cur_title, "lesson": fallback})
            cur_title = f"反例 {m.group(1)}：{m.group(2)}"
            fallback = None
            continue
        if cur_title:
            m2 = re.match(r"\s*[-*]\s*\*\*一句话教训\*\*[：:](.+)", l)
            if m2:
                ce.append({"title": cur_title, "lesson": m2.group(1).strip()})
                cur_title = None
                fallback = None
                continue
            m3 = re.match(r">\s*(?:修复|教训|工程含义|记忆口诀|排查法|启示)[：:](.+)", l)
            if m3 and not fallback:
                fallback = m3.group(1).strip()
            elif l.startswith("> ") and not fallback:
                fallback = l[2:].strip()
            m4 = re.match(r"[-*]\s*\*\*一句话教训\*\*[：:](.+)", l.strip())
            if m4:
                ce.append({"title": cur_title, "lesson": m4.group(1).strip()})
                cur_title = None
                fallback = None
    if cur_title and fallback:
        ce.append({"title": cur_title, "lesson": fallback})
    ch["counterExamples"] = ce

    # 过关标准（含续行）
    m = re.search(r"\*\*过关标准\*\*[：:](.+?)(?=\n\n|\n### |$)", text, re.S)
    ch["passCriteria"] = re.sub(r"\s+", " ", m.group(1)).strip() if m else ""

    # 自测·基础题（含【答】的编号项；缩进续行并入上一题）
    quiz = []
    cur: list[str] = []
    for l in block(lines, "过关自测"):
        t = l.rstrip()
        if re.match(r"^\s*\d+\.\s+", t):
            if cur:
                quiz.append(" ".join(cur))
            cur = [re.sub(r"^\s*\d+\.\s+", "", t).strip()]
        elif cur and t.strip():
            cur.append(t.strip())
    if cur:
        quiz.append(" ".join(cur))
    parsed = []
    for item in quiz:
        if "【答】" in item:
            q, _, a = item.partition("【答】")
            parsed.append({
                "question": re.sub(r"\*\*", "", q).strip(),
                "answer": re.sub(r"\*\*", "", a).strip(),
            })
    ch["quiz"] = parsed[:6]

    ch["reading"] = bullets(block(lines, "延伸阅读"))[:6]
    return ch


def main() -> int:
    lines = MD.read_text(encoding="utf-8").split("\n")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "chapters").mkdir(parents=True, exist_ok=True)

    index = []
    for chapter_id, header, num, title in SECTIONS:
        start, end = section_range(lines, header)
        text = "\n".join(lines[start - 1: end - 1])
        ch = parse_section(text, chapter_id)
        ch.update({
            "id": chapter_id,
            "num": num,
            "title": title,
            "deps": DEPS[chapter_id],
            "mdRange": [start, end],
        })
        out = OUT / "chapters" / f"{chapter_id}.json"
        out.write_text(json.dumps(ch, ensure_ascii=False, indent=2), encoding="utf-8")
        index.append({
            "id": chapter_id, "num": num, "title": title,
            "entry": ch["entry"], "deps": DEPS[chapter_id],
        })
        quiz_n = len(ch["quiz"])
        ce_n = len(ch["counterExamples"])
        print(f"  {chapter_id:26s} quiz={quiz_n} 反例={ce_n} "
              f"概念={len(ch['concepts'])} 实验={len(ch['experiments'])} 坑={len(ch['pitfalls'])}")

    (OUT / "chapters.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"generated {len(index)} chapters -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
