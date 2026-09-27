#!/usr/bin/env python3
"""
将 learning_handbook_zh.md 转换为排版精美的 PDF。
用法：python scripts/build_handbook_pdf.py

修复：
- 自动替换 emoji 为文字标签（微软雅黑不支持 emoji）
- 特殊 Unicode 字符归一化
- 正确注册微软雅黑字体（TTC 格式）
"""

import os
import re
import sys
import platform

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor, white
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle,
    KeepTogether, Preformatted, Flowable
)
from reportlab.platypus import Image as RLImage
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors

# ============================================================
#  配色方案
# ============================================================

C_PRIMARY = HexColor("#5B21B6")
C_ACCENT = HexColor("#7C3AED")
C_TEXT = HexColor("#1F2937")
C_MUTED = HexColor("#6B7280")
C_BG = HexColor("#F9FAFB")
C_BORDER = HexColor("#E5E7EB")
C_CODE_BG = HexColor("#F3F4F6")
C_CODE_TEXT = HexColor("#374151")
C_INFO_BG = HexColor("#EEF2FF")
C_INFO_BORDER = HexColor("#6366F1")
C_WARN_BG = HexColor("#FEF3C7")
C_WARN_BORDER = HexColor("#F59E0B")


# ============================================================
#  Emoji 与特殊字符替换表
# ============================================================

EMOJI_REPLACEMENTS = {
    # 学习手册中的 emoji
    "🧠": "[直觉]",
    "📌": "[核心概念]",
    "✋": "[动手试一试]",
    "⚠️": "[注意]",
    "⚠": "[注意]",
    "💡": "[提示]",
    "🎯": "[目标]",
    "📍": "[定位]",
    "🔍": "[为什么]",
    "📝": "[笔记]",
    "✅": "[✓]",
    "❌": "[✗]",
    "🔧": "[工具]",
    "🚀": "[快速]",
    "⭐": "[★]",
    "🔥": "[重要]",
    "💪": "[加油]",
    "🎓": "[毕业]",
    "📚": "[资料]",
    "🐛": "[bug]",
    "⚡": "[快速]",
    "🎨": "[设计]",
    "🧩": "[拼图]",
    "🏆": "[成就]",
    "📖": "[阅读]",
    "🧪": "[实验]",
    "🎬": "[演示]",
    # Variation selectors
    "\ufe0f": "",
    "\ufe0e": "",
}

# 特殊字符替换（微软雅黑可能不全的符号）
SPECIAL_CHAR_REPLACEMENTS = {
    "—": "—",  # em dash 保留，雅黑支持
    "–": "-",
    "…": "...",
    "→": "→",  # 保留
    "←": "←",
    "↑": "↑",
    "↓": "↓",
    "↔": "<->",
    "≥": ">=",
    "≤": "<=",
    "≠": "!=",
    "≈": "≈",
    "±": "+/-",
    "∞": "∞",
    "√": "√",
    "×": "x",
    "÷": "/",
    "①": "(1)",
    "②": "(2)",
    "③": "(3)",
    "④": "(4)",
    "⑤": "(5)",
    "⑥": "(6)",
    "⑦": "(7)",
    "⑧": "(8)",
    "⑨": "(9)",
    "⑩": "(10)",
    "►": ">",
    "◄": "<",
    "─": "-",
    "━": "=",
    "│": "|",
    "┃": "||",
    "┌": "+",
    "┐": "+",
    "└": "+",
    "┘": "+",
    "├": "+",
    "┤": "+",
    "┬": "+",
    "┴": "+",
    "┼": "+",
    "★": "[★]",
    "☆": "[☆]",
    "●": "●",  # 保留
    "○": "o",
    "◆": "◆",  # 保留
    "◇": "◇",
    "■": "[■]",
    "□": "[ ]",
    "▪": "•",
    "▫": "◦",
    "✓": "✓",  # 保留
    "✗": "✗",  # 保留
    "✓": "✓",
    "✕": "✕",
    "§": "§",
    "¶": "¶",
    "©": "(c)",
    "®": "(r)",
    "™": "(tm)",
    "°": "°",  # 保留
    "²": "^2",
    "³": "^3",
    "½": "1/2",
    "¼": "1/4",
    "¾": "3/4",
    "«": "<<",
    "»": ">>",
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "「": '"',
    "」": '"',
    "『": '"',
    "』": '"',
    "【": "[",
    "】": "]",
    "《": "<",
    "》": ">",
    "〈": "<",
    "〉": ">",
    "·": "·",  # 保留（间隔号）
    "…": "...",
    "\u3000": "  ",  # 全角空格
}


def clean_text_for_pdf(text: str) -> str:
    """清理文本，替换不支持的字符"""
    # 替换 emoji
    for emoji, replacement in EMOJI_REPLACEMENTS.items():
        text = text.replace(emoji, replacement)

    # 替换特殊字符
    for char, replacement in SPECIAL_CHAR_REPLACEMENTS.items():
        text = text.replace(char, replacement)

    # 移除所有非 BMP 字符（emoji 等），替换为 ?
    result = []
    for c in text:
        if ord(c) > 0xFFFF:
            result.append("?")
        else:
            result.append(c)
    text = "".join(result)

    return text


# ============================================================
#  字体注册（中文支持）
# ============================================================

def register_cjk_font():
    """
    注册中文字体，返回 (regular_font_name, bold_font_name)
    优先使用微软雅黑（Windows），其次 PingFang（macOS），最后 Noto Sans CJK（Linux）
    """
    font_candidates = []

    if platform.system() == "Windows":
        font_candidates = [
            # (regular_path, regular_index, bold_path, bold_index)
            ("C:/Windows/Fonts/msyh.ttc", 0, "C:/Windows/Fonts/msyhbd.ttc", 0),
            ("C:/Windows/Fonts/msyh.ttc", 0, "C:/Windows/Fonts/msyh.ttc", 1),
            ("C:/Windows/Fonts/simhei.ttf", 0, "C:/Windows/Fonts/simhei.ttf", 0),
            ("C:/Windows/Fonts/simsun.ttc", 0, "C:/Windows/Fonts/simsun.ttc", 0),
        ]
    elif platform.system() == "Darwin":
        font_candidates = [
            ("/System/Library/Fonts/PingFang.ttc", 0, "/System/Library/Fonts/PingFang.ttc", 1),
            ("/System/Library/Fonts/STHeiti Medium.ttc", 0, "/System/Library/Fonts/STHeiti Medium.ttc", 0),
            ("/Library/Fonts/Arial Unicode.ttf", 0, "/Library/Fonts/Arial Unicode.ttf", 0),
        ]
    else:  # Linux
        font_candidates = [
            ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 0,
             "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 0),
            ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 0,
             "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 0),
            ("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf", 0,
             "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf", 0),
        ]

    for reg_path, reg_idx, bold_path, bold_idx in font_candidates:
        if not os.path.exists(reg_path):
            continue

        try:
            # 注册常规字体
            reg_name = "ZHRegular"
            if reg_path.lower().endswith(".ttc"):
                pdfmetrics.registerFont(TTFont(reg_name, reg_path, subfontIndex=reg_idx))
            else:
                pdfmetrics.registerFont(TTFont(reg_name, reg_path))

            # 注册粗体
            bold_name = "ZHBold"
            if os.path.exists(bold_path):
                if bold_path.lower().endswith(".ttc"):
                    pdfmetrics.registerFont(TTFont(bold_name, bold_path, subfontIndex=bold_idx))
                else:
                    pdfmetrics.registerFont(TTFont(bold_name, bold_path))
            elif reg_path.lower().endswith(".ttc") and bold_idx != reg_idx:
                # 同一个 TTC 文件的不同子字体作为粗体
                try:
                    pdfmetrics.registerFont(TTFont(bold_name, reg_path, subfontIndex=bold_idx))
                except Exception:
                    bold_name = reg_name  # 注册失败就退回
            else:
                bold_name = reg_name  # 退回使用同一字体

            print(f"  ✓ 使用字体: {os.path.basename(reg_path)} (regular={reg_idx}, bold={bold_idx})")
            return reg_name, bold_name

        except Exception as e:
            print(f"  ✗ 字体注册失败 {reg_path}: {e}")
            continue

    raise RuntimeError(
        "找不到可用的中文字体！\n"
        "Windows 请确认 C:\\Windows\\Fonts\\msyh.ttc 存在\n"
        "macOS 请安装 PingFang\n"
        "Linux 请安装 fonts-noto-cjk 或 fonts-wqy-zenhei"
    )


# ============================================================
#  分隔线 Flowable
# ============================================================

class ColoredDivider(Flowable):
    """装饰分隔线"""

    def __init__(self, width, height=1, color=C_PRIMARY, space_after=8):
        super().__init__()
        self._w = width
        self._h = height
        self._color = color
        self.spaceAfter = space_after

    def wrap(self, *args):
        return (self._w, self._h + self.spaceAfter)

    def draw(self):
        self.canv.setFillColor(self._color)
        self.canv.rect(0, 0, self._w, self._h, fill=1, stroke=0)


# ============================================================
#  样式定义
# ============================================================

def build_styles(zh_font, zh_bold_font):
    """构建所有段落样式"""
    s = {}

    # 封面
    s["cover_title"] = ParagraphStyle(
        "CoverTitle", fontName=zh_bold_font, fontSize=32, leading=42,
        textColor=C_PRIMARY, alignment=TA_CENTER, wordWrap="CJK",
        spaceAfter=12,
    )
    s["cover_subtitle"] = ParagraphStyle(
        "CoverSubtitle", fontName=zh_font, fontSize=14, leading=20,
        textColor=C_MUTED, alignment=TA_CENTER, wordWrap="CJK",
        spaceAfter=8,
    )
    s["cover_meta"] = ParagraphStyle(
        "CoverMeta", fontName=zh_font, fontSize=11, leading=16,
        textColor=C_MUTED, alignment=TA_CENTER, wordWrap="CJK",
    )

    # 标题
    s["h1"] = ParagraphStyle(
        "H1", fontName=zh_bold_font, fontSize=20, leading=28,
        textColor=C_PRIMARY, spaceBefore=22, spaceAfter=10,
        wordWrap="CJK",
    )
    s["h2"] = ParagraphStyle(
        "H2", fontName=zh_bold_font, fontSize=15, leading=22,
        textColor=C_ACCENT, spaceBefore=16, spaceAfter=6,
        wordWrap="CJK",
    )
    s["h3"] = ParagraphStyle(
        "H3", fontName=zh_bold_font, fontSize=12.5, leading=18,
        textColor=C_TEXT, spaceBefore=12, spaceAfter=4,
        wordWrap="CJK",
    )

    # 正文
    s["body"] = ParagraphStyle(
        "Body", fontName=zh_font, fontSize=10.5, leading=17,
        textColor=C_TEXT, spaceAfter=6, wordWrap="CJK",
        alignment=TA_JUSTIFY,
    )
    s["body_indent"] = ParagraphStyle(
        "BodyIndent", parent=s["body"], firstLineIndent=21,
    )

    # 图注
    s["caption"] = ParagraphStyle(
        "Caption", fontName=zh_font, fontSize=9, leading=13,
        textColor=C_MUTED, alignment=TA_CENTER, spaceBefore=4, spaceAfter=8,
        wordWrap="CJK",
    )

    # 代码（行内代码用等宽字体）
    s["code"] = ParagraphStyle(
        "Code", fontName="Courier", fontSize=9, leading=13,
        textColor=C_CODE_TEXT, backColor=C_CODE_BG,
        leftIndent=8, rightIndent=8, spaceBefore=4, spaceAfter=4,
        borderPadding=6, wordWrap="CJK",
    )

    # 提示框
    s["callout"] = ParagraphStyle(
        "Callout", fontName=zh_font, fontSize=10.5, leading=17,
        textColor=C_TEXT, leftIndent=14, rightIndent=6,
        spaceBefore=4, spaceAfter=4, wordWrap="CJK",
        backColor=C_INFO_BG, borderPadding=8,
        leftPadding=10,
    )

    # 警告框
    s["warning"] = ParagraphStyle(
        "Warning", parent=s["callout"], backColor=C_WARN_BG,
    )

    return s


# ============================================================
#  Markdown → ReportLab 解析器
# ============================================================

class MarkdownToPDF:
    """将 Markdown 文本解析为 ReportLab Flowable 列表"""

    def __init__(self, styles, zh_font, zh_bold_font, base_dir="."):
        self.styles = styles
        self.zh_font = zh_font
        self.zh_bold_font = zh_bold_font
        # 图片相对路径的解析基准（md 文件所在目录）
        self.base_dir = base_dir

    def parse(self, md_text):
        """解析 Markdown，返回 flowable 列表"""
        # 先清理文本
        md_text = clean_text_for_pdf(md_text)
        lines = md_text.split("\n")
        story = []

        # 状态
        in_code = False
        code_buf = []
        in_quote = False
        quote_buf = []
        in_ul = False
        ul_buf = []
        in_table = False
        table_buf = []

        def flush_code():
            nonlocal in_code, code_buf
            if in_code:
                story.append(Spacer(1, 4))
                code_text = "\n".join(code_buf)
                # 使用中文字体显示代码块（避免中文显示为方块）
                # 英文用 <font face="Courier"> 保持等宽感
                style = ParagraphStyle(
                    "CodeBlock", fontName=self.zh_font, fontSize=9, leading=13,
                    textColor=C_CODE_TEXT, backColor=C_CODE_BG,
                    leftIndent=8, rightIndent=8, borderPadding=8,
                    wordWrap="CJK",
                )
                # 将代码文本转义后逐行用 Paragraph 渲染
                # 用 <br/> 连接各行，空格用 &nbsp; 保持缩进
                import html
                lines_html = []
                for cline in code_text.split("\n"):
                    escaped = html.escape(cline)
                    # 把行首空格替换为 &nbsp; 保持缩进
                    leading_spaces = len(escaped) - len(escaped.lstrip())
                    if leading_spaces > 0:
                        escaped = "&nbsp;" * leading_spaces + escaped.lstrip()
                    lines_html.append(escaped)
                para_text = "<br/>".join(lines_html)
                story.append(Paragraph(para_text, style))
                story.append(Spacer(1, 4))
                in_code = False
                code_buf = []

        def flush_quote():
            nonlocal in_quote, quote_buf
            if in_quote:
                # 用空格连接多行，然后处理行内格式
                text = self._inline(" ".join(quote_buf))
                story.append(Spacer(1, 4))
                story.append(Paragraph(text, self.styles["callout"]))
                story.append(Spacer(1, 6))
                in_quote = False
                quote_buf = []

        def flush_ul():
            nonlocal in_ul, ul_buf
            if in_ul:
                for item in ul_buf:
                    text = "• " + self._inline(item)
                    story.append(Paragraph(text, self.styles["body"]))
                in_ul = False
                ul_buf = []

        def flush_table():
            nonlocal in_table, table_buf
            if in_table and len(table_buf) >= 2:
                # 解析表格行
                rows = []
                for row in table_buf:
                    cells = [c.strip() for c in row.strip("|").split("|")]
                    rows.append(cells)

                # 移除分隔行（|---|---|）
                if len(rows) >= 2 and all("---" in c for c in rows[1]):
                    rows.pop(1)

                if rows and all(len(r) == len(rows[0]) for r in rows):
                    story.append(Spacer(1, 4))
                    # 转 Paragraph
                    para_rows = []
                    for ri, row in enumerate(rows):
                        para_row = []
                        for cell in row:
                            style = ParagraphStyle(
                                f"TC{ri}", fontName=self.zh_font, fontSize=9.5,
                                leading=14, textColor=C_TEXT, wordWrap="CJK",
                            )
                            if ri == 0:
                                style.fontName = self.zh_bold_font
                                style.textColor = white
                            para_row.append(Paragraph(self._inline(cell), style))
                        para_rows.append(para_row)

                    n_cols = len(para_rows[0])
                    col_w = 16.0 / n_cols * cm
                    t = Table(para_rows, colWidths=[col_w] * n_cols, repeatRows=1)
                    t.setStyle(TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0), C_PRIMARY),
                        ("TEXTCOLOR", (0, 0), (-1, 0), white),
                        ("FONTSIZE", (0, 0), (-1, 0), 10),
                        ("FONTSIZE", (0, 1), (-1, -1), 9),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                        ("TOPPADDING", (0, 0), (-1, 0), 8),
                        ("BOTTOMPADDING", (0, 1), (-1, -1), 6),
                        ("TOPPADDING", (0, 1), (-1, -1), 6),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, C_BG]),
                        ("GRID", (0, 0), (-1, -1), 0.5, C_BORDER),
                    ]))
                    story.append(KeepTogether([t]))
                    story.append(Spacer(1, 6))
                in_table = False
                table_buf = []

        # 逐行处理
        i = 0
        while i < len(lines):
            line = lines[i]

            # 代码块
            if line.startswith("```"):
                flush_ul(); flush_quote(); flush_table()
                if not in_code:
                    in_code = True
                else:
                    flush_code()
                i += 1
                continue

            if in_code:
                code_buf.append(line)
                i += 1
                continue

            # 空行
            if line.strip() == "":
                flush_ul(); flush_quote(); flush_table()
                story.append(Spacer(1, 3))
                i += 1
                continue

            # 引用块
            if line.startswith(">"):
                flush_ul(); flush_table()
                in_quote = True
                quote_buf.append(line[1:].strip())
                i += 1
                continue
            else:
                flush_quote()

            # 表格行
            stripped = line.strip()
            if "|" in stripped and stripped.startswith("|") and stripped.endswith("|"):
                flush_ul()
                in_table = True
                table_buf.append(line)
                i += 1
                continue
            else:
                flush_table()

            # 标题
            if line.startswith("# "):
                flush_ul()
                text = line[2:].strip()
                story.append(Spacer(1, 8))
                story.append(Paragraph(self._inline(text), self.styles["h1"]))
                story.append(ColoredDivider(16 * cm, 2, C_PRIMARY, 6))
                i += 1
                continue

            if line.startswith("## "):
                flush_ul()
                text = line[3:].strip()
                story.append(Paragraph(self._inline(text), self.styles["h2"]))
                i += 1
                continue

            if line.startswith("### "):
                flush_ul()
                text = line[4:].strip()
                story.append(Paragraph(self._inline(text), self.styles["h3"]))
                i += 1
                continue

            # 无序列表
            lstripped = line.lstrip()
            if lstripped.startswith(("- ", "* ")):
                in_ul = True
                ul_buf.append(lstripped[2:].strip())
                i += 1
                continue
            else:
                flush_ul()

            # 图片: ![图注](相对路径)
            m_img = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$", stripped)
            if m_img:
                flush_ul()
                alt, rel = m_img.group(1).strip(), m_img.group(2).strip()
                img_path = os.path.join(self.base_dir, rel)
                if os.path.exists(img_path):
                    try:
                        rl_img = RLImage(img_path)
                        max_w, max_h = 15.0 * cm, 17.0 * cm
                        scale = min(max_w / rl_img.drawWidth, max_h / rl_img.drawHeight, 1.0)
                        rl_img.drawWidth *= scale
                        rl_img.drawHeight *= scale
                        rl_img.hAlign = "CENTER"
                        story.append(Spacer(1, 4))
                        story.append(rl_img)
                        if alt:
                            cap_style = ParagraphStyle(
                                "ImgCaption", fontName=self.zh_font, fontSize=8.5,
                                leading=12, textColor=C_MUTED, alignment=TA_CENTER,
                                spaceAfter=8,
                            )
                            story.append(Paragraph(self._inline(alt), cap_style))
                        story.append(Spacer(1, 4))
                    except Exception as e:  # 图片损坏也不让整本手册构建失败
                        story.append(Paragraph(
                            f"[图片渲染失败: {alt} ({e})]", self.styles["body"]))
                else:
                    story.append(Paragraph(f"[缺图: {rel}]", self.styles["body"]))
                i += 1
                continue

            # 水平线
            if re.match(r"^-{3,}$", stripped):
                story.append(Spacer(1, 4))
                story.append(ColoredDivider(16 * cm, 0.5, C_BORDER, 4))
                i += 1
                continue

            # 普通段落
            if stripped:
                story.append(Paragraph(self._inline(stripped), self.styles["body"]))

            i += 1

        # 收尾
        flush_code(); flush_quote(); flush_ul(); flush_table()
        return story

    def _inline(self, text):
        """处理行内格式：粗体/斜体/代码/链接"""
        import html

        # 转义 HTML 特殊字符（先转义，再处理行内格式）
        text = html.escape(text)

        # 行内代码：用背景色标识，不用 Courier（避免中文变方块）
        # 用 <font> 加背景色的方式不可行，所以用颜色 + 字号略小来标识
        text = re.sub(
            r"`([^`]+)`",
            r'<font color="#DC2626" size="9">\1</font>',
            text,
        )

        # 粗体
        text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)

        # 斜体
        text = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", text)

        # 链接（只保留文字，去掉 URL，PDF 里点击不了）
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r'<font color="#6366F1">\1</font>', text)

        return text


# ============================================================
#  页眉页脚
# ============================================================

def on_page(canvas, doc):
    """每页的页眉页脚"""
    page_num = canvas.getPageNumber()
    # 封面（第1页）不加页脚
    if page_num <= 1:
        return

    canvas.saveState()
    canvas.setFont("ZHRegular", 8)
    canvas.setFillColor(C_MUTED)

    # 页码
    content_page = page_num - 1  # 封面不计
    canvas.drawCentredString(A4[0] / 2, 1.5 * cm, f"— {content_page} —")

    # 右侧书名
    canvas.drawRightString(A4[0] - 2 * cm, 1.5 * cm, "TF2 学习手册")

    # 顶部细线
    canvas.setStrokeColor(C_BORDER)
    canvas.setLineWidth(0.5)
    canvas.line(2 * cm, A4[1] - 1.5 * cm, A4[0] - 2 * cm, A4[1] - 1.5 * cm)

    canvas.restoreState()


# ============================================================
#  主函数
# ============================================================

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    md_path = os.path.join(project_root, "docs", "learning_handbook_zh.md")
    output_path = os.path.join(project_root, "docs", "learning_handbook_zh.pdf")

    print("=" * 50)
    print("  TF2 学习手册 PDF 生成")
    print("=" * 50)

    # 1. 读取 Markdown
    print(f"\n[1/4] 读取 Markdown...")
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()
    print(f"  ✓ {len(md_text)} 字符")

    # 2. 注册字体
    print(f"\n[2/4] 注册中文字体...")
    zh_font, zh_bold_font = register_cjk_font()

    # 3. 构建样式与解析
    print(f"\n[3/4] 解析 Markdown...")
    styles = build_styles(zh_font, zh_bold_font)
    parser = MarkdownToPDF(styles, zh_font, zh_bold_font, base_dir=os.path.dirname(md_path))
    body_flowables = parser.parse(md_text)
    print(f"  ✓ {len(body_flowables)} 个 flowable")

    # 4. 生成 PDF
    print(f"\n[4/4] 生成 PDF...")
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=2.2 * cm,
        rightMargin=2.2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="TensorFlow 2 学习手册",
        author="tf2-tutorial",
        subject="深度学习入门教程",
    )

    story = []

    # —— 封面 ——
    story.append(Spacer(1, 3.5 * cm))
    story.append(Paragraph("TensorFlow 2", styles["cover_title"]))
    story.append(Paragraph("学习手册", styles["cover_title"]))
    story.append(Spacer(1, 0.6 * cm))
    story.append(ColoredDivider(8 * cm, 3, C_PRIMARY, 20))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("从零读懂深度学习与 TensorFlow 2", styles["cover_subtitle"]))
    story.append(Spacer(1, 4 * cm))
    story.append(Paragraph("面向机器学习零基础的计算机专业学生", styles["cover_meta"]))
    story.append(Paragraph("tf2-tutorial 项目配套手册", styles["cover_meta"]))
    story.append(PageBreak())

    # —— 正文 ——
    story.extend(body_flowables)

    # 构建
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)

    # 统计
    from pypdf import PdfReader

    reader = PdfReader(output_path)
    n_pages = len(reader.pages)
    file_size = os.path.getsize(output_path) / 1024

    print(f"\n{'=' * 50}")
    print(f"  ✓ 完成：{n_pages} 页，{file_size:.0f} KB")
    print(f"  ✓ 输出：{output_path}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
