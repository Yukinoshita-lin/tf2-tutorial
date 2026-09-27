# teaching/

教学专用工具与产物。本目录模仿 `F:\Steganography\teaching\`，把"教材生产"和"代码教程"分离开。

| 子目录 / 文件 | 作用 |
| --- | --- |
| `chapter_outline.md` | 0-10 章共 11 章的教学大纲（一页可打印，含概念覆盖矩阵） |
| `exercises.md` | 0-10 章练习题集（40+ 题，含参考答案） |
| `project_template.md` | 自己动手项目模板：7 步从零搭项目 + 报告模板 |
| `gen_chapter_figures.py` | 校验各章节声明的产物（图 / 模型）是否齐全，并生成 `figures/figures_index.md` |
| `build_notebooks.py` | 从 `chapters/*.py` 生成 `notebooks/NN_*.ipynb`（runpy 包装，不复制代码） |
| `build_handbook.py` | 把 `docs/learning_handbook_*.md` 编译成单文件 PDF（需要安装 `markdown` + `weasyprint` 或 `pandoc`） |
| `figures/` | 占位目录；章节产物图统一输出到项目根 `figures/` |

## 与 `chapters/` 的关系

- `chapters/`：**给读者**（可执行的最小示例）
- `teaching/`：**给作者**（讲义生成工具）
- `docs/`：最终的学习手册（产物）

## 快速使用

```powershell
# 重新生成所有章节图（按需）
python scripts\use_tf2_env.py teaching\gen_chapter_figures.py

# 从章节脚本生成 Jupyter 笔记本
python teaching\build_notebooks.py

# 生成 PDF 学习手册
pip install markdown weasyprint
python teaching\build_handbook.py --lang zh --out docs\learning_handbook_zh.pdf
```
