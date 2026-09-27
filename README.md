# tf2-tutorial — TensorFlow 2 教学项目

![python](https://img.shields.io/badge/python-3.9%2B-blue)
![tf](https://img.shields.io/badge/tensorflow-2.16%2B-orange)
![license](https://img.shields.io/badge/license-Apache_2.0-blue)
![stage](https://img.shields.io/badge/stage-teaching--only-green)

> 一个面向**机器学习零基础的计算机专业学生**、可在 CPU 上跑通样例的 TensorFlow 2 教学项目。
> 代码以"可读优先、可运行其次"组织，每个章节对应一个独立可执行脚本。
> 追求 **"广、全、实"**：知识面广、体系完整、实战导向。

---

## 1. 项目目标

- 用最小的代码量讲清 TF2 核心概念：**张量、自动求导、Keras、训练循环、回调、数据管道、模型保存**；
- 面向机器学习零基础的计算机专业学生，从"什么是机器学习"讲起，逐层深入；
- 所有章节均提供"纯 CPU 也能跑通"的最小例子，部分章节附带 GPU/Colab 加速说明；
- 模块化设计：`src/tf2tutorial/` 提供可复用组件，`chapters/` 是教学入口；
- 与 `F:\Steganography` 保持相同的工程结构：docs / src / tests / notebooks / scripts / data / models / output / figures / teaching。

## 2. 章节地图

| 章节 | 主题 | 入口脚本 | 数据集 | 预计耗时 (CPU) | 已验证 |
| --- | --- | --- | --- | --- | --- |
| **00** | **机器学习基础概念** | `chapters/00_ml_basics.py` | 无 | < 10 s | ✅ 零 TF 依赖 |
| 01 | 张量与自动求导 | `chapters/01_tensors_autograd.py` | 无 | < 10 s | ✅ |
| 02 | 线性回归 (Keras Sequential) | `chapters/02_linear_regression.py` | 合成数据 | < 10 s | ✅ |
| 03 | MLP 分类 MNIST | `chapters/03_mlp_mnist.py` | MNIST | 2-5 min | ✅ test_acc 96.97 % |
| 04 | CNN 分类 CIFAR-10 | `chapters/04_cnn_cifar10.py` | CIFAR-10 | 首次下载约 170 MB，训练约 2 min | ✅ test_acc 72.64 %（本机 CPU） |
| 05 | 文本分类 (IMDB) | `chapters/05_text_imdb.py` | IMDB | 2-5 min | ✅ test_acc 83.90 % |
| 06 | 迁移学习 (MobileNetV2) | `chapters/06_transfer_learning.py` | tf_flowers | 3-8 min | ✅ val_acc 91.27 %（微调后） |
| 07 | 回调函数 / TensorBoard | `chapters/07_callbacks_tensorboard.py` | MNIST | 2-5 min | ✅ val_acc 97.55 % |
| 08 | 保存与部署 (SavedModel / TFLite) | `chapters/08_save_and_export.py` | MNIST | 1-2 min | ✅ .keras + SavedModel + .tflite 全生成 |
| 09 | 综合项目：图像分类系统 | `chapters/09_capstone_image_classifier.py` | tf_flowers | 3-5 min | ✅ val_acc 90.00 %（微调后） |
| **10** | **边缘计算与树莓派部署** | `chapters/10_edge_raspberry_pi.py` | tf_flowers | 1-2 min | ✅ 三种量化 + 基准测试 |
| **11** | **tf.data 数据管道 (进阶)** | `chapters/11_tfdata_pipeline.py` | 合成数据 | ~20 s | ✅ cache 实测 2.1x + DCE 反例 |
| **12** | **序列建模 RNN/LSTM (进阶)** | `chapters/12_rnn_timeseries.py` | 合成正弦 | ~1 min | ✅ 四模型对照 (含 Dense 基线) |
| **13** | **手写注意力/Transformer (进阶)** | `chapters/13_attention_transformer.py` | 合成数据 | ~30 s | ✅ test_acc 96.8% + 注意力热力图 |
| **14** | **生成模型 AE+GAN (进阶)** | `chapters/14_autoencoder_gan.py` | MNIST | 2-4 min | ✅ AE 重建 + GAN 3000 步 |
| **15** | **自定义训练与性能 (进阶)** | `chapters/15_custom_training.py` | 合成数据 | ~1 min | ✅ Huber 鲁棒实验 + graph 4.6x |
| **16** | **进阶篇毕业项目: 文本分类 (进阶)** | `chapters/16_text_capstone.py` | IMDB(缓存) | 3-5 min | ✅ tf.data+注意力+tape 训练, acc 85.4% / F1 0.85 |

**VS Code 插件**：`extension/` 内置「TF 学习伴侣」——讲解/运行/报错翻译/自测/进度一体化的学习面板（已随本仓库打包 `tf-learning-companion-0.1.0.vsix`，详见 [extension/README.md](extension/README.md) 与 [extension/DESIGN.md](extension/DESIGN.md)）。

### 教学配套材料

| 材料 | 位置 | 说明 |
| --- | --- | --- |
| **知识地图** | `docs/knowledge_map.md` | 主线故事 + 全景路线 + 章节依赖图 + 三条学习路径 + 每章过关标准 |
| 学习手册（中文） | `docs/learning_handbook_zh.md` | 12 章 + 实战 09/10 + 进阶篇 11-16 + 附录 A-I（反例索引/版本迁移/数学补充/术语表/FAQ/评估指标/进阶方向），42 幅插图 + 17 张字符图，含直觉解释、为什么这样设计、反例教室（含索引表）、过关自测、如何读报错 |
| **学习手册 Word 版** | `docs/learning_handbook_zh.docx` | 可自行编辑批注的 Word 源文件 |
| **学习手册 PDF** | `docs/learning_handbook_zh.pdf` | Word 排版渲染（约 95 页，40 幅插图，含可更新目录），可打印阅读 |
| **14 天学习计划** | `docs/study_plan.md` | 每天 1-2 小时，含每日目标 + 动手任务 + 过关自测 |
| **16 周学期制课程表** | `docs/study_plan_semester.md` | 进阶篇 5 专题 + 调参调试 + 4 周大项目 + 自评分表 |
| **知识点自检清单** | `docs/knowledge_checklist.md` | 0-9 章共 60+ 个知识点，学完打勾 |
| 术语表 | `docs/glossary_zh.md` | 40+ 术语，中英对照 + 直觉类比 |
| 调试手册 | `docs/FAQ.md` | 10 章 / 54 个问题 / 18 条报错速查 |
| **树莓派部署指南** | `docs/raspberry_pi_deployment_zh.md` | 9 章完整实战指南，从刷系统到实时推理 |
| 练习题集 | `teaching/exercises.md` | 主线 + 进阶篇共 60+ 道题，含参考答案 |
| 教学大纲 | `teaching/chapter_outline.md` | 一页可打印的章节地图 + 概念覆盖矩阵 |
| **自己动手项目模板** | `teaching/project_template.md` | 7 步从零搭项目 + 报告模板 + 常见陷阱 |
| 章节笔记本 | `notebooks/*.ipynb` | 16 个 Jupyter Notebook，自动从章节生成 |

## 3. 快速开始

本机 TensorFlow 装在虚拟环境 `F:\tf2-env`（Python 3.12 + TF 2.21，仅 CPU）。
有两种使用方式：

### 方式 A：直接用 `F:\tf2-env` 的解释器

```powershell
F:\tf2-env\Scripts\python.exe scripts\env_check.py
F:\tf2-env\Scripts\python.exe chapters\03_mlp_mnist.py
```

### 方式 B：通过转发脚本（推荐，无需记忆路径）

```powershell
python scripts\use_tf2_env.py scripts\env_check.py
python scripts\use_tf2_env.py chapters\03_mlp_mnist.py
python scripts\use_tf2_env.py chapters\00_ml_basics.py
```

### 方式 C：在当前 PowerShell 中激活 venv

```powershell
cd F:\Tensorflow
F:\tf2-env\Scripts\Activate.ps1
python chapters\03_mlp_mnist.py
```

### 方式 D：把项目依赖装到当前 base Python

```powershell
cd F:\Tensorflow
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python chapters\00_ml_basics.py    # 只需要 numpy + matplotlib
python chapters\03_mlp_mnist.py    # 装上 TF 才能跑
```

使用 Makefile（也支持 Windows 的 `make`，若已安装 GNU Make）：

```bash
make install      # 安装依赖
make chapter00    # 跑第 0 章
make test         # 跑测试
make clean        # 清理 output / __pycache__
```

## 4. 工程结构

```
F:\Tensorflow\
├── README.md                  # 你正在读的文件
├── Makefile                   # make 常用命令
├── pyproject.toml             # 包配置（pip install -e . 可安装）
├── requirements.txt           # 依赖列表
├── run_tests.py               # 一键跑测试
├── run_all_chapters.py        # 一键跑全部章节（Windows 友好）
├── .gitignore
├── LICENSE
├── chapters/                  # 教学入口（每章一个独立脚本）
│   ├── 00_ml_basics.py
│   ├── 01_tensors_autograd.py
│   ├── 02_linear_regression.py
│   ├── 03_mlp_mnist.py
│   ├── 04_cnn_cifar10.py
│   ├── 05_text_imdb.py
│   ├── 06_transfer_learning.py
│   ├── 07_callbacks_tensorboard.py
│   └── 08_save_and_export.py
│   ├── 09_capstone_image_classifier.py
│   ├── 10_edge_raspberry_pi.py
│   ├── 11_tfdata_pipeline.py      # 进阶: 数据管道
│   ├── 12_rnn_timeseries.py       # 进阶: 序列建模
│   ├── 13_attention_transformer.py # 进阶: 手写注意力
│   ├── 14_autoencoder_gan.py      # 进阶: 生成模型
│   └── 15_custom_training.py      # 进阶: 自定义训练与性能
├── src/tf2tutorial/           # 可复用 Python 包
│   ├── config.py              # 路径 / 超参数
│   ├── data.py                # 数据集加载
│   ├── models.py              # 模型构建函数
│   ├── training.py            # 训练辅助
│   ├── utils.py               # 计时器 / 种子 / 通用工具
│   ├── visualize.py           # 画图
│   └── chapters_dispatch.py   # python -m tf2tutorial.chapters_dispatch 01
├── docs/                      # 文档
│   ├── README.md              # 文档索引
│   ├── learning_handbook_zh.md
│   ├── learning_handbook_en.md
│   ├── learning_handbook_zh.pdf
│   ├── glossary_zh.md
│   ├── FAQ.md
│   ├── study_plan.md
│   ├── knowledge_checklist.md
│   └── raspberry_pi_deployment_zh.md
├── teaching/                  # 教学辅助（给老师/作者）
│   ├── README.md
│   ├── chapter_outline.md
│   ├── exercises.md
│   ├── project_template.md
│   ├── gen_chapter_figures.py # 校验各章产物并生成 figures/figures_index.md
│   ├── build_notebooks.py     # 从 chapters 生成 notebooks/*.ipynb
│   ├── build_handbook.py
│   └── figures/               # 占位目录（章节图统一输出到项目根 figures/）
├── tests/                     # 测试（无需 TF 也可跑，TF 相关测试自动跳过）
│   ├── conftest.py
│   ├── test_utils.py
│   ├── test_config.py
│   ├── test_data.py
│   ├── test_models.py
│   ├── test_visualize.py
│   └── test_chapter01_runs.py # 第 1 章端到端冒烟测试
├── notebooks/                 # Jupyter 笔记本（16 个，从 chapters 自动生成）
├── scripts/                   # 辅助脚本
│   ├── env_check.py
│   ├── use_tf2_env.py
│   ├── download_data.py
│   ├── build_handbook_word.py # 学习手册 Markdown → Word → PDF（主推，pandoc+Word）
│   └── build_handbook_pdf.py  # 旧版 reportlab 直排（备用）
│   ├── smoke_chapter04.py
│   └── smoke_chapter04_offline.py
├── data/                      # 数据（原始 / 处理后）
│   ├── README.md
│   ├── raw/
│   ├── processed/
│   └── tfds/
├── models/                    # 保存的模型
│   └── saved/
├── output/                    # 训练日志 / 检查点 / 中间结果
│   ├── logs/
│   └── checkpoints/
├── figures/                   # 训练曲线图 / 示意图（各章节产物）
├── extension/                 # VS Code 插件「TF 学习伴侣」（讲解+运行+报错翻译+进度）
├── deploy/                    # 树莓派部署包（可直接拷到 Pi 上运行）
│   ├── pi_classify.py         # 独立推理脚本（单图/基准/摄像头 三种模式）
│   ├── labels_flowers.txt     # 花朵分类标签
│   └── README.md              # 部署说明
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions：安装依赖 + pytest
```

## 5. 学习建议

1. **先看知识地图**：打开 `docs/knowledge_map.md`，搞清主线、自己在哪、为什么这么走；
2. **按 14 天计划走**：打开 `docs/study_plan.md`，跟着每天的安排学，完成一项打一个勾；
3. **从第 0 章开始**：如果你完全没接触过机器学习，先跑第 0 章建立直觉；
4. **读一段，跑一段**：每章先读代码注释和打印输出，再跑代码，观察结果；
5. **动手试一试**：每章末尾都有 2-3 个小实验建议，改参数、看变化；
6. **做练习题 + 自检**：`teaching/exercises.md` 练手，`docs/knowledge_checklist.md` 检查掌握程度，每章的「过关自测」达标了再进下一章；
7. **查术语表**：遇到不懂的术语，去 `docs/glossary_zh.md` 查直觉解释；
8. **遇到问题先查调试手册**：`docs/FAQ.md` 有 54 个常见问题 + 18 条报错速查；
9. **做自己的项目**：用 `teaching/project_template.md` 当模板，搭一个你自己感兴趣的图像分类项目；
10. **看学习手册**：`docs/learning_handbook_zh.md` 有更详细的概念讲解和调参指南。
11. **主线毕业之后**：进阶篇 11-15 章继续进阶，`docs/study_plan_semester.md` 提供一整个学期的系统路线——学习周期不设上限。

## 6. 反馈与求助

- 遇到报错：先看手册**第 12 章「如何读报错」**与**附录 G FAQ 速答**；
- 要提问：用手册 **9.6 节的提问模板**（最小可复现 + 完整报错 + 版本号）提 Issue；
- 想知道项目改了什么：看 [CHANGELOG.md](CHANGELOG.md) 版本更新日志；
- 版本兼容问题：跑 `python scripts/env_check.py` 自检，对照手册附录 D。

## 7. 许可

Apache License 2.0，详见 `LICENSE` 文件。
