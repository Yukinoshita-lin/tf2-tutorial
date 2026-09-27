# TF 学习伴侣 (TensorFlow Learning Companion)

> 在 VS Code 中提供"**读、跑、改、看、查、测**"一体化的 TensorFlow 学习环境。
> 手册（`docs/learning_handbook_zh.md`）是系统教材，本插件是交互式伴侣。

![stage](https://img.shields.io/badge/version-0.1.0_MVP-blue)

## 功能总览

| 功能 | 说明 | 对应手册 |
| --- | --- | --- |
| 📖 **讲解面板** | 五 Tab（讲解/输出/图表/自测/进度），渲染手册原文 + 结构化摘要 | 全书五板块 |
| ▶️ **一键运行** | `Ctrl+Enter` 整文件 / `Shift+Enter` 单元格(`# %%`) / `Ctrl+Shift+Enter` 选中 | 每章入口脚本 |
| ❌ **报错翻译** | 运行报错自动弹出卡片：现象→解释→原因→修复→相关章节；未命中可一键搜索 | 附录 C 反例索引表 |
| ✋ **动手试一试** | 点击"填入"把实验建议插入光标处，鼓励改参数（标记"已改"进度） | 每章动手试一试 |
| 🗺 **知识地图** | 侧边栏树视图：章节依赖 + 四态进度（已读/已跑/已改/已过关） | knowledge_map.md |
| ✅ **过关自测** | 从手册"过关自测·基础题"自动生成，自评过关点亮章节 | 每章过关自测 |
| 🩺 **环境自检** | 一键调度 `scripts/env_check.py`（TF≥2.16 / Keras 3 兼容性结论） | 附录 D |
| 🧩 **代码片段** | 11 个 TF 片段：Sequential/tape 循环/回调/tf.data/迁移学习/注意力… | 各章 |
| 🔬 **参数扫描** | 选中参数赋值 → 输入取值列表 → 自动多次运行 + 汇总最终指标 | 第 10 章 |

## 快速开始

1. 用 VS Code 打开 `F:\Tensorflow`（tf2-tutorial 工作区）；
2. 侧边栏点击 **TF 学习伴侣** 图标，从知识地图选择一章（如 `01 张量与自动求导`）；
3. 讲解面板打开后点「▶ 运行」，输出与新生成的图表直接出现在面板里；
4. 报错了？翻译卡片已经弹在输出上方——按修复建议改，`Ctrl+Enter` 再跑；
5. 学完做「过关自测」，全部答对点亮 🟢 已过关。

> 解释器自动探测：工作区 `.venv` → `F:\tf2-env` → PATH（可在设置 `tfTutor.pythonPath` 覆盖）。

## 开发

```powershell
cd extension
npm install          # 仅 TypeScript 与 @types（无运行时依赖）
npm run compile      # tsc -> out/
```

- 调试：VS Code 打开本目录按 F5 启动扩展开发宿主；
- 打包：`npx @vscode/vsce package` 生成 `.vsix`；
- 内容同步：手册更新后运行 `python scripts/generate_content.py`（单一事实来源，见 DESIGN.md）。

## 手动测试清单（对照设计要点 4.3）

- [ ] 环境不满足时（改 tfTutor.pythonPath 指向不存在的 python），运行给出友好提示
- [ ] 跑 `chapters/01` 触发 shape 反例 → 报错卡片命中"矩阵乘法 shape 不匹配"
- [ ] 参数扫描 4 个学习率 → 图表 Tab 出现 4 行指标汇总
- [ ] 进度保存后重启 VS Code → 知识地图状态恢复（globalState）
- [ ] 深色/浅色主题切换 → 面板跟随 VS Code 主题变量

## 已知边界（MVP）

- 参数扫描为文本替换版（不解析 AST），要求被扫描赋值在文件中唯一；
- 断点调试直接复用 VS Code 内置 Python 调试器（本插件不做封装）；
- 自测判分为基础题自评制，代码题判分（测试用例运行）规划于 V1.0。

## 故障排除

**Webview 报 `Could not register service worker: InvalidStateError`？**

这是编辑器（VS Code / 其分支）webview 服务 worker 的环境级问题，与插件内容无关。
按顺序尝试：

1. 命令面板运行 `Developer: Reload Window`，重试打开讲解面板；
2. **完全退出编辑器**，运行 `extension\scripts\fix-webview-cache.cmd` 清理
   Service Worker 缓存，再重开；
3. 即使 webview 加载失败，运行输出也已同步写入「输出」面板的
   **TF 学习伴侣** 通道（查看：命令面板 → 输出: 显示输出通道 → 选 TF 学习伴侣），
   学习功能不中断。
