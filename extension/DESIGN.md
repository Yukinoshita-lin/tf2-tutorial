# TF 学习伴侣 · 设计决策记录（DESIGN.md）

> 需求来源：`需求分析.md`（痛点与 MVP 边界）+ `设计要点.md`（架构与实现清单）。
> 本文档记录把"纸面设计"落到"本仓库可用插件"时的**具体决策与取舍**。

## 1. 范围裁剪（对照 5.1 MVP 路线）

| 设计要点功能 | 本版状态 | 决策说明 |
| --- | --- | --- |
| 命令注册（12 条） | ✅ 全量实现 | 13 条命令 + 3 组键位 + 编辑器右键菜单 |
| Webview 讲解面板 | ✅ 实现 | 五 Tab（讲解/输出/图表/自测/进度） |
| Python 执行 + 输出 | ✅ 实现 | spawn 方式；`# %%` 单元格；图表自动捕获 |
| 报错翻译 | ✅ 实现 | 17 条正则模式，源自手册附录 C 反例索引表 |
| 环境自检 + 一键配置 | ✅ 实现 | 调度项目自带的 `scripts/env_check.py`（单一事实来源） |
| 进度追踪（简单版） | ✅ 实现 | 已读/已跑/已改/已过关 + 自测过关 |
| 知识地图（P2） | ✅ 提前实现 | 数据来自手册依赖图，成本低收益高 |
| 自测判分（P2） | ✅ 基础版 | 从手册"过关自测·基础题（带【答】）"自动生成 3 题/章 |
| 代码片段（P2） | ✅ 声明式 | 11 个片段，全部对齐手册章节 |
| 参数扫描（P1） | ✅ MVP 版 | 文本替换 + 逐值运行 + 指标正则提取（AST 版入 V2） |
| 断点调试（P1） | ⏭ 延后 | 直接复用 VS Code 内置 debugpy 调试（launch.json 后续版本提供预设） |
| 多语言 / 云同步 / AI 答疑（V2.0） | ⏭ 延后 | 按 5.3 路线 |

## 2. 关键技术决策

### 2.1 零运行时依赖 + vanilla Webview
设计要点建议 Webview 用 React + `marked` + Plotly。**MVP 决策：不用构建链**——
前端为 vanilla JS + 内置迷你 Markdown 渲染器（覆盖手册使用的语法子集：标题/粗体/
代码块/列表/引用/表格），图表直接展示章节脚本产出的 PNG（本仓库每章本来就有
`figures/` 产物）。收益：`tsc` 一次编译即完成构建、无 bundle 体积、CSP 更容易收紧。
面板复杂化后再引入 React + esbuild（V2 路线）。

### 2.2 内容单一事实来源
设计要点风险表明确要求"用同一份源文件生成手册和插件内容"。实现：
`extension/scripts/generate_content.py` 解析 `docs/learning_handbook_zh.md` 的
五板块教学结构 → `content/chapters/{id}.json`（目标/概念/实验/坑/反例/自测/阅读）+
`content/chapters.json`（依赖图）。**手册更新后重跑脚本即可同步插件**。
解析器按章输出统计（quiz/反例/概念/实验/坑 计数）便于核对完整性。
此外 Webview 优先渲染手册原文的章节切片（`mdRange`），结构化字段只做摘要增强。

### 2.3 报错翻译库的来源
报错模式不是凭空写的：直接取材手册**附录 C 反例索引表（48 条）**与反例教室的
真实报错文本，浓缩为 17 条高频正则模式。每条含 `现象匹配 → 解释 → 原因 → 修复 →
相关章节`，与需求分析功能 4 的卡片格式一一对应。未命中的报错提供"搜索此报错"
按钮（设计要点 2.4 第 4 条）。

### 2.4 Python 执行
按设计要点 3.4 方式一（`child_process.spawn`）实现：cwd=工作区根、
`PYTHONPATH` 追加 `<root>/src`（复用 tf2tutorial 包）、`TF_CPP_MIN_LOG_LEVEL=3` 降噪、
超时自动终止 + 面板停止按钮（SIGTERM→SIGKILL）。单元格 = `# %%` 标记，
自动附加项目路径 prologue，保证"分步教学"不踩 import 坑。
图表捕获：运行前后对比 `figures/` 目录 mtime，新图自动推送 Webview 展示。

### 2.5 环境检测的单一事实来源
自检逻辑不在扩展里重写——直接调度项目自带的 `scripts/env_check.py`
（TF≥2.16 / Keras 3 / 关键 API / 兼容性结论），扩展负责执行、解析与把结论
转成用户动作（一键建 venv、装依赖、给修复建议）。手册附录 D 与脚本同源。

### 2.6 进度模型
`已读 / 已跑 / 已改 / 已过关` 四状态（需求分析功能 5），存 `globalState`
（跨窗口持久化）。"已过关"规则：本章自测题全部自评正确。知识地图树视图按
四状态着色（🟢 过关 / 🔵 已跑 / 🟡 已读 / ⚪ 未开始），点击即打开讲解。

### 2.7 安全
CSP：`default-src 'none'; script-src 'nonce-…'; img-src vscodeResource data:`；
所有传给 Webview 的文本先 HTML 转义；不执行 Webview 传来的任何代码；
本地资源经 `cspSource` 白名单。

## 3. 与手册/仓库的整合点（需求分析·五）

| 需求 | 实现 |
| --- | --- |
| 手册章节 → 插件讲解 | generate_content.py 生成 chapters/*.json + mdRange 渲染手册原文 |
| 动手试一试 → 一键实验 | 讲解 Tab 的"填入"按钮：把建议插入光标处并标记"已改" |
| 常见坑 → 报错时弹出 | 报错卡片内含 relatedChapter（对应反例编号） |
| 过关自测 → 自动判分 | 自测 Tab：从手册基础题生成，自评过关写入进度 |
| 知识地图 | 树视图 + 依赖关系（deps 来自 knowledge_map.md） |
| 打开手册 | openManual 命令直接打开 docs/learning_handbook_zh.md/PDF |

## 4. 目录结构

```
extension/
├── package.json            # 13 命令 / 键位 / 视图容器 / 片段 / 6 项配置
├── tsconfig.json
├── src/
│   ├── extension.ts        # 激活入口：注册视图与命令
│   ├── types.ts            # 章节/报错/进度/消息协议
│   ├── commands/index.ts   # 13 个命令 + 消息路由 + 进度标记
│   ├── panels/lessonPanel.ts   # 五 Tab Webview（CSP/迷你MD渲染/图表/自测）
│   ├── python/executor.ts  # spawn 执行 + 超时停止 + 图表捕获 + 报错匹配
│   ├── python/env.ts       # 环境自检调度 + 一键配置 + 轻量模式
│   ├── data/content.ts     # 章节/报错库/手册切片/进度存储
│   ├── tree/knowledgeMap.ts    # 知识地图 + 进度树视图
│   └── utils/config.ts     # 设置读取 + 日志通道
├── content/
│   ├── chapters.json       # 章节索引 + 依赖（generate_content.py 生成）
│   ├── chapters/*.json     # 17 章结构化内容（同上）
│   └── errors/errorPatterns.json  # 17 条报错翻译模式
├── snippets/tf-snippets.json   # 11 个 TF 代码片段
└── scripts/generate_content.py # 手册 → 插件内容（单一事实来源）
```

## 5. 自动化测试与迭代更新

`npm test` = `tsc 编译 + node --test test/*.test.mjs`（Node 内置 runner，零测试依赖）：

| 测试文件 | 覆盖 | 来源 |
| --- | --- | --- |
| `test/noise.test.mjs`（6） | 噪音过滤：隐藏/提示一次/半行跨批/flush/混合/普通输出 | 0.1.2 乱码修复的回归 |
| `test/errorMatch.test.mjs`（7） | 16 条真实报错样本逐条命中 + 误报/优先级/非法正则 | 附录 C 反例索引表 |
| `test/content.test.mjs`（6） | 章节 JSON 完整性、依赖图可信、入口脚本存在、mdRange 定位、报错库结构 | 单一事实来源校验 |
| `test/manifest.test.mjs`（4） | 命令唯一/键位/图标/片段/**产物未编译即失败**/CHANGELOG 版本同步 | 发布前检查自动化 |
| `test/pythonSmoke.test.mjs`（2） | 真实跑第 0 章（编码/退出码/耗时）+ 故意报错验证翻译链路 | executor 端到端；无解释器自动跳过 |
| `test/messageRoute.test.mjs`（5） | stub vscode 加载编译产物：13 命令注册、**openEntry 路由回归**（0.1.4 修复的 bug）、WebToExt 协议全覆盖 | 消息路由回归 |
| `test/projectRoot.test.mjs`（5） | 仓库根向上探测（单文件模式）：从章节脚本/子目录找到根、无根 undefined、混合路径 | 0.1.5 单文件模式修复 |
| `test/extract.test.mjs`（9） | 单元格区间（含标记行语义，与 VS Code Jupyter 一致）+ 指标提取（同名取末值） | 功能自检新增 |
| `test/figures.test.mjs`（2） | 图表捕获：新图捕获/旧图忽略/非图片忽略/目录缺失安全 | 功能自检新增 |
| `test/html.test.mjs`（3） | Webview HTML：CSP+nonce、五 Tab、无闭合标签破坏 | 功能自检新增 |
| `test/envCheck.test.mjs`（4） | 环境自检真实调度、解释器探测、UTF-8 环境变量 | 功能自检新增 |
| `test/tfLiveNoise.test.mjs`（1） | 真机导入 TF 的 stderr 走完整过滤管线（端到端） | 功能自检新增 |

**迭代闭环**：跑测试 → 修失败 → 重跑。两轮迭代各抓到 1 个真 bug——
① numpy 2.x 矩阵乘法报错文案变更导致翻译库失配（已兼容新旧文案）；
② "打开"按钮的 openEntry 消息路由 case 漏失（已修复 + 5 项回归测试防复发）。
测试总数 53 项，全部通过。
CI（`.github/workflows/ci.yml`）已加 extension job：每次 push 自动 安装→编译→测试；
内容生成脚本验证幂等（两次生成 diff 为空），手册→插件同步可信。
Python 冒烟在无 TF 环境下自动跳过，不阻塞 CI。

## 6. 开发与发布

```powershell
cd extension
npm install        # 仅 devDependencies：typescript + @types/vscode + @types/node
npm run compile    # tsc -> out/extension.js
```

- **本地试用**：VS Code 打开 `extension/` 目录 → F5（扩展开发宿主）；
  或 `npx @vscode/vsce package` 生成 .vsix → `code --install-extension *.vsix`。
- **测试**（设计要点 4.3）：`tsc --noEmit` 静态检查已过；手动用例清单见 README。
- **发布**：vsce package → Marketplace（需发布者账号），版本号与 CHANGELOG.md 同步。
