# Changelog — TF 学习伴侣

## 0.1.5

修复单文件模式（未打开工作区文件夹）下"打开/运行"不可用：

- **项目根自动探测**：新增 `utils/repoRoot.ts`——从已打开文件的路径向上最多 6 层
  寻找仓库根（特征：`chapters/` 目录或手册文件），并缓存；工作区未打开时也能定位
  `F:\Tensorflow`。找到根后，运行/图表捕获/PYTHONPATH/入口脚本全部恢复可用。
- 打不开时的警告改为可操作指引（打开文件夹 / 先打开任一章节脚本）。
- 新增 `test/projectRoot.test.mjs`（5 项）：向上探测、起点即根、无根返回 undefined、
  正反斜杠混合路径。

## 0.1.7

功能完好性自检（自动化全覆盖）：

- 可测性重构：单元格提取与指标提取抽到 `utils/extract.ts`；执行环境变量集中为
  `buildExecutorEnv`；图表捕获 `newFiguresSince` 导出；Webview HTML 构建
  `LessonPanel.buildHtml(cspSource)` 静态化。
- 测试矩阵扩到 **53 项**（全部通过）：新增 单元格/指标提取（9）、图表捕获（2）、
  Webview HTML 结构与 CSP（3）、环境自检真实调度（4）、TF 导入真实噪音端到端（1）、
  进度流转与过关判定（2）。
- **迭代成果**：单元格语义统一为"包含所属 # %% 标记行"（与 VS Code Jupyter 一致），
  由测试驱动澄清并写入实现注释；内容生成脚本验证幂等（两次生成完全一致）。
- 内容层回归：`generate_content.py` 幂等 + 章节 JSON 校验，保证手册→插件同步可靠。
- **全章真实执行矩阵**：`run_all_chapters.py` 补登 11-16 章后，17 章端到端
  全部通过（主线 415s + 进阶 180s，零失败）——本轮最重要的完备性证据。

## 0.1.7

功能完好性自动自检（多轮迭代至零新发现）：

- 测试矩阵扩到 **61 项**，新增：属性化随机测试（noiseFilter 200 组随机切分、
  extractCellRange 300 组随机文档、extractMetrics 500 组随机键值、报错匹配 200 组
  随机拼接——可复现种子，失败可重放）；前端 JS 语法编译验证 + 迷你 Markdown 渲染器
  转译/XSS 转义；报错库对真实训练输出**零误报**；进度流转（自测全对→过关）断言；
  环境自检真实调度；TF 导入真实 stderr 过滤端到端。
- 可测性重构：单元格/指标提取 → `utils/extract.ts`；Markdown 渲染器 →
  `utils/miniMd.ts`（自包含纯函数，前端经 toString 注入）；执行环境集中
  `buildExecutorEnv`；HTML 构建 `LessonPanel.buildHtml(cspSource)` 静态化。
- **迭代成果**：属性测试抓到单元格语义真缺陷（光标在标记行上时会吞并上一格，
  空体格场景破坏"每格恰一标记"），已修正为"标记属于其下方的格"（与 VS Code
  Jupyter 一致）并固化不变式测试。

## 0.1.6

- **降级保障**：运行输出同步写入「输出」通道（TF 学习伴侣）——即使 webview 因
  编辑器环境问题加载失败（如 `Could not register service worker: InvalidStateError`），
  学习功能也不中断。
- 新增 `scripts/fix-webview-cache.cmd`：一键清理编辑器 Service Worker 缓存
  （该报错为编辑器环境层已知问题，与插件内容无关；README 增加故障排除指引）。

## 0.1.5

- 单文件模式（未打开工作区）支持：从打开文件向上自动探测项目根
  （`utils/repoRoot.ts`），运行/打开/图表/PYTHONPATH 全链路恢复；警告改为可操作指引。

## 0.1.4

修复「打开」按钮点击无反应：

- **根因**：Webview 消息路由的 switch 里漏掉了 `openEntry` 分支（"打开"按钮发出的
  消息无人处理），点击后既不报错也不打开文件。
- **修复**：补全路由——打开按钮现在会通过 `showTextDocument` 真正打开入口脚本；
  路径不存在时给出包含工作区根路径的警告。
- **回归测试**：新增 `test/messageRoute.test.mjs`（5 项）——stub `vscode` 模块加载
  编译产物，验证 13 条命令全部注册、openEntry 正常/缺失路径、WebToExt 协议每个
  消息类型都有路由、run/stop 路由存在。该测试在修复前**稳定复现**此 bug。

## 0.1.3

自动化测试与迭代更新（对照 设计要点 4.3）：

- 可测性重构：噪音过滤（`noiseFilter.ts`）与报错匹配（`errorParser.ts`）抽成
  纯函数，脱离 vscode 依赖即可单测；executor 改为调用纯模块。
- 测试集 24 项（Node 内置 test runner，零依赖）：噪音过滤 6 / 报错匹配 7（含
  16 条真实报错样本与误报/优先级/非法正则鲁棒性）/ 内容完整性 6（章节 JSON、
  依赖图、入口脚本、mdRange 定位、报错库）/ 清单一致性 4（命令唯一、键位、
  片段、**产物未编译即失败**、CHANGELOG 版本同步）/ Python 冒烟 2（真实跑
  第 0 章验证编码与耗时 + 故意报错验证翻译链路；无解释器环境自动跳过）。
- `npm test` 一键：tsc 编译 + 全部测试。
- **迭代成果**：测试抓到并修复 1 个真 bug——numpy 2.x 修改了矩阵乘法报错文案
  （`matmul: Input operand ... has a mismatch`），报错库已兼容新旧两种文案。
- CI：`.github/workflows/ci.yml` 新增 extension job（安装→编译→测试）。

## 0.1.2

- 输出面板按**到达顺序**交错渲染 stdout/stderr（此前 stderr 全部追加在末尾，
  导致 TF 导入日志出现在输出最后，顺序错位）。
- 自动隐藏 TensorFlow 初始化的无害日志（absl::InitializeLog / oneDNN /
  CPU 特性提示），首次出现时提示一次，结束时汇总隐藏条数；完整 stderr 仍保留
  供报错翻译匹配。

## 0.1.1

修复两个首用硬伤：

- **面板"▶ 运行"不再依赖编辑器焦点**：Webview 抢焦点导致 activeTextEditor 为空、
  误报"请先打开一个 .py 文件"。现在运行时优先取活动/可见的 .py 编辑器，
  都没有则**直接运行当前章节的入口脚本**（`python chapters/xx.py`），面板"打开"按钮
  也改为真正打开 .py 文件；运行后自动切到"输出"Tab。
- **输出乱码修复**：Windows 中文系统下 Python 管道输出默认 GBK，中文变
  "Ñ§Ï°ÂÊ"。执行环境强制 `PYTHONUTF8=1` + `PYTHONIOENCODING=utf-8`。

## 0.1.0 (MVP)

首批功能（对照 设计要点 5.1 MVP 清单 + 需求分析五个必做功能）：

- 命令注册：13 条命令（运行文件/单元格/选中、停止、讲解、输出、参数扫描、
  报错解释、自测、进度、环境配置、打开手册、重置进度）+ 3 组快捷键 + 右键菜单
- Webview 教学面板：讲解/输出/图表/自测/进度 五 Tab，深浅色主题自适应
- Python 执行层：spawn + 超时终止 + 停止按钮 + `# %%` 单元格 + figures/ 图表自动捕获
- 报错翻译：17 条高频模式（源自手册附录 C 反例索引表），未命中提供搜索按钮
- 环境自检与一键配置：调度 scripts/env_check.py（TF≥2.16 / Keras 3 结论）
- 进度追踪：已读/已跑/已改/已过关 + 自测过关，globalState 持久化
- 知识地图与进度树视图（章节依赖来自 knowledge_map.md）
- 11 个 TF 代码片段（对齐手册章节）
- 内容单一事实来源：scripts/generate_content.py 从手册生成 17 章结构化内容
