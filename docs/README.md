# docs/

教学文档目录（共 10 份，按使用顺序排列）：

| 文件 | 内容 |
| --- | --- |
| `knowledge_map.md` | **知识地图**：主线故事、全景路线、章节依赖图、三条学习路径、每章过关标准 |
| `learning_handbook_zh.md` | 中文学习手册：逐章对照代码讲解，含直觉解释、常见坑、调参指南 |
| `learning_handbook_zh.docx` | 学习手册 Word 版（可编辑、可批注） |
| `learning_handbook_zh.pdf` | 学习手册 PDF 版（约 95 页，Word 渲染，含目录与 40 幅插图） |
| `learning_handbook_en.md` | 英文学习手册 |
| `study_plan.md` | 14 天学习计划：每天 1-2 小时，含每日目标 + 动手任务 + 过关自测 |
| `study_plan_semester.md` | **16 周学期制课程表**：进阶篇 5 专题 + 调参调试 + 4 周大项目 + 自评分表 |
| `knowledge_checklist.md` | 知识点自检清单：0-9 章共 60+ 个知识点，学完打勾 |
| `glossary_zh.md` | 术语表：40+ 术语，中英对照 + 直觉类比 |
| `FAQ.md` | 调试手册：10 章 / 54 个常见问题 / 18 条报错速查 |
| `raspberry_pi_deployment_zh.md` | 树莓派部署指南：9 章完整实战，从刷系统到实时推理 |

章节脚本 (`chapters/*.py`) 是文档的"最小可运行"版本。文档则负责：
- 把每个 API 用法放到概念地图中；
- 指出常见陷阱 (shape / dtype / 设备不一致等)；
- 给出"先看这一段就够"的路径。

> 如果章节脚本跑不通，先到 `FAQ.md` 找答案，再翻学习手册定位章节。
