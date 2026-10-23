# S0｜只读盘点（提交给 Codex 的第一条独立任务）

**工作模式：只读，不允许移动/删除/修改任何已有文件；唯一允许新增的内容为 `docs/refactor/S0_*.md` 和 `docs/refactor/S0_*.csv`（必要时先询问是否允许新建目录）。本阶段执行完即停止。**

请先探测本仓库真实目录与 Git 状态：`pwd`、`git status --short`、`git rev-parse --show-toplevel`、必要的有限深度目录树（忽略大缓存，不忽略正式 results/ 与 docs/）。若工作区 dirty、存在未跟踪数据或上游代码混在主目录，必须如实报告，不得自动清理。

1. 找出两篇论文各自原始源码的真实路径，识别是否是 git submodule / vendor / copy，记录 commit/SHA 与只读建议。
2. 搜索 R0–R5 模块与调用关系：数据生成 → Julia Taylor `get_θd` → 标准化 → Ridge 权重 → LASSO / Adaptive / STLSQ → BIC/验证选参 → 原尺度还原 → 结构解码 → 指标/绘图。
3. 找到 M0、M0s、M1、M1v、M2、M3 的实际实现、调用入口、配置及关键依赖。检查跨 Julia/Python 接口与常量列处理。
4. 对报告、测试、图表、缓存、临时实验、aborted 历史做逐文件清单；标记是否被配置、文档、脚本、测试、正式结果引用。
5. 找出下列正式证据文件的真实路径并标记是否存在：`r5_e1_full_independent.toml`, `r4_dry_run.toml`, `R5_report.md`, `protocol.toml`, `selection_locked.toml`, `trajectory_independence.toml`, `artifact_manifest.toml`, `trials.csv`, `summary_30.csv`, `paired_summary_30.csv`, `weak_edge_diagnostics.csv`, `candidate_warnings.csv`。不能因不在预期路径就认定不存在。
6. 识别环境与潜在风险：Python/Julia 版本、包锁、路径硬编码、生成目录大小、可复用测试、是否包含绝对路径或秘密（不要输出秘密）。

交付：
- `docs/refactor/S0_inventory.csv`: `path,type,tracked,size,owner_or_purpose,referenced_by,criticality,proposed_action,reason`。
- `docs/refactor/S0_dependency_map.md`: 真实入口、关键调用图和上游只读界限。
- `docs/refactor/S0_open_questions.md`: 缺失文件、重名文件、无法确认引用与潜在风险。
- 一个精简摘要：保留、建议合并、建议归档、待核查可删的数量和空间成本。

**禁止生成删除命令；禁止执行 S1–S6。** 完成后请询问用户是否批准 S1 冻结基线。
