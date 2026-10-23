# S4｜删除候选清单审计（不执行任何删除；需 S3 通过后单独执行）

**本阶段严格为 dry-run。即使文件看起来完全无用，也不得执行删除或覆盖。**

请对文件提出四类建议：
- `KEEP`: 正式配置、核心代码、两篇原始论文源码、测试、现役报告、R4 未完成对照、R5 正式审计/数值证据。
- `ARCHIVE`: 历史报告、smoke/aborted 日志、具有研究解释价值但非主线的数据。
- `DELETE_CANDIDATE`: 可由已冻结源完全再生成的缓存、明确重复且 hash 一致的副本、无引用的过时临时产物。
- `UNKNOWN`: 找不到确定拥有者、无法确认引用或无法恢复的文件，一律不删。

生成 `docs/refactor/S4_delete_allowlist_PROPOSED.csv`，至少包含：`path,size,sha256,tracked,refs_found,duplicate_of_or_regeneration_command,backup_path,risk,reason,approval_status`。

逐文件递归检索静态依赖：Python import/路径字面量、Julia include、TOML、shell、Makefile、README、图表及报告 Markdown 链接；标记动态加载造成的检索盲点。若无法证明可恢复性，标记 UNKNOWN。

特别排除：
- `protocol.toml`、`selection_locked.toml`、`trajectory_independence.toml`、`artifact_manifest.toml` 和所有正式 trial 原始数据/指标/警告日志。
- 两篇论文原始源代码、环境锁、正式训练/验证/测试划分、关键 RNG seed 信息。
- 0% BIC NA 的日志、初始化问题的弱边 CSV、aborted 独立性修正证据。
- `M0s`、`M1v`、R4 B、E2/E3 后续研究配置和说明。

阶段输出为清单、预计节省空间、风险报告、明确的需要用户逐项审批说明。**不执行 S5、不生成可批量一键危险删除脚本。**
