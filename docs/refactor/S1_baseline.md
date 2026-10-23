# S1：可复现基线冻结

日期：2026-10-19（Asia/Shanghai）。用户已批准执行 S1。范围为 `Codex_THIS_Refactor_Plan/02_S1_Freeze_Baseline.md`；未进入 S2–S6，未合并、移动或删除原文件。

## 基线身份与保护方式

主项目 `/home/yg/THIS-Lasso` 没有 `.git`，根 Git status/HEAD 返回128；因此主项目 commit 为 N/A。采用本轮可审计的逐文件哈希清单和审计目录内源码快照冻结，不擅自初始化 Git、创建分支/标签或提交文件。

基线标识：`S1-20261019-f14a15e52881`。

| 产物 | 范围/身份 |
| --- | --- |
| [S1_baseline_manifest.csv](S1_baseline_manifest.csv) | 31,479个原有文件或符号链接，包含源代码、正式/历史数据结果、Git元数据、环境及4份S0文档 |
| [S1_R5_formal_manifest.csv](S1_R5_formal_manifest.csv) | 正式 `results/r5/e1_full_independent/` 的4,113个文件，含原manifest自身 |
| [S1_upstream_manifest.csv](S1_upstream_manifest.csv) | 两原论文118个跟踪文件的SHA256与commit |
| [S1_source_snapshot.tar.gz](S1_source_snapshot.tar.gz) | 222个源码/测试/配置/包锁/报告/原论文跟踪内容等文件，压缩后2,822,856字节；原文件留在原处 |
| [S1_source_snapshot_index.csv](S1_source_snapshot_index.csv) | 快照成员原路径、大小、SHA256；已逐成员读回比对 |
| [S1_environment.json](S1_environment.json) | 环境版本、原仓库HEAD、真实Git退出码、既有清单核对结果及快照身份 |
| [S1_golden_index.csv](S1_golden_index.csv) | 111个现有文件的角色、固定路径、大小和哈希 |
| [S1_golden_contract.json](S1_golden_contract.json) | paired trial、样本位置、节点诊断、标准化/索引约定、既有指标与选择锁 |

全清单SHA256：`f14a15e52881c8efbaa4bb39608f033d8020d5d1613e9a227ce5ea168c51ce5a`。

源码快照SHA256：`90b215c3d678cde6bc160c2c4e3a43c5cd6029fb02cfcc555bdfd3e0daf36672`。

清单字段为 `path,type,size,sha256,scope`；路径相对于项目根目录。普通文件直接哈希内容；符号链接哈希原链接文本，不遍历链接目标。S1新增产物仅位于 `docs/refactor/S1_*`，不计入原文件清单以避免自引用；S0文档仍受本轮保护。不是 chmod 只读文件系统，权限未变。

快照包括实际算法和原论文内容；运行环境、数据和结果仅按完整清单固定并原地保留，没有复制成第二套全部数据。快照在同一工作目录，**不是独立介质的灾难恢复备份**。后续获准重构时可用源码快照恢复源文件，正式结果继续以原目录及清单核对，不得覆盖历史产物。

## 原论文与既有证据核对

| 原目录 | HEAD | 只读核对 |
| --- | --- | --- |
| `upstream/THIS/` | `865efc43fe8ec21966659b482ca5756396974425` | Git status空；9个跟踪文件与R0锁一致 |
| `upstream/Su-AdaptiveLASSO/` | `fa0d662f68b7b41a1c6c3401af9504929126dafd` | Git status空；109个跟踪文件与R0锁一致 |

未fetch、改动、移动上游目录，也未替换其Git元数据。自有Adaptive solver仍独立于Su原振荡系统实现，未借冻结改变算法来源声明。

既有manifest逐项读取核对：R1=54、R3=58、R4=109、R5阶段级=5921、R5正式目录manifest=5921，均匹配。两份R5清单包含重复保护范围，不是11842个不同文件。S1新清单补充覆盖两份旧manifest自身和旧清单有意排除的审计日志，旧清单字节不变。

正式R5保留4,113文件、211,351,447字节；所有30个seed、480条trial及每模型选择记录、候选警告、初值独立性、raw/standardized系数、边/指标、bootstrap索引、原始图、测试/运行日志完整保留。

中止批次 `results/r5/e1_full/`、smoke及独立性修正证据受全清单保护。没有删掉中止数据，也没有将其与正式统计混合。`selection_locked.toml`、`protocol.toml`、`trajectory_independence.toml`、旧manifest、正式CSV/TOML和原图均未重写。

## 运行环境及重建依据

- Python：项目内Conda prefix `.venv-r0`，Python3.11.17；NumPy1.26.4、SciPy1.11.4、scikit-learn1.6.1、matplotlib3.9.4、pandas2.2.3、statsmodels0.14.0、skglm0.5。`requirements.lock.txt` 的28项全部与安装元数据一致。
- Julia：1.9.4，原二进制 `/home/yg/.local/opt/julia-1.9.4/bin/julia`；项目 `environment/r0/julia/Project.toml` 与 `Manifest.toml` 保持原样。
- 重建依据：原 `environment/r0/README.md`、`conda-linux-64.lock.txt`、`requirements.lock.txt`、Julia Project/Manifest及上游锁，均在全清单与源码快照中。包锁和原部署说明是重建来源；本轮没有另建环境、安装依赖或下载包，也未声称在第二台机器验证过重建。
- 当前默认shell解释器和Julia PATH不一定指向项目环境。测试显式使用项目Python和已记录的Julia二进制，未修改激活脚本/IDE设置。

为避免写回原depot，Julia测试使用 `--startup-file=no --history-file=no --compiled-modules=no`，新depot入口放在 `docs/refactor/S1_work/julia_depot`，旧depot仅作为查找后备。Python禁写pycache，BLAS/Julia均单线程；临时文件、可选绘图配置目录都定位在S1审计目录。没有使用会写历史results的旧运行logger或audit入口。

## Golden与轻量重放

选择S0已建议的seed22261035，同一完整数据切片的0%和5%响应噪声。两档共享相同X/theta/d、真值和轨迹分割，保留300/90/90样本与完整165特征，避免缩短样本改变BIC适用条件。Golden建立只索引/摘录原产物，没有拟合补字段。

Ridge初值、权重、alpha及路径在已有诊断中完整。M0没有LASSO alpha/Ridge权重、M1没有Adaptive初值，属于不适用字段；零噪声M1/M2在BIC guard后停止，没有选中系数/alpha，保留为NA，不能补造输出。无缺失的应有字段。

随后仅做已有样本的最小兼容重放：R5同seed两档M0/M1/M2/M3，另保留R4 seed22261017的M0s/M1v控制。Python拟合在内存中，Julia直接读固定数据检查原生THIS；新输出仅为审计JSON/日志，不写正式结果。不运行R4 B、E2/E3或30-seed全量。

全部重放的有效系数与原文件逐位一致；比较到的alpha、Ridge初值、权重差值均为0。Taylor矩阵、特征顺序、均值/尺度、结构解码、支持集、边score与TP/FP/FN/TN匹配。M1/M2零噪声每个节点仍为原理由NA，未fallback。详见 [S1_golden.md](S1_golden.md) 和 [S1_test_results.md](S1_test_results.md)。

## 正式结论哨兵

只读汇总已保存的trial F1，核对保存的summary和任务包四位小数值；不根据结果选择参数。

| 响应噪声标准差比例 | M0 | M1 | M2 | M3 |
| --- | --- | --- | --- | --- |
| 5% | 0.6700 | 0.8363 | 0.8769 | 0.4960 |
| 10% | 0.6700 | 0.8363 | 0.8224 | 0.4234 |
| 20% | 0.6600 | 0.7995 | 0.6681 | 0.3489 |

记录数仍为420 OK、60零噪声BIC NA、0 FAILED；候选警告979、选中警告0。没有重算30-seed拟合或修改bootstrap/置信区间；原始负面结果、弱交互漏检和M3假阳性结论不变。明细在 [S1_sentinel_checks.json](S1_sentinel_checks.json)。

## 收尾与S2闸门

收尾原文件完整性核查已通过：31,479个原文件/链接逐项比较无变化，无原文件缺失或审计范围外新增；其中正式R5仍为4,113文件。结果见 [S1_integrity_check.json](S1_integrity_check.json) 与 [S1_integrity.log](S1_integrity.log)。审计驱动程序为 [S1_audit.py](S1_audit.py)，Julia兼容检查为 [S1_julia_compatibility.jl](S1_julia_compatibility.jl)；它们只是本轮审计产物，不是项目算法替代实现。

新产物哈希在 `S1_audit_artifacts.csv`（不包含自身），作为本轮交付完整性索引。日志使用独占创建，既有S1目录不能直接重复freeze/golden覆盖；后续复查须采用新的、获准的审计输出位置或新日志标签。

S1完成，已停止，等待人工验收。下一阶段如获批，应先阅读S2要求，从确实等价的I/O、普通LASSO/选参辅助职责着手，保留旧API以及M0s/M1v对照，不同时搬迁历史报告或变更正式产物。S2–S6均未执行。
