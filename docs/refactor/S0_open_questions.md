# S0：风险、缺失检查与下一阶段待确认项

盘点日期：2026-10-19，Asia/Shanghai。仅只读检查及新增 S0 审计文档；以下方案不等于执行许可。S0 完成后停止，S1–S6 未执行。

## 1. Git 与环境：必须先处理的事实

1. 根目录 `git status --short`、`git rev-parse --show-toplevel`、`git submodule status` 均退出128：根目录不是 Git 仓库。不能将主项目称为 clean，也没有可供比较的主项目 HEAD；清单的 tracked 列使用 N/A，而不是假称“已跟踪”。没有擅自 git init、add、commit、改分支或标签。
2. 两个上游仓库分别独立，status 含全部未跟踪文件检查的输出为空，退出0；与 R0 锁118个文件逐项匹配。它们的 `.git` 与数据均保护，不因根 `.gitignore` 忽略 upstream 就视为可删。
3. `.venv-r0` 是 Conda 环境目录，有 `conda-meta/history`，无 `pyvenv.cfg`；名称不能作为环境类型依据。Python 3.11.17、NumPy1.26.4、SciPy1.11.4、scikit-learn1.6.1、matplotlib3.9.4；requirements.lock.txt 的28个包版本与实际安装元数据全部一致。未安装、升级、激活修改或导入运行算法。
4. 包装器返回 Julia1.9.4，Manifest 为 Julia1.9.4、format2.0、54个 deps 条目。仅 `--version` 检查，未执行 Pkg.instantiate/precompile 或 Julia 数值测试。默认 shell 的 python 指向 `/home/yg/miniconda3/bin/python`，PATH 未找到 julia；重现应使用项目解释器及包装器。
5. `environment/r0/activate.sh`、`julia.sh`、`.vscode/settings.json` 有用户目录/安装位置硬编码；R5数据 metadata 中 `source_data_dir` 也有绝对路径。扫描发现331个含 `/home/yg` 的项目文本，很多是应保留的历史日志。后续只在新的运行入口支持路径配置，不回写旧日志/metadata。
6. 发现1408个环境符号链接，均可解析，目标均在项目内。本轮不跟随链接重复统计内容，也不修改链接。环境、depot 内的包/制品不等于缓存；删除当前环境不是本计划的默认操作。

S1 的保护方式待用户批准：在无 Git 状态下，可以先采用只读清单及明确范围的外部备份方案；如希望初始化主项目 Git，必须另行确认跟踪范围、分支和提交行为，不能把结果、环境或潜在隐私内容直接整体提交。S0 未执行任何一种 S1 冻结方案。

## 2. 正式证据存在性与保护清单

| 要求文件 | 实际路径 | 本轮只读结果 |
| --- | --- | --- |
| 正式配置 | `configs/r5_e1_full_independent.toml` | 存在，正式锁定副本在正式目录 |
| R4基础配置 | `configs/r4_dry_run.toml` | 存在，正式 R5 仍引用 |
| 当前报告 | `reports/R5_report.md` | 存在，30独立seed报告 |
| 协议 | `results/r5/e1_full_independent/protocol.toml` | 存在 |
| 模型选择记录 | `results/r5/e1_full_independent/selection_locked.toml` | 存在，480项；先锁后评分的既有记录 |
| 独立性证据 | `results/r5/e1_full_independent/trajectory_independence.toml` | 存在；4800个不同初值，中止批次189个 |
| 正式manifest | `results/r5/e1_full_independent/artifact_manifest.toml` | 存在，与阶段级manifest相同内容 |
| 阶段级manifest | `results/r5/artifact_manifest.toml` | 存在，full_verified=true；不得被smoke审计覆盖 |
| 原始指标 | `results/r5/e1_full_independent/trials.csv` | 存在，480行；420 OK、60 NA、0 FAILED |
| 汇总 | `results/r5/e1_full_independent/summary_30.csv` | 存在 |
| 配对统计 | `results/r5/e1_full_independent/paired_summary_30.csv` | 存在 |
| 弱边诊断 | `results/r5/e1_full_independent/weak_edge_diagnostics.csv` | 存在，不能只留报告里的单例 |
| 候选警告 | `results/r5/e1_full_independent/candidate_warnings.csv` | 存在；既有审计记录979个未选中警告、0个选中警告 |
| 正式测试日志 | `results/r5/logs/full_test_independent.log` | 存在；既有23项测试通过、EXIT_CODE=0 |
| 正式运行/审计日志 | `results/r5/logs/full_run_independent.log`、`full_audit.log` | 存在，EXIT_CODE=0 |

同时保留 `bootstrap_seed_indices.csv`、逐seed配对差值、28噪声seed敏感性分析、强度分组记录、420个有效模型的原尺度/标准化系数和诊断、60个NA理由、每seed worker日志、原始PNG/PDF。既有清单有意排除自引用manifest及当前审计日志，不能把“清单未列自身”误报为缺失。

只读重算现有 manifest 所列哈希：R1 54项、R3 58项、R4 109项、R5阶段级5921项、R5正式目录清单5921项，全部匹配（后两份清单保护范围重复，不是11842个不同文件）。这是完整性核查，未重跑结构评分、bootstrap、测试或30-seed实验。

没有发现任务包要求的必要配置、源码或正式证据缺失。未创建替代数据、未将中止批次当作基线。文件名相同不表示实验身份相同：三套R5的 protocol/selection lock/config副本及多份弱边诊断分别保留。

## 3. 高风险依赖

| 风险 | 实际依赖 | 后续约束 |
| --- | --- | --- |
| 把“audit”视为只读 | `r3_audit.py`、`r4_audit.py`、`r5_audit.py`、`r5_batch_audit.py` 会写清单/指标/诊断等；独立性脚本也写正式TOML | 本轮未调用；S1只能用经确认的只读检查或隔离输出，不能原路径重跑 |
| smoke覆盖正式阶段状态 | `r5_audit.py` 写根 `results/r5/artifact_manifest.toml`，字段声明仅smoke、full未执行 | 不得为“验证通过”直接重跑旧smoke审计 |
| 源码哈希和文档哈希跨阶段 | R3/R4/R5 preflight、protocol及manifest锁定旧路径/旧字节 | S2改源码后旧审计可能按设计拒绝；先冻结代码来源和golden，再设计新的比较层，禁止改历史manifest迁就新代码 |
| 把旧阶段公共代码删掉 | R5批量、worker、分析、审计直接导入 `r4_experiment` / `r4_methods` / `r4_metrics`；Julia依赖 `r1_common.jl` | 保留旧API wrapper，确认所有import/include/子进程调用后逐步提取 |
| M0s/M1v被当作遗留 | R4六方法配置、prepare与Python派发、R4比较设计 | 它们是控制预处理和选参影响的必要对照，仍有后续用途 |
| 双重标准化或截距变化 | 外部训练Z + solver内部 `standardize=false`；M0/STLSQ仍惩罚常量 | 不用统一接口偷偷改变目标函数、常量处理或STLSQ输入 |
| 私有API强耦合 | `r4_methods.py` 导入 Adaptive solver `_prepare`、`_bic_variance` | 可在冻结后提取稳定API，但不能直接替换BIC或方差估计 |
| 桥接路径与批量路径不一致 | R3走request/response；R4/R5直接fit_node后落CSV并Julia解码 | 两条路径都要golden覆盖，不能只测桥接就宣称批量等价 |
| 数据RNG独立性被回退 | 原生成器 `seed+trajectory_id`；R5专有stride命名空间 | 共用prepare时保留两种历史语义，不把R4原数据改写成新R5数据 |
| aborted数据仍被读取 | `r5_check_independent_trajectories.py`读取 `results/r5/e1_full/clean/seed_*/data/initial_conditions.csv` | 不是普通临时目录；逻辑归档优先，不能删除或移动后声称复核链完整 |
| 再生成覆盖正式证据 | 分析脚本写CSV/PNG/PDF；运行包装器日志有拒绝覆盖，但并非所有下游脚本均拒绝写入 | 不运行既有正式输出目录，后续验证结果仅进入明确新目录 |
| 统计口径漂移 | R1历史指标与R4候选全集/AP不同；seed簇、NA分母、弱边分层有特定定义 | 不强行合并不同口径；不更改“AP”为另一种AUPRC或将NA填0 |
| 科学结论被重构改写 | M2在20%噪声弱三体Recall低、F1低于M1；M3有大量FP | 这些是冻结事实，不是本轮要修复的bug；不调rho/gamma/eps/grid/阈值 |

## 4. 重复、历史与可再生成内容

- 排除第三方环境和Git元数据后，发现259组字节完全相同的文件。大量是固定真值/indices、配置锁定副本、历史快照、重复保存的输入，属于溯源证据。内容相同不能证明可删。
- `configs/r4_dry_run.toml` 与R4/R5各套 `r4_base_locked.toml` 相同；三套R5配置各有自身 config_locked 副本。保留这些历史锁，不做去重链接替换。
- 中止批次 `aborted_source/r5_batch_analyse.py`、`r5_batch_worker.py` 与当前同名脚本字节相同；其他批次源码快照也按原实验身份保留，不能跨批替换。
- 报告及旧logger占空间很小；真正的大项是环境、Taylor矩阵及整批数据。正式R5 theta、标准化theta、噪声、诊断、图虽然可通过程序再生成，但重现实验证据价值高，均标为保留。
- 唯一“待核查可删”类别为环境中的7525个Python `.pyc` 与44个Julia compiled文件，共169.74 MiB。仅候选、不代表已获删除批准；须验证可再生成、无上游/正式证据依赖、逐项审批。本轮没有生成删除命令。
- `.venv-r0` 的包源码、共享库、Conda metadata；Julia packages/artifacts/registries 均保留，不纳入此候选。

## 5. 链接与秘密扫描范围

静态文本扫描发现56处相对链接在文件现位置无法解析：54处在两套不可变 `history/` 快照中，快照原文按 `reports/` 或 `docs/` 所处位置编写，复制后相对路径失效；另2处在 `reports/r0_logs/setup_history.log` 引用已被历史合并的旧环境说明/日志。没有发现本轮任务要求的正式证据缺失。

不修改这些历史字节，后续用外部导航记录“原位置→当前证据路径”。当前有效主报告与原有活动docs的链接在本次扫描范围内未发现缺失。静态结果不覆盖运行期拼接、第三方文档自定义链接语法或网络链接可用性。

扫描项目自有代码、配置、文档、日志及上游源码文本/原仓库remote配置，未检出指定模式的私钥或硬编码凭据赋值；只记录扫描结论，不输出潜在秘密值。未扫描第三方环境包全文及外部用户凭据目录，因此不能据此保证整个用户目录或全部二进制均无秘密。任何后续备份/初始化Git都应遵循最小跟踪范围。

## 6. S1建议：需单独批准，本轮未执行

1. 先由用户确认无主项目Git情况下的保护方案；如新增Git分支/提交，单独确认具体行为。建议先冻结源文件和正式证据索引，避免把环境和结果整体纳入版本库。
2. 在S1允许的审计目录新增正式文件逐项SHA256清单，记录两上游commit/哈希和包锁；历史manifest原样保留。S0的目录摘要仅用于证明本轮只读，不替代S1基线交付。
3. 直接索引已有golden，优先 `seed_22261035/noise_050` 的M0/M1/M2/M3，以及同seed的 `noise_000`：后者M1/M2应保留NA。前者M2诊断已有 initial_coef、weights、alpha、feature_means/scales、alpha_path/bic_path、KKT，无需先拟合生成新字段。
4. 为golden关联 clean数据、theta/d、外部均值/尺度、Y_noisy、split、status、原尺度系数、边分数、metrics、模型选择锁；记录已有字段可用性。若要提取切片或补运行缺失字段，先按S1边界确认输出位置与命令，绝不写正式目录。
5. 现有Python静态测试计数：R2=30、R3=15、R4=17、R5noise=14、R5batch=9，共85个测试方法；另有3个Julia测试文件。S1先审查测试副作用，再使用项目解释器、单线程和禁止pycache，将新测试日志放批准的审计目录，不能调用写历史results的旧logger/audit作为只读检查。
6. R3 Julia测试依赖跨语言子进程/临时工作目录与Julia包加载；会用到depot/JIT，S0未运行。S1最小等价检查需覆盖实际批量路径、R3桥接、M0原生解码、M0s/M1v；不自行扩展到30-seed重跑或R4 B/E2/E3。
7. 以既有三体F1等作为审计哨兵，不用于选参；测试失败、必要基线缺失或结果无法解释时停止，不进入S2。S1完成后仍需人工批准S2。

S0要求的文件均已找到，无需用户补造文件。下一项需要人工决定的是：是否批准按任务包进入S1冻结基线，以及采用何种主项目保护方式。S0后立即停止。
