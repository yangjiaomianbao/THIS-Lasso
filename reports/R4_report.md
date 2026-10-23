# R4 A 阶段执行报告

日期：2026-10-17（Asia/Shanghai）。**两个固定种子的 dry run 已完成：17 项测试通过，12 条方法×种子记录完整保留，8 条有效、4 条 BIC NA，没有剔除种子。** 本次只执行 A；30 个种子的 B 阶段及 R5 均未启动。

## 执行边界与修改文件

[R4 任务](../codex任务文件夹/R4_comparison.md) 明确规定：“本次首次执行只完成 A 并停止，除非用户后续明确批准 B。”因此本次完成可审阅的配置、脚本、测试、两种子结果和解释后停止。程序也拒绝以 B 阶段或 30 种子配置运行，人工检查表保持未勾选。

| 新增文件 | 用途 |
| --- | --- |
| [configs/r4_dry_run.toml](../configs/r4_dry_run.toml) | 预先锁定数据规模、种子、方法、尺度、阈值和选参规则 |
| [scripts/r4_prepare.jl](../scripts/r4_prepare.jl) | 同网络新轨迹、共同标准化、原始 STLSQ 和预处理控制组 |
| [scripts/r4_methods.py](../scripts/r4_methods.py) | 只读调用 R2/R3，增加有门槛的普通 LASSO/BIC 包装 |
| [scripts/r4_metrics.py](../scripts/r4_metrics.py) | 两层支持/边评分、严格阈值、AP、NA 与分割检查 |
| [scripts/r4_experiment.py](../scripts/r4_experiment.py) | 统一训练、锁定全部选择、Julia 解码、测试/结构评分和逐种子明细 |
| [scripts/r4_decode.jl](../scripts/r4_decode.jl) | 复用 R3 原生 THIS 规则，解码还原后的系数 |
| [scripts/r4_run.py](../scripts/r4_run.py)、[scripts/r4_audit.py](../scripts/r4_audit.py) | 命令日志及不可变性/数据/模型锁/指标审计 |
| [tests/test_r4_comparison.py](../tests/test_r4_comparison.py) | 17 项方法、数值和评价边界测试 |
| `results/r4/`、本报告、[R4 学习笔记](R4_learning.md) | 原始结果、逐方法配置/诊断、全部候选、异常与教学说明 |

前序数据、求解器、适配器、测试、报告和上游源码保持只读；没有增加依赖或修改 Pearson、H/SC、跨阶分组。代码与中间数据均写入 R4 允许范围。

## 预注册数据与方法

固定种子为 **22261017、22261018**，未搜索种子。沿用 R1 的 N=8 固定多项式网络、系数与完整 dmax=3 库，10 条成对真边、3 条三体真边。仅初始条件/轨迹随种子变化，不是不同网络拓扑。

每种子训练/验证/测试独立轨迹数为 100/30/30，每轨迹 t=0、0.05、0.10 三点，所以 **K_train=300**，验证/测试各 90，合计 480。状态/导数为 8×480，theta 为 165×480，d 为 165×3。导数来自解析 rhs，无噪声或差分；关闭 Pearson。新数据只保存在 `results/r4/dry_run/seed_<seed>/data/`，逐文件 SHA-256、原始秩/条件数及 R1 真值/特征一致性都有记录。

| 方法 | 特征与策略 |
| --- | --- |
| M0 | 原始 THIS/STLSQ，原尺度，锁定上游 this.jl 官方默认 λ=0.1、ρ=1、niter=10；不使用 R1 修订选参值 |
| M0s | 原始 mySINDy、相同官方数值参数，改用公共训练标准化特征，再还原系数；不更换算法 |
| M1 | 普通 LASSO＋BIC，公共标准化、无惩罚偏置，共用 R2 方差/秩门槛 |
| M2 | Ridge 初值 Adaptive LASSO＋BIC，与 M1 使用可比特征/偏置/方差条件 |
| M3 | Adaptive LASSO＋独立轨迹验证，γ=1、ε=10^-6、Ridge ρ=1 |
| M1v | 额外配对普通 LASSO＋与 M3 相同的验证准则，独立命名，绝不代替 M1 |

公共预处理只用训练列求均值/总体 RMS 尺度，第一行常数保持 1；M0s/M1/M2/M3/M1v 共用同一个 Z。最终统一还原原始 THIS 单位再评分，保证预测不变。R2 配置 `standardize=false` 因为 R4 已做公共缩放，权重明确定义在公共标准化坐标。M0/M0s 保留作者对常数项的原始惩罚，L1 方法不惩罚常数；这一差异没有被隐藏。

M3/M1v 使用相同的训练 alpha_max 比例网格（12 点）、验证 MSE 和 10^-12 相对响应方差的同分容差，容差内选较大 alpha。全部候选与警告保留。结构真值不参与选择，测试误差不参与选择。拟合前保存 [protocol.toml](../results/r4/dry_run/protocol.toml) 和配置/拟合脚本哈希；**全部 12 条方法记录在 21:53:20.318522+08:00 写入 [selection_locked.toml](../results/r4/dry_run/selection_locked.toml)，之后才解码评分与计算测试误差**。没有测试披露后的重选参。

## 逐种子结果

两个种子的下表计数相同；这是分别实测所得，不是用一个种子代替另一个。各自完整值、运行时长、测试 MSE 见 [trials.csv](../results/r4/dry_run/trials.csv)。AP 约 1 的数值可能保存为 0.9999999999999999。

成对交互（每种子 56 个候选）：

| 方法 | TP | FP | FN | TN | Precision | Recall | F1 | MCC | AUPRC(AP) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M0 | 5 | 0 | 5 | 46 | 1 | 0.5 | 0.6667 | 0.6716 | 0.5893 |
| M0s | 0 | 0 | 10 | 46 | 0 | 0 | 0 | NA | 0.1786 |
| M1：普通 LASSO＋BIC | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| M2：Adaptive LASSO＋BIC | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| M3 | 10 | 0 | 0 | 46 | 1 | 1 | 1 | 1 | 1 |
| M1v | 10 | 0 | 0 | 46 | 1 | 1 | 1 | 1 | 1 |

三体交互（每种子 168 个候选）：

| 方法 | TP | FP | FN | TN | Precision | Recall | F1 | MCC | AUPRC(AP) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M0 | 2 | 0 | 1 | 165 | 1 | 0.6667 | 0.8 | 0.8140 | 0.6726 |
| M0s | 0 | 0 | 3 | 165 | 0 | 0 | 0 | NA | 0.01786 |
| M1：普通 LASSO＋BIC | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| M2：Adaptive LASSO＋BIC | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| M3 | 3 | 0 | 0 | 165 | 1 | 1 | 1 | 1 | 1 |
| M1v | 3 | 0 | 0 | 165 | 1 | 1 | 1 | 1 | 1 |

两个种子中 M1/M2 的八个节点都因“满模型残差方差为零或低于可靠性门槛”拒绝 BIC。它们是 **4 条明确的 NA 方法记录，不是失败 F1=0，也没有静默切换到验证模式**。原因保存在对应 `status.toml` 和 `diagnostics.toml`，未生成伪系数或边列表。

M0s 的 MCC 分母为零，保存 NaN/NA；AP 仍有定义，因为真实正负类都存在。所有边分数为零时 AP 等于阳性占比。M0s 表现差反映“公共标准化＋保持作者数值阈值”这一预注册控制：标准化改变系数尺度，λ=0.1 的物理含义随之改变，不能解释成标准化普遍有害。

Taylor 支持层次每种子有 1320 个位置、21 个真非零：

| 方法 | Taylor TP/FP/FN | Taylor F1 |
| --- | --- | --- |
| M0 | 15 / 1 / 6 | 0.8108 |
| M0s | 8 / 8 / 13 | 0.4324 |
| M3 | 21 / 4 / 0 | 0.9130 |
| M1v | 21 / 9 / 0 | 0.8235 |

这与结构边表不同：自衰减、常数、重复源/含目标的多项式不能直接算三体边。每条有效 trial 的 `taylor_support.csv` 保存全部位置；`edge_scores.csv` 保存全部 56+168 个候选、真值、连续分数和判定；`inferred_structure.csv` 保存 Julia 返回的所有阶次。o=1/4 不计入成对/三体评分，记录在 metrics 的 ignored_returned_rows 中。

[paired_dry_run.csv](../results/r4/dry_run/paired_dry_run.csv) 明确保存同种子的差值：M3−M1v 两次的成对/三体 F1 与三体 AP 差均为 0；M2−M1、M3−M2 因 BIC NA 而记 NA。**本次没有显示自适应权重带来结构恢复优势，也无法回答 BIC 下 M2 是否优于 M1。** 两个种子不足以给出正式均值/标准差/置信区间或方法优劣结论，B 阶段尚未执行。

## 评价定义与完整性

结构阈值固定为原始 THIS 的严格 `abs(C_raw)>1e-8`，无阈值扫描。分数为同一合法目标/源集合所对应原尺度系数的最大绝对值，缺项记 0；原库中合法集合通常只有一列。Python 预测边集合逐项与 R3 Julia 解码器一致性核对，不把多项式次数直接作为交互阶数。

AUPRC 使用 scikit-learn `average_precision_score`，即 AP，非梯形积分；真实标签同时含正负类才计算，否则 NA。测试覆盖空正例、全正例、空候选、没有预测正例、类别不平衡和重复/含目标源。Precision 在无预测正例时记 0；Recall 无真实正例时 NA；MCC 分母 0 时 NA，不用人为补 0。

本次数据两种子原尺度条件数为 2894.99 / 2861.97，公共标准化后为 1476.56 / 1490.35，均满秩 165。K=300 比 R1 修订的大训练集更容易相关；没有因为条件数高或结果不好换种子。

## 数值警告、耗时与限制

**M1v 最小 alpha 的未选中候选累计产生 14 条 ConvergenceWarning，每种子 7 条，均达到 100000 次迭代上限。** 全部候选误差、迭代数和警告保存在诊断文件，并汇总到 [candidate_warnings.csv](../results/r4/candidate_warnings.csv)。最终选中候选均无收敛警告，KKT 违背量最多约 2.99×10^-11；M3 全路径没有警告，KKT 最大约 5.24×10^-13。

这些未收敛候选意味着整条路径未达到同等数值精度，是扩大 B 前需要审阅的问题；不能将其隐去或认为完整数值对照已无条件就绪。本次未在看到测试结果后修改预算、网格或剔除候选，也未修改已验收的 R2/R3 求解器。

全管线耗时约 **60.70 秒**；M3 每种子约 4.61/4.70 秒，M1v 约 20.06/20.25 秒，差异包含未选中候选的迭代成本。Julia 首次拟合含 JIT，Python 记录仅拟合/门槛/路径耗时，计时范围已逐 trial 保存；这些不是严格性能基准，不作速度优势结论。

有效方法测试 MSE：M0 约 1.83×10^-3；M0s 约 9.61×10^-3、9.83×10^-3；M3 约 2.87×10^-14、2.85×10^-14；M1v 约 8.06×10^-13、7.99×10^-13。它们仅在全部模型选择锁定后计算，不用于回调参数。预测误差较低不必然提升三体恢复，本次 M3 与 M1v 的结构指标就相同。

当前只有无噪声固定网络、两个初值种子和一个 K_train=300 case；不涉及 R5 的噪声网格，也未覆盖 K=1000、N=12。学习笔记回答权重归因、M2/M0 对照不足和预测/结构差别三个问题。

## 实际命令、环境和验收产物

使用原独立环境：Python 3.11.17、NumPy 1.26.4、scikit-learn 1.6.1、Julia 1.9.4，沿用 R0 依赖锁定。设置 BLAS/Julia 单线程、禁写 pycache。实际执行：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-r0/bin/python scripts/r4_run.py test
PYTHONDONTWRITEBYTECODE=1 .venv-r0/bin/python scripts/r4_run.py dry_run
PYTHONDONTWRITEBYTECODE=1 .venv-r0/bin/python scripts/r4_run.py audit
```

- [test.log](../results/r4/logs/test.log)：17 项单元测试通过，退出 0；覆盖双 BIC 合法/方差不可用/秩亏、配对验证、原尺度恢复、类别边界、严格阈值及 A/B 阶段限制。
- [dry_run.log](../results/r4/logs/dry_run.log)：真实 Python/Julia 命令、两种子数据诊断、各方法状态、选择锁、解码及评价，退出 0。
- [audit.log](../results/r4/logs/audit.log)：前序不可变性、上游锁定、模型锁、指标复算、候选警告和 R4 边界核对。
- [completion.toml](../results/r4/dry_run/completion.toml)：A 完成、B 未执行、8 OK/4 NA、无剔除种子、总运行时间。
- [artifact_manifest.toml](../results/r4/artifact_manifest.toml)：配置、脚本、报告、结果哈希；排除清单自身与正在写的审计日志。

正式输出拒绝覆盖。测试复跑需指定新日志标签；另做 dry run 可直接调用 `r4_experiment.py --config configs/r4_dry_run.toml --output results/r4/<新的目录>`，仍只允许同一预注册的两个种子；本次没有额外重复执行。运行前核对前序 R1/R3 清单，运行后再次核对保护文件，完整数据与真值哈希可追溯。

机器完成状态：A 的方法/数据/评分规则已锁定；两种子管线、逐 seed 明细、边列表、NA 与警告完整；教学解释完成。**B 的 30 种子、正式配对不确定性报告及人工验收尚未完成**，不能勾选全量完成或启动 R5。

R4 A 阶段完成，等待人工审阅及后续明确授权 B。

R4 未完成；等待人工验收；未执行下一阶段。
