# R1 固定基准数据与结果说明

本阶段入口为 [R1 报告](../reports/R1_report.md)。**当前修订基准使用 `data/r1/revision/` 与 `results/r1/revision/`；原 `data/r1/` 根目录的数据保留为失败对照。** 两者都为 N=8、单个固定种子、无噪声解析响应。以下先说明修订版，再保留原版契约。

## 当前修订版：更多独立初值与验证选参

真实方程、全部耦合系数、dmax=3、Julia 特征顺序与原版本完全一致。主种子固定为 21261016，第 t 条轨迹仍用 MersenneTwister(seed+t) 和 Uniform[-1,1] 初值，不搜索种子。

采用 512 条训练、128 条验证、128 条测试轨迹；每条仍以 RK4、dt=0.01 积分，只保存 t=0、0.05、0.10，避免将大量观测集中在接近零的衰减尾部。共 768 条轨迹、2304 个样本。训练/验证/测试轨迹编号分别为 1–512、513–640、641–768，对应 1536/384/384 列。

修订数据文件名及索引约定与原版本一致，数值矩阵维度改为：states/derivatives 为 8×2304，theta 为 165×2304，monomial_indices 为 165×3，true_coefficients 为 8×165，initial_conditions 为 8×768。sample_manifest.csv 标记每列来源，metadata.toml 锁定 8 个 CSV 哈希。

训练特征矩阵数值秩 165，原尺度条件数约 21.28，按列 RMS 缩放后的条件数约 16.07；后者仅用于质量检查，拟合保持原尺度。固定门槛为满行秩、两个条件数分别≤1000/300，未通过时停止而非重新挑种子。

验证选参不读取结构标签或测试样本。30 组 λ/ρ 的全部验证误差保存在 results/r1/revision/validation_candidates.csv；优先最小验证归一化平方误差，在 10^-12 数值容差内依次偏好更稀疏、λ 更大、ρ 更小的模型。最终 λ=0.03、ρ=0、niter=10，在测试评价前写入 selection_locked.toml。

默认参数对照放在 results/r1/revision/default/，验证选参结果放在 selected/；各含 coefficients.csv、inferred_structure.csv、edge_audit.csv、metrics.toml。二者读取同一修订数据，均使用作者未修改的 THIS/STLSQ。metadata.config 中继承的 lambda=0.1、rho=1 指作者默认对照；选参后的实际值以 selection_locked.toml 和 selected/metrics.toml 为准。

验收文件 acceptance.toml 显示本次干净基准通过：成对/三体 Precision、Recall 均为 1，测试归一化导数误差约 1.41×10^-30。结构标签对应固定网络；测试轨迹验证的是导数预测，并不构成跨网络泛化结论。

复查修订版只读保存数据：

```bash
environment/r0/julia.sh --startup-file=no --project=environment/r0/julia scripts/r1_run.jl revision_replay revision_replay_manual
```

生成入口拒绝覆盖既有修订数据或协议。参数锁定、验证表和旧失败案例都需要保留；没有因效果改善而删除旧结果。以下各节的旧尺寸和参数仅描述保留的原基准。

## 动力学与稳定性

令 `a_i=0.8+0.05*(i-1)`，所有目标都有自衰减 `-a_i*x_i`。此外包含三类预先固定的项：

- i=2…N：`b_i*x_(i-1)`，b 按 `[0.20,-0.16,0.12]` 循环。
- i=4,6,8,…：`0.07*(-1)^(i/2)*x_(i-3)`。
- i=3,5,7,…：`c_i*x_(i-2)*x_(i-1)`，c 按 `[0.08,-0.14,0.20]` 循环。

真实结构为 10 条有向成对作用、3 条有目标节点的三体作用。自衰减参与系数拟合，但不作为成对/三体真值边。完整边及系数见 [true_structure.csv](../data/r1/true_structure.csv)。

每个源节点编号都小于目标编号，系统呈三角结构。逐节点衰减与变参数公式可说明所有状态最终趋于零；在本次初值范围内还可直接检查立方体不变性：当 `|x_j|≤1` 时，第 i 个方程的耦合绝对上界为其耦合系数绝对值之和，严格小于 a_i。故在 x_i=1/-1 边界，向量场指向内部。最小余量为 0.65，`[-1,1]^N` 对精确动力学正向不变。

数值轨迹采用固定步长经典 RK4；没有截断、夹紧状态或更改真实边来改善结果。所有积分步上的实测最大绝对状态为 0.9997599374590771。

## 轨迹、时间和分割

- 主种子 20261016；第 t 条轨迹使用独立的 Julia `MersenneTwister(20261016+t)`。
- 初始条件各分量独立 Uniform[-1,1]，40 条轨迹均保存初始状态。
- 积分步长 0.01，200 步，时间区间 [0,2]；每 5 步保存一次，含初始时刻，每轨迹 41 个样本。
- 轨迹 1–24 为训练集（984 列），25–32 为验证集（328 列），33–40 为测试集（328 列）。同一轨迹不跨集合。
- 拼接顺序为轨迹编号递增，每条轨迹内时间递增。时间会在新轨迹开始时归零；没有跨轨迹数值微分。
- Y 直接在每个保存状态上计算 `rhs(X)`。RK4 状态近似误差与数值差分误差不同；这里没有数值差分响应误差。
- 所有矩阵保持原尺度，没有中心化或标准化。验证/测试数据不参与训练、权重拟合或参数选择。

## 文件与精度

数值矩阵不含表头，Float64 以 `%.17g` 保存；整数索引按十进制保存。表格型文件含表头。节点、样本和轨迹编号均从 1 开始；d 内 0 为填充。

| 文件 | 形状/字段 | 含义 |
| --- | --- | --- |
| `states.csv` | 8×1640 | 状态 X，按列拼接轨迹 |
| `derivatives.csv` | 8×1640 | 解析 Y=f(X) |
| `theta.csv` | 165×1640 | 原始 get_θd 生成的 dmax=3 特征，全量样本 |
| `monomial_indices.csv` | 165×3 | 与 theta 行顺序完全对应的 Julia 节点编号 |
| `true_coefficients.csv` | 8×165 | 含自衰减的完整真值 Taylor 系数 |
| `initial_conditions.csv` | 8×40 | 每列一个轨迹初始状态 |
| `sample_manifest.csv` | 1640 行；sample_id, trajectory_id, split, time | 每个样本的来源与集合 |
| `true_structure.csv` | 13 行；order, target, source_1, source_2, coefficient | 2/3 阶真实作用；成对项 source_2=0 |
| `metadata.toml` | 配置、维度、轨迹种子、哈希、秩和条件数 | 数据来源和可追踪契约 |

`metadata.toml` 的 file_sha256 锁定全部 8 个 CSV；feature_order_hash 为 monomial_indices.csv 的逐字节 SHA-256。不得重排特征或去除 d；真值系数同样严格使用此特征顺序。

训练 θ 为 165×984，数值秩 165，条件数约 6.8856×10^8。满秩不等于良好的数值条件；单条轨迹内样本相关，984 列也不表示 984 个独立初始条件。

## 固定求解与结果

作者原始调用为 `this(X_train,Y_train,[2,3],3,0.1,1.0,10)`。参数为作者默认值，在生成数据和查看恢复指标之前写入 protocol.toml；本次没有参数搜索。

| 文件 | 内容 |
| --- | --- |
| `results/r1/protocol.toml` | 预先固定的配置、训练集合与源码 commit |
| `results/r1/coefficients.csv` | 原始 THIS 返回的 8×165 系数 |
| `results/r1/inferred_structure.csv` | 全部阶次的原始 Ainf 行，统一为 order,target,source_1,source_2,source_3,coefficient |
| `results/r1/edge_audit.csv` | 全部 224 个合法 2/3 阶候选的真值、预测、TP/FP/FN/TN、系数 |
| `results/r1/metrics.toml` | 分阶计数与指标、分集合回归误差、原始 relerr、运行时间 |
| `results/r1/replay_check.toml` | 从保存矩阵重放的哈希、特征值、系数和边一致性 |
| `results/r1/history.log` | 旧诊断源码、逐系数表、摘要及原单元测试/首次基线/重放日志的合并记录，保留原文件名、SHA-256、真实命令与退出码；原独立文件已清理，当前修订日志仍在 revision/logs/ |

inferred_structure.csv 的 0 源节点仅用于统一表宽；按 order 取相应源数量即可无损还原 Ainf。只对输出表的边顺序排序，未重排 Taylor 特征。原始 Ainf 的 1/4 阶一并保留表示能力，本次两者均无返回行；评价仅取 2、3 阶。

候选全集包含 N*(N-1)=56 条成对边、N*binomial(N-1,2)=168 条三体边。三体源无序，目标有向；不重复计算源排列。没有预测正例时 Precision 记 0，F1 按计数公式记 0；MCC 分母为 0 时记 NaN，表示未定义。

## 复查与扩展

正常复查只需读取已保存数据，不重新生成：

```bash
cd /home/yg/THIS-Lasso
environment/r0/julia.sh --startup-file=no --project=environment/r0/julia scripts/r1_run.jl replay replay_manual
```

日志标签不可覆盖，重复执行时换一个新标签。生成入口会拒绝覆盖已有数据目录或 protocol.toml，保护当前基准。N=12 扩展入口为 `scripts/r1_baseline.jl --nodes 12 --data-dir <新的数据目录> --result-dir <新的结果目录>`；本次没有执行 N=12 实验。R1 不包含 Python 回归接口或跨语言桥接。
