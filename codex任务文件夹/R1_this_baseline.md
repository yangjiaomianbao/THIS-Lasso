# R1 — 原始 THIS 基线复现与可控数据

**前置条件**：R0 已完成人工验收；`docs/data_contract.md` 存在。
**本阶段目标**：基于官方 Julia 代码建立小规模、真值已知、可重复的 THIS → STLSQ → 高阶结构解码实验。**不能修改原始 THIS 求解器、不实现 Adaptive LASSO**。

## 复制给 Codex 的独立指令

> 你只能执行 `R1_this_baseline.md`，先核对 R0 产物和 Julia 环境，若缺失就停下，不要自行跳过 R0。检查 `upstream/THIS/this.jl` 中 `this`, `mySINDy` 的实际接口；必须保留原始实现，不在上游改代码。
>
> 1. 用 Julia 实现小规模受控生成器：先 N=8，后留 N=12 扩展入口。优先构造有解析导数的稳定多项式动力学，例如自衰减项 + 稀疏成对项 + 少量跨两个不同源节点的三体乘积项。给每个目标节点明确的真值交互标签（源节点集合与目标节点）；防止不稳定发散。采用多组不同初始条件使特征矩阵有足够激励；轨迹如何拼接须明确记录。
> 2. 首轮用解析 `Y=dot X` 而不是含噪差分导数，先排除数值微分误差。用 `get_θd(X,dmax)` 生成 `θ,d`；建议 `ooi=[2,3]`, `dmax=3`，并在报告中说明 dmax=3 可能产生可对应更高阶的单项式，评分时**只评估指定的 2、3 阶**，不把 Taylor 多项式次数误当交互阶数。
> 3. 在固定数据上调用作者原始 `this(X,Y,ooi,dmax,λ,ρ,niter)`，保存 `Ainf,coeff,relerr`；原始回归 `mySINDy` 使用的是先岭初始化、阈值置零、重拟合。λ、ρ 和 niter 的任何调参必须记录，不能根据测试标签偷偷选取。
> 4. 写单位测试：简单 3 节点手工构造一组 `X`，核对 `get_θd` 与单项式索引；对给定已知多项式 `Y` 核查系数与交互索引的映射；检查 `θ:[P,K]`, `coeff:[N,P]`。如果不能完美恢复，诚实保存失败案例和特征矩阵秩/条件数，不得调整真值使结果看起来完美。
> 5. 保存 `data/r1/`：`states`, `derivatives`, `theta`, `monomial_indices`, `true_structure`, `metadata`，以及结果 `results/r1/coefficients`, `inferred_structure`, `metrics`。使用文件格式需注明精度、索引基底与维度；第一阶段可用 CSV + TOML/JSON 元数据。
> 6. 在 `reports/R1_report.md` 给出模型公式、真值结构、实验参数、实际命令、通过/失败的测试、恢复指标及误判例子。`reports/R1_learning.md` 用 1 个目标节点解释怎样从 `x_j*x_k` 特征得到针对 i 的三体交互证据；指出 `x_j^2` 不等同于两个不同源节点的三体作用。
> 7. 停在 R1：不写 Python 接口、不调用 Adaptive LASSO、不启用 Pearson 预筛选。

## 本阶段允许文件

`scripts/r1_*.jl`, `tests/test_r1_*.jl`, `data/r1/*`, `results/r1/*`, `reports/R1_report.md`, `reports/R1_learning.md`, 必要的 `docs/r1_data_description.md`。禁止修改上游和 R2 以后的新模块。

## 建议的验收问题

- 能否用同一份保存的数据重复得到相同 `Ainf`？
- 如果真实结构中不存在 i←(j,k)，THIS 为什么仍可能出现假阳性？
- 当 Taylor 特征高度相关，精确拟合 Y 是否等于精确恢复结构？（否）

## R1 研究进度检查表

- [ ] R0 前置条件与数据契约已核对。
- [ ] 已生成 N=8、真值可追踪的稳定多项式轨迹与解析导数。
- [ ] 已检验 `get_θd` 输出值、索引及矩阵维度。
- [ ] 已运行未经改动的原始 `mySINDy/this`。
- [ ] 已保存固定数据、单项式索引、真值结构和恢复系数。
- [ ] 已计算成对及三体的结构恢复指标并记录失败情形。
- [ ] **人工验收通过**，明确授权开始 R2。
