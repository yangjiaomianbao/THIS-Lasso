# R3 — Julia THIS + Python Adaptive LASSO 融合接口

**前置条件**：R1 数据与 R2 通过测试的求解器都已验收。
**本阶段目标**：保留 THIS 的 `get_θd` 与结构解码，通过最小化接口替换 `mySINDy` 输出的系数矩阵；让同一组数据可以切换 STLSQ / LASSO / Adaptive LASSO。**不开展批量性能测试**。

## 复制给 Codex 的独立指令

> 你现在只执行 `R3_integration.md`，先只读核对 R1 的数据合同、R2 solver 输出、THIS `this.jl` 的结构解码逻辑。报告如何最少修改文件；禁止直接编辑 `upstream/THIS/*` 或 `upstream/Su-AdaptiveLASSO/*`。
>
> 1. 使用简单、透明、可调试的 CSV + TOML 文件桥接（首轮网络小，不追求极限速度）。Julia 写出 `theta.csv:[P,K]`, `Y.csv:[N,K]`, `monomial_indices.csv:[P,dmax]`, `metadata.toml`，Python 读取后 `X_design=theta.T:[K,P]`；Python 回写 `coefficients.csv:[N,P]`。浮点数保存至少 17 位有效数字，并自动检测列数/行数。注意只有 1 行时的 CSV 读取形状。
> 2. 约定 `d` 为 Julia 从 1 开始的单项式节点编号，0 表示空白；Python 中不得随意将其当作数组索引转换或重排序。元数据保存 `feature_order_hash` / 特征条目总数，供回写校验。
> 3. 新建 `src/this_adapter.jl` 或等价 Julia 包装器，实现 `run_this_solver(X,Y,ooi,dmax;solver=...)`：对于 `stlsq` 路径，调用原始 `this`；对于 `lasso/adaptive_lasso`，先按 `get_θd` 构造特征并调用 Python 返回 coeff，再**复用同样的**单项式过滤和 Ainf 解码规则。可提取原算法解码逻辑为本项目私有、测试覆盖的共享函数，但不得在未经说明时改动判零阈值或交互定义。
> 4. 编写 `tests/test_r3_*`：矩阵往返误差、索引与哈希一致、1 行/1 列形状、每个节点系数与真值的特征对应、使用原始 STLSQ 系数走包装器解码时 Ainf 与原生 this 的精确一致或约定容差等价。
> 5. 先只在 R1 的一个 N=8 种子上运行三种 solver，打印非零 Taylor 系数表与解码后的 2/3 阶交互，计算简单 F1/MCC 作为接口检查。它们此时谁好谁坏均不作结论。保存真实命令、运行日志、错误与修复步骤。
> 6. `reports/R3_learning.md` 明确演示一个 `x_j*x_k` 的列索引怎样从 Julia 导出，在 Python 回归，再回到 Julia 正确映射为目标 i 的三体交互；解释为什么重复索引 `x_j^2` 不算两个不同源节点。
> 7. 立即停在 R3，不运行 R4 对照网格，不修改 Pearson 候选筛选，不写 H/SC 分类器。

## 本阶段允许文件

`src/this_adapter.jl`, `src/data_bridge.py`, `scripts/r3_*.jl`, `scripts/r3_*.py`, `tests/test_r3_*`, `results/r3/*`, `reports/R3_report.md`, `reports/R3_learning.md`。R1 基准数据及 R2 solver 只读（除非出现经用户确认的明确 bug）。

## 接口验证关键点

THIS 源码原始 `this` 会构建 `Ainf`，它不是仅仅按多项式次数判边；其内部还检查源节点是否重复、目标节点是否属于源集合。**首轮评价只包含 2 阶和 3 阶，明确是否忽略 o=1/4 的返回项**。

## R3 研究进度检查表

- [ ] 已核对 Julia/Python 维度、索引起点及输出契约。
- [ ] 已建立可重复、具备精度检查的 CSV/TOML 文件交换。
- [ ] 三种 solver 接口能作用于同一 R1 数据。
- [ ] 特征顺序与 monomial indices 在来回交换后保持一致。
- [ ] 原始 STLSQ 系数通过适配器的结构解码与原程序一致。
- [ ] 已打印目标节点三体交互的“特征→系数→边”实例。
- [ ] **人工验收通过**，明确授权开始 R4。
