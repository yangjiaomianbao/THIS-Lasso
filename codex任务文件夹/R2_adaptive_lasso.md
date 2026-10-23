# R2 — 独立复现与验证 Adaptive LASSO 求解器

**前置条件**：R1 人工验收；已有可读取的固定回归数据，且数据契约明确。
**本阶段目标**：独立实现并数值验证可迁移的 Ridge-initialized Adaptive LASSO。**暂不接入 THIS Julia 结构解码**。

## 数学接口（务必先让 Codex 解释）

在样本数 K、特征数 P 的训练数据上求解：

`min_beta [ ||y-X beta||²/(2K) + alpha * Σ_j w_j |beta_j| ]`

`w_j=(|beta_init_j|+eps)^(-gamma)`，其中 `beta_init` 来自训练集 Ridge；截距/常数特征不应被惩罚。令 `a_j=1/w_j=(|beta_init_j|+eps)^gamma`，`X*_j=X_j*a_j`，求解标准 LASSO 得 `theta_j` 后恢复 `beta_j=a_j*theta_j`。**最终仍是 Adaptive LASSO**。

## 复制给 Codex 的独立指令

> 你现在只执行 `R2_adaptive_lasso.md`。先阅读 Su 官方 `GeneralInteraction.py` 相关求解代码和其 README，给我一页对照说明，解释 Su 原实现是如何利用初始系数做加权特征变换、以及如何调用 LARS/BIC。不要复制整套 Kuramoto 动力学模拟到项目。
>
> 1. 新建 `src/solvers/adaptive_lasso.py`，实现清晰的单节点求解接口 `fit_adaptive_lasso(X: [K,P], y:[K], config) -> coef:[P], diagnostics`。默认 Ridge 初始化，同时保留可以切换的初始化方式以便记录与 Su 原始 OLS 的差异。明确说明 gamma、eps、Ridge rho 的作用。
> 2. 使用**训练数据**计算归一化、Ridge 和自适应权重；禁止读取测试集拟合标准化或决定权重。Taylor 特征含常数项时，避免与额外截距重复；支持把系数恢复到原始 THIS 特征尺度，并编写测试确认恢复后预测不变。
> 3. 用 `LassoLarsIC(criterion="bic")` 或 LARS 路径处理经自适应缩放的特征；写入有效性检查：当 K≤P、满模型残差方差无法可靠估计或数值不稳定时，不允许静默套用默认 BIC；提供清晰异常说明以及**独立轨迹验证集**替代选择方式。务必核对 scikit-learn 的实际目标函数缩放，区分 L1 参数 alpha 与 THIS STLSQ 阈值 λ。
> 4. 编写 `tests/test_r2_*.py`：已知稀疏系数、含噪样本、K<P、高相关/重复特征、极小初始系数、无惩罚截距、固定随机种子、加权目标与缩放目标等价、KKT 近似检查、原始尺度系数及预测还原。
> 5. 保存日志和配置：算法版本、sklearn 版本、训练/验证分割、gamma/eps/rho/alpha、收敛警告和选出的系数。没有失败数据就不要宣称“对弱耦合更好”，本阶段只验证求解器实现正确。
> 6. 用 `reports/R2_learning.md` 向我逐行解释最核心的 20–40 行代码及 `beta_j=a_j*theta_j` 的含义；回答为什么“标准 LASSO 求解器”不等于研究模型使用普通 LASSO。
> 7. R2 结束后停止。绝对不要接入 Julia、不要运行大规模网络对比。

## 本阶段允许文件

`src/solvers/adaptive_lasso.py`, `src/solvers/__init__.py`, `tests/test_r2_*.py`, `scripts/r2_solver_demo.py`, `reports/R2_report.md`, `reports/R2_learning.md`, 必需的 Python 依赖文件。仅可读取 `data/r1/`，不得改写。

## 验收要点

一个优化目标的数值等价性测试比“看上去恢复了几个非零系数”更重要。无噪声且 alpha>0 时真实系数也可能被缩小，不以逐项严格等于真值作为唯一合格标准。

## R2 研究进度检查表

- [ ] 已阅读 Su 原实现并标注其中真正负责 Adaptive LASSO 的步骤。
- [ ] Ridge 初始化、自适应权重及尺度变换公式已实现。
- [ ] 常数项与特征标准化后系数恢复测试通过。
- [ ] 加权目标等价性、KKT 与边界情况测试通过。
- [ ] BIC 适用条件与验证集备选策略已实现并记录。
- [ ] 回归结果和参数记录完整，学习笔记可解释核心代码。
- [ ] **人工验收通过**，明确授权开始 R3。
