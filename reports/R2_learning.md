# R2 学习笔记

## 与 Su 原实现的一页对照

阅读来源为本地只读的 [Su README](../upstream/Su-AdaptiveLASSO/README.md) 和 [GeneralInteraction.py](../upstream/Su-AdaptiveLASSO/GeneralInteraction.py)。锁定 commit 为 `fa0d662f68b7b41a1c6c3401af9504929126dafd`，MIT；THIS 的锁定 commit 为 `865efc43fe8ec21966659b482ca5756396974425`，GPL-3.0。完整源码锁定记录见 [upstream_lock.json](../environment/r0/upstream_lock.json)。没有运行作者的振子模拟，也没有改写上游文件。

Su 的 `solve_ada_lasso_one_node`（1936–1976 行）接收外部初始系数；主入口 `demo_solve`（2312 行起）实际调用 `solve_ols`，虽然变量名仍含 mle。它把二体和三体特征拼接成 `[P,K]`，取初值绝对值 `a=abs(beta_init)`，逐特征乘 `a`，再转置为 `[K,P]`。随后调用 `LassoLarsIC(criterion="bic")`，最后以 `coef_ * a` 还原各阶系数。多节点的 `solve_ada_lasso`（1843 行起）做相同变换。因此，真正的自适应步骤是“由初值确定特征缩放”及“乘回缩放”，而非另有一个特殊优化器。

| 方面 | Su 当前源码 | 本项目 R2 |
| --- | --- | --- |
| 初值 | 由调用方提供 OLS/MLE 结果，主演示为 OLS | 默认训练集 Ridge/SVD，可显式切换 OLS |
| 缩放 | `a=abs(beta_init)`，对应 γ=1、ε=0 | `a=(abs(beta_init)+eps)**gamma`，默认 γ=1、ε=10^-6 |
| 库和维度 | 振子二/三体正弦库，再转置 | 接收任意固定 `[K,P]` 回归矩阵，保持列顺序 |
| 截距 | 部分交互类型禁用，否则输出频率时取模 2π | 单个非零常数列承载无惩罚偏置；无常数列时可加截距，无取模 |
| 选参 | LARS 路径及 BIC | 合法时同用 LARS/BIC；不合法时抛异常，调用方显式选独立轨迹验证 |
| 数值核验 | 该入口未提供权重稳定项及这些检查 | 原始尺度目标等价、预测还原、KKT、警告与路径记录 |

ε 防止零初值永久删除特征，但不保证弱耦合可恢复；γ 控制不同初值对应的惩罚差别。ρ 控制 Ridge 初值的收缩，在高相关库中提供稳定起点，它不是最终 L1 惩罚。OLS 模式配合关闭标准化可对照作者初始化；本项目仍保留 ε 与额外检查，因此不声称逐位复刻作者结果。

## 接口、目标与原始尺度

入口 [fit_adaptive_lasso](../src/solvers/adaptive_lasso.py)：`fit_adaptive_lasso(X, y, config, ...) -> (coef, diagnostics)`。输入 `X:[K,P]`、`y:[K]`，输出 `coef:[P]`，预测为 `X @ coef + diagnostics['intercept']`。调用者负责传入 `theta.T`，求解器不认识 Julia 的结构或交互阶数。

设训练均值为 μ、列尺度为 s，`Z=(X-μ)/s`，`yc=y-y_mean`。Ridge 先在 Z 上求初值 η，再转回 `beta_init=η/s`。按原始单位定义

`w=(abs(beta_init)+eps)^(-gamma)`，`a=1/w`。

本项目目标为 `||y-b-X beta||²/(2K) + alpha*sum(w*abs(beta))`，常数项不参与惩罚。令 `beta=a*theta`，则 `w*abs(beta)=abs(theta)`，设计矩阵变为 `U=(X-μ)*a`。因此在 U 上调用标准 LASSO，恰好求解上述加权目标。

标准化不能漏掉 s：代码用 `U=Z*(s*a)`，还原仍为 `beta=a*theta`。若直接用 `Z*a`，却仍乘 a 还原，就会改变预测或目标。在标准化系数 `η=s*beta` 上等价权重应为 `w/s`。这里的标准化主要服务初始化，最终目标明确对应原始单位；换特征物理单位也会改变此自适应模型，不能声称其尺度不变。

本地 scikit-learn 1.6.1 的 `_coordinate_descent.py` 与 `_least_angle.py` 确认 LASSO 为 `SSE/(2K)+alpha*L1`；`_ridge.py` 确认 Ridge 为 `SSE+rho*L2²`。若把 Ridge 目标也写成 `SSE/(2K)+lambda_R*L2²`，则 `lambda_R=rho/(2K)`。L1 的 alpha 是目标中的惩罚强度；THIS/STLSQ 的 λ 是硬阈值，二者不能直接互换。

## 核心 28 行逐行解释

以下为 [adaptive_lasso.py](../src/solvers/adaptive_lasso.py) 第 214–241 行的原样摘录；Z、yc、均值、尺度、常数列已在训练数据上准备好。

```python
    if len(active):
        if c.initialization == 'ridge':
            initial_scaled = Ridge(alpha=c.ridge_rho, fit_intercept=False, solver='svd').fit(z, yc).coef_
        else:
            initial_scaled = np.linalg.lstsq(z, yc, rcond=None)[0]
        beta_initial = initial_scaled / scales[active]
        with np.errstate(over='raise', under='raise', invalid='raise', divide='raise'):
            try:
                a = (np.abs(beta_initial) + c.eps)**c.gamma
                w = 1.0 / a
            except FloatingPointError as exc:
                raise ValueError('Adaptive weights overflow/underflow; revise gamma/eps and units explicitly') from exc
        U = z * (scales[active] * a)  # == (X_active - train_mean) * a
        theta, alpha, selection = _select(X, p, U, a, c, validation)
        beta = a * theta
    else:
        beta_initial = a = w = theta = beta = np.empty(0)
        U = np.empty((n, 0))
        alpha = float(c.alpha) if c.selection == 'fixed' else 0.0
        selection = dict(selection='constant_only', requested_selection=c.selection,
            reason='No penalized variable; model selection not performed', warnings=[], convergence_warning=False)
    coef = np.zeros(features)
    coef[active] = beta
    bias = float(p['response_mean'] - p['means'][active] @ beta)
    intercept = bias
    if p['bias_column'] is not None:
        coef[p['bias_column']] = bias / X[0, p['bias_column']]
        intercept = 0.0  # bias is already represented by the original constant column
```

| 相对行号 | 解释 |
| --- | --- |
| 1 | 仅在有非常数特征时进入稀疏回归。 |
| 2 | 根据配置选择 Ridge 初值。 |
| 3 | 用训练 Z、yc 做 SVD Ridge；居中已完成，不再拟合截距。 |
| 4 | 另一条初始化分支。 |
| 5 | 最小二乘初值；秩不足时返回最小范数解。 |
| 6 | 从标准化系数转回原特征单位。 |
| 7 | 溢出、下溢、非法运算、除零均显式报错。 |
| 8 | 捕获权重计算失败。 |
| 9 | 由训练初值、ε 和 γ 构造逆权重 a。 |
| 10 | 惩罚权重 w 为 a 的倒数。 |
| 11 | 捕获浮点异常。 |
| 12 | 提示显式修改单位或参数，不静默裁剪权重。 |
| 13 | 抵消标准化尺度，构造与原始加权目标等价的 U。 |
| 14 | 按 BIC、显式验证集或固定 alpha 求 theta。 |
| 15 | 乘回 a，得到原始尺度的 beta。 |
| 16 | 无非常数特征时进入解析偏置分支。 |
| 17 | 不存在惩罚坐标，相关向量为空。 |
| 18 | 保留样本数，构造零列设计矩阵。 |
| 19 | 固定模式记录原 alpha，其他模式用 0 表示未选参。 |
| 20 | 诊断明确标记仅常数模型并记录请求的策略。 |
| 21 | 说明未执行模型选择，未产生优化器警告。 |
| 22 | 分配原列数的输出，保证不重排或丢列。 |
| 23 | 将回归结果放回非常数列。 |
| 24 | 从训练均值计算原坐标偏置。 |
| 25 | 无显式常数列时由 intercept 承载偏置。 |
| 26 | 检查原设计中是否已有非零常数列。 |
| 27 | 用常数列的值换算其系数。 |
| 28 | 避免常数列与额外截距重复承载偏置。 |

“使用标准 LASSO 求解器”仍得到 Adaptive LASSO，因为求解器看到的是依训练初值改变过的 U，乘回后在原 X 上每个 beta 的惩罚权重不同。普通 LASSO 对原 X 的每个系数施加相同 L1 权重，这两个研究模型的目标不同。

## BIC 与验证集的边界

BIC 仅在 K>P、满模型有残差自由度、非常数设计满秩、标准化及加权条件数不超过 10^8、残差 MSE 大于训练信号 MSE 的 10^-12 时启用。噪声方差用训练满模型 `SSE/(K-q-bias_df)`，再显式传给 `LassoLarsIC`；这是一组可复查的实用数值门槛，不是任意数据下 BIC 正确的理论保证。

固定 R1 是无噪声的精确多项式，满模型残差接近零，BIC 拒绝是预期行为。演示显式设置 `selection='validation'`，要求训练/验证逐样本轨迹 ID 不相交。训练均值、Ridge 和权重固定；验证轨迹只评价预先定义的 alpha 网格。所有候选与警告保留，误差在指定容差内选较大 alpha。接口不接收测试数据或结构真值。

30 项测试和演示只支持“实现满足这些数值检查”。未作跨网络、含噪强度或三体恢复性能比较，因此尚不能认定 Adaptive LASSO 优于 STLSQ。

供自己思考的 3 个问题：

1. 若缩放设计矩阵后漏掉 `beta=a*theta`，预测和惩罚各会发生什么变化？
2. 为什么零初值时 ε=0 会永久删除一列，而 ε>0 仍不保证恢复该列？
3. 为什么同一轨迹的相邻采样点不能随机拆到训练集和验证集？

R2 完成；等待人工验收；未执行下一阶段。
