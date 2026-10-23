# S2：执行前精确变更计划

日期2026-10-19。用户在S1完成后要求“继续执行”，本轮仅执行任务包S2；不执行S3归档、S4–S6或删除。S1基线为 `S1-20261019-f14a15e52881`。

## 按模块迁移顺序

| 批次 | old_path / 函数 | new_path / 函数 | 旧入口保留方式与风险 |
| --- | --- | --- | --- |
| A | src/data_bridge.py：file_hash、_toml、write_toml、read_matrix | src/bridge/numeric_io.py：原名 | 原文件重新导出；精度、TOML类型/顺序、异常文本不变 |
| A | src/data_bridge.py：fit_plain_lasso | src/solvers/plain_lasso.py：原名 | 原桥接继续导出；固定/验证路径整函数提取，不改数学 |
| A | scripts/r4_methods.py：fit_plain_bic | src/solvers/plain_lasso.py：原名 | 原脚本重新导出；复用原Adaptive私有_prepare/_bic_variance，不重写BIC |
| A | scripts/r4_methods.py：fit_node | src/solvers/methods.py：原名 | 原脚本重新导出；M1/M1v/M2/M3配置合成、身份、默认值不变 |
| B | scripts/r4_experiment.py：stamp、write_rows | src/bridge/numeric_io.py：原名 | 原实验模块重新导出；历史日志/CSV字段和序列化不变 |
| B | scripts/r4_experiment.py：load_dataset；scripts/r4_metrics.py：validate_splits | src/data/contracts.py：原名 | 两原位置重新导出；保留哈希检查、原数组方向和轨迹互斥规则 |
| B | scripts/r4_metrics.py：restore_coefficients | src/preprocessing/taylor.py：原名 | 原位置重新导出；原尺度偏置和向量/矩阵乘法顺序不变 |
| B | scripts/r4_metrics.py：binary_metrics、edge_scores | src/metrics/structure.py：原名 | 原位置重新导出；严格阈值、全集、max聚合、AP与NA口径不变 |
| C | scripts/r4_prepare.jl、scripts/r5_batch_julia.jl：重复训练标准化代码段 | src/preprocessing/taylor_scaling.jl：standardize_taylor | 旧Julia入口调用helper；保留各自退化错误文本、RNG和输出路径 |
| C | scripts/r4_prepare.jl：M0/M0s；scripts/r5_batch_julia.jl：M0调用 | src/solvers/stlsq.jl：fit_stlsq_raw、fit_stlsq_standardized | 仅封装原THISAdapter/native mySINDy与原尺度还原；M0保持原尺度，M0s仍独立身份 |

原有6个文件允许修改：`src/data_bridge.py`、`scripts/r4_methods.py`、`scripts/r4_metrics.py`、`scripts/r4_experiment.py`、`scripts/r4_prepare.jl`、`scripts/r5_batch_julia.jl`。新增模块/测试与S2审计产物；不删除原文件。既有Python/Julia CLI、TOML配置及正式method IDs不变。

`src/solvers/adaptive_lasso.py`、`src/this_adapter.jl`、两原论文目录、所有data/results/configs/reports、S0/S1审计保持原字节。R5噪声/统计工具与原R1生成器不迁移，避免把smoke/full授权、独立RNG namespace或历史口径混为一体。

## 逐步验收

先核对S1全清单、源码快照和审计产物，再依次A/B/C。每批运行针对该模块的测试、同一S1 golden的Python模型/边/指标比较及一个完整N8 trial的Julia→CSV/TOML→Python Adaptive+BIC→Julia原尺度还原/解码检查，新输出均在docs/refactor/S2_work内。

新测试覆盖训练标准化不泄漏、截距与预测还原、单节点/重复源/严格阈值、BIC NA和警告、gamma=0的加权公式/固定及BIC/验证路径退化。**公共Adaptive配置原本要求gamma>0；保持这一校验不变，仅在测试中将零指数得到的单位权重传入既有内部选择函数，验证数学退化，不对生产gamma=1做改动。**

原函数提取部分将与S1快照做AST正文比较。每批如发现核心数值/支持集变化，立即停止并回退该批相关修改，记录原因；不放宽容差，不接受“效果更好”。最后对除明确6文件外的原始文件和全部S1产物做哈希核对。

旧audit的源码哈希约束依然保持：它们可能因合法的S2源码变化按设计拒绝执行，不能改写历史manifest绕过。S2只使用新的审计入口，记录旧源码→新源码与差异，不覆盖正式协议。完成后停止，等待S3授权。
