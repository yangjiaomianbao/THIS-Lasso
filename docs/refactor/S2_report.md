# S2：公共组件提取与兼容验收

日期：2026-10-19（Asia/Shanghai）。用户在S1完成后要求“继续执行”，本轮按顺序只执行S2；S3–S6未执行。主项目仍没有Git，本轮没有初始化仓库或提交文件。

## 结果

按预先记录的 [S2_change_plan.md](S2_change_plan.md) 完成三批提取。只修改6个原有自有文件，新增8个公共模块与1个测试文件，没有删除或移动原文件。14个Python提取函数的完整AST与S1冻结源码相同，保持原签名、异常、数值运算和诊断字段。

最终95项Python测试通过（85项原有、10项新增），158项既有Julia断言通过；A/B/C每批均通过S1 Golden和一个完整N8 trial跨语言检查。最终C批21项跨语言/标准化/STLSQ断言通过。批量路径有效模型的系数、alpha、初值和权重与S1逐位一致，零噪声BIC NA及警告记录保留。

收尾全项目比较：S1清单31,479条中31,473条完全不变，另外6条恰好为获准修改的源码；无原文件缺失。正式R5全部4,113个文件、S1的111个Golden文件、S0/S1审计材料及两原论文目录保持原字节。详见 [S2_integrity_check.json](S2_integrity_check.json) 和 [S2_integrity.log](S2_integrity.log)。

## 实际代码布局与接口

| 新模块 | 职责 | 保留的旧入口 |
| --- | --- | --- |
| `src/bridge/numeric_io.py` | SHA256、CSV读取、TOML序列化、表格写入和时间字段 | `src.data_bridge`、`r4_experiment`原函数名继续导出 |
| `src/solvers/plain_lasso.py` | 普通LASSO固定/验证及guarded BIC | `src.data_bridge.fit_plain_lasso`、`r4_methods.fit_plain_bic` |
| `src/solvers/methods.py` | M1/M1v/M2/M3的原配置合成和派发 | `r4_methods.fit_node` |
| `src/data/contracts.py` | 固定数据哈希/维度读取、整轨迹split检查 | `r4_experiment.load_dataset`、`r4_metrics.validate_splits` |
| `src/preprocessing/taylor.py` | Python原尺度系数与截距还原 | `r4_metrics.restore_coefficients` |
| `src/metrics/structure.py` | directed候选全集、边score和原指标定义 | `r4_metrics.edge_scores`、`binary_metrics` |
| `src/preprocessing/taylor_scaling.jl` | Julia训练中心化/RMS及常量列、M0s逆变换 | 两旧Juliaprepare入口调用共享helper |
| `src/solvers/stlsq.jl` | M0原尺度和M0s标准化的轻量原生包装 | `r4_prepare.jl`、`r5_batch_julia.jl`路径不变 |

Python方法仍明确区分M1普通+BIC、M1v普通+独立验证、M2 Adaptive+BIC、M3 Adaptive+独立验证。Julia的 `fit_stlsq_raw` 保持M0原尺度与原生解码检查；`fit_stlsq_standardized` 保持M0s的原生mySINDy与原常量惩罚，不与M0混同。没有统一为一条数学求解路径。

完整函数签名及旧→新映射见 [S2_api_map.csv](S2_api_map.csv)，15个生产/测试文件差异索引见 [S2_changed_files.csv](S2_changed_files.csv)，逐行差异见 [S2_changes.patch](S2_changes.patch)。6个改动原文件为：

- `src/data_bridge.py`：保留request/response、schema验证及CLI；数值I/O与普通solver改为兼容导出。
- `scripts/r4_methods.py`：成为方法入口的兼容导出。
- `scripts/r4_metrics.py`：继续保留R4阶段guard，其他公共函数改为兼容导出。
- `scripts/r4_experiment.py`：提取表格/时间/数据加载，拟合、锁定、评分和阶段guard保留。
- `scripts/r4_prepare.jl`、`scripts/r5_batch_julia.jl`：只替换重复预处理和STLSQ调用，数据生成、RNG命名空间、配置、路径、metadata、选择/解码顺序保留。

Julia helper保留两个调用点原有的退化错误文本；M0s的还原运算顺序保留。R4/R5既有runtime字段依然表示实际耗时，提取包装后的未来墙钟计时不能要求与历史数值相同，也不用于速度结论。

## 不变的数学与实验契约

`src/solvers/adaptive_lasso.py`、`src/this_adapter.jl`、两原论文目录没有修改。Taylor递归/顺序/d索引、Ridge初值、gamma/eps、L1/LARS、BIC门槛/方差、alpha网格、并列规则、STLSQ、严格阈值、重复源/目标排除和AP定义原样保留。

Julia X/Y为[N,K]、theta为[P,K]、d为[P,3]；Python回归设计为[K,P]，d中的1-based节点和0填充不变。外部中心化/RMS仅使用训练列，常量均值0尺度1；R4/R5 Adaptive仍设置 `standardize=false`，没有二次标准化。原尺度偏置仍与系数一起还原。

没有改动任何现有配置、数据、results、reports、任务包、环境锁、S0/S1审计文件；没有运行30-seed全量、R4 B、E2/E3，没有调参或重选有利阈值。R5的420 OK、60 NA、0 FAILED和979条未选中候选警告继续由S1完整性清单保护，原始负面结果不变。

## 每批验证与1 trial对照

| 批次 | Python | S1 Golden重放 | Julia→Python→Julia |
| --- | --- | --- | --- |
| A最终验收（retry2） | 91通过 | R5同seed 0%/5%三Python方法及R4 M1v均匹配 | 15项通过 |
| B | 95通过 | 同一固定样本均匹配 | 15项通过 |
| C | 95通过 | 同一固定样本均匹配 | 21项通过，增加训练无泄漏、M0/M0s共享包装检查 |

不是把三批重复测试加总成不同测试数量。最终新增10项测试覆盖：gamma=0公式及固定/BIC/验证路径退化、公共gamma校验不变、旧API对象身份、14个函数AST、数据合同/split、单目标重复/目标排除与边界、原尺度预测/偏置还原。原R2/R3测试继续覆盖加权列变换、常量列、KKT、精度交换和NA。

固定trial为S1已选seed22261035，保留300/90/90样本和全部165特征。Python每批仍精确重放原始批量输入布局，对有效模型系数及选中参数/路径进行比较；M1/M2零噪声的16个节点NA理由逐字一致。R4 seed22261017的M1v也保持逐位相等。

跨语言smoke由当前桥接生成新审计输出，并在同一输入/布局下执行S1快照的桥接作逐位对照。原生theta/d与存储值完全相同；当前桥接与冻结桥接的标准化系数逐位相同，支持集/解码保持。原始批量和桥接两条路径之间原有的浮点差异保持，见下节，未强行统一其内存布局。

所有临时/桥接/重放输出和日志只写入 `docs/refactor/S2_*`，既有正式结果没有重新生成。更完整的实际命令及结果索引见 [S2_test_results.md](S2_test_results.md)。

## 两次新增断言失败及回退记录

首次A批的新gamma=0验证测试错误地要求普通与Adaptive两种**既有预测评价公式**的MSE逐位相等。差异最大8.673617379884035e−18；85项旧测试全部通过。按计划先把A批两个原文件恢复为S1字节，保留新文件但不调用，再从S1快照执行原普通路径，确认同样差异在重构前已存在。修正新增测试的比较基准为原路径，系数、alpha路径和选中索引仍要求精确相等；未提高容差、改算法或gamma校验。记录：[失败日志](S2_A_python_tests.log)、[回退](S2_A_rollback.json)、[诊断](S2_A_diagnostic.json)。

第二次A批的91项Python测试与批量Golden均通过，但新增跨语言断言把桥接输出与直接批量输出要求逐位相等。再次先恢复两个A批原文件，执行S1原桥接确认：当前桥接与原桥接逐位相等；原桥接和原批量本来存在约2.84e−15的标准化系数差、约1.34e−14的还原系数差。相同矩阵值在两条旧路径中分别为Fortran/C布局，轴均值运算已有约1.51e−16差异。记录：[失败日志](S2_A_retry1_trial_smoke.log)、[回退](S2_A_retry1_rollback.json)、[冻结桥接诊断](S2_A_bridge_diagnostic.json)、[布局检查](S2_array_layout_check.json)。

随后仅将跨语言逐位比较对象修正为**同一条冻结桥接路径**。跨路径仍沿用S1原有 `atol=1e−12, rtol=1e−10` 和严格支持集检查；没有放宽容差，也没有修改数据布局/中心化以“消除”原差异。两个原文件重新应用同一份AST不变的提取后，A最终、B、C均通过。所有失败日志、两次回退和诊断均保留，没有覆盖失败证据。

## CLI与历史哈希的边界

R3 bridge、R4 experiment、R5 batch的旧路径和参数仍存在，三个 `--help` 检查退出0；现有TOML默认值未变。修改的两个Julia入口及新增Julia模块语法检查通过，方法包装经真实trial验证。

**历史R阶段审计并不是当前源码的可变清单。** R3旧manifest中 `src/data_bridge.py`、R4中4个r4源码、R5中 `scripts/r5_batch_julia.jl` 的当前源码哈希与历史值不同，正好对应本次6个获准变化；历史manifest自身和其中的所有实验产物不变。详见完整性JSON的 `historical_manifests`。

包含旧源码保护的历史audit和实验driver可能因此按原规则拒绝继续运行；本轮保留这些guard，没有重写历史manifest来假称“所有旧源码未变”。本轮验收走独立S2入口和S1冻结源码对照。后续若要发起新的科研实验，应在新的明确协议/输出及源码清单下运行；是否另做版本化实验入口是后续需要人工确认的设计分叉，不在本轮绕过旧保护。

## 尚未做的整理与阶段停止

R1历史I/O/指标和smoke/full特有guard没有强行合并，避免改变原指标口径或独立seed修正。原logger完整CLI仍保留，只提取了共用表格/时间工具。未更新过期根README，未搬迁报告、历史快照、smoke或aborted数据，未创建删除清单或执行删除。

用户验收时可重点确认：上述保守的gamma=0测试范围、保留历史源码guard的选择，以及S3只做报告索引/归档而不继续算法修改。其余重复职责是否再提取，应作为明确的新范围评估，不以“更整洁”为由扩大本轮。

**S2完成；已停止，等待人工验收及单独批准S3。未执行S3–S6。**
