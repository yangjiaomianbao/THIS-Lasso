# S1 Golden索引与字段契约

日期：2026-10-19。Golden用于后续重构的行为等价比较，不能用来选择算法参数或代表总体性能。原始数据和正式结果保持原位；本文件和索引不改写原产物。

## 固定输入与索引

[S1_golden_index.csv](S1_golden_index.csv) 对111个既有文件逐项记录 `path,size,sha256,fixture,role`，路径以项目根目录为基准。[S1_golden_contract.json](S1_golden_contract.json) 记录样本下标、矩阵维度、逐节点诊断摘录、状态、既有指标和全局模型选择锁。索引所有文件的SHA256也受全项目基线清单约束。

主样本：

```text
results/r5/e1_full_independent/
├── clean/seed_22261035/data/
└── seed_22261035/
    ├── noise_000/data/ 与 M0/M1/M2/M3/
    └── noise_050/data/ 与 M0/M1/M2/M3/
```

两档使用同一组480列，按原sample_manifest固定300训练、90验证、90测试；全部165个Taylor特征保留，未改变K/P或BIC条件。JSON提供各split的Python0-based位置；Julia矩阵位置1-based，d内节点ID保持1-based，0继续表示填充。X/Y为8×480，theta为165×480，d为165×3，系数为8×165。

配置为原 `configs/r5_e1_full_independent.toml` 及其引用的 `configs/r4_dry_run.toml`；全局 `selection_locked.toml` 的480项记录原样保留。完整配置、噪声输入、数据metadata、固定真值、轨迹和特征顺序哈希都能由索引追溯。

## 所需字段的真实来源

| 字段 | 既有文件与字段 | 语义 |
| --- | --- | --- |
| 原始状态/响应 | clean data中的states.csv、Y.csv；两档data/Y_noisy.csv、eta.csv | E1仅响应加噪，标准差比例0和0.05 |
| Taylor/monomial indices | clean data中的theta.csv、monomial_indices.csv | 原生THIS构造及顺序，不重新排序 |
| 训练均值/尺度 | clean data中的feature_means.csv、feature_scales.csv、standardized_theta.csv | 外部训练总体RMS标准化，常量均值0尺度1 |
| 原始拆分 | sample_manifest.csv、metadata.toml、initial_conditions.csv | 同一轨迹不能跨split；实际RNG namespace保留 |
| Ridge初值与权重 | M2/M3 diagnostics.toml中nodes[*].initial_coef、weights | 在外部标准化后的求解器输入单位中定义；不误称原物理单位系数 |
| 选中alpha与路径 | diagnostics.toml中alpha，alpha_path/bic_path或validation_path/selected_path_index | 原参数、网格、并列和警告语义 |
| 模型状态和理由 | status.toml及diagnostics.toml | OK或原BIC NA；不生成NA模型的虚假系数 |
| 标准化/原尺度系数 | standardized_coefficients.csv、coefficients.csv | beta_raw=beta_z/s；偏置同步还原 |
| 边score与真值 | edge_scores.csv | 全部成对56和三体168候选，零分也保留 |
| 结构解码 | inferred_structure.csv | 保留原生1–4阶返回，评价2/3阶；严格abs(C)>1e−8 |
| TP/FP/FN/TN/F1 | metrics.toml、全局trials.csv | 锁后既有评分，未用来选参 |
| 警告与溯源 | diagnostics.toml、candidate_warnings.csv、原测试日志、protocol/lock | 失败/警告记录和原始文件hash均保留 |

M0的lambda/rho/niter来自固定STLSQ配置，Ridge权重和LASSO alpha不适用；M1没有自适应权重/初值。零噪声M1/M2没有选中alpha和coefficients文件，因为BIC guard原本停止。它们是应保留的状态，不是需要补跑的缺失字段。

当前 `missing_required_applicable_fields=[]`。Golden创建无新拟合、无新数据生成。后续如果发现额外所需字段缺失，应明确列出，再按阶段要求申请必要的新基线生成，不能悄悄写回正式目录。

## 补充控制样本

固定原 `results/r4/dry_run/seed_22261017/data/` 以及M0s、M1v输出，涵盖公共标准化STLSQ与匹配独立验证的普通LASSO。它们未包含在正式R5方法列表中，但仍是不能删除的对照。只重放这一个已有R4 seed，不启动R4 B。

## 等价比较结果

Python重放结果为 [S1_python_compatibility.json](S1_python_compatibility.json)：R5零噪声M1/M2的16个节点仍精确返回既有NA理由；有效M1/M2/M3及R4 M1v的系数与原文件逐位一致。alpha、Ridge初值/权重、路径或选中索引、候选警告数量、严格支持集与结构计数均核对通过。

Julia结果为 [S1_julia_compatibility.log](S1_julia_compatibility.log)：原生get_θd重建theta/d、训练均值/尺度/Z与存储值完全相等；两档M0及R4 M0s系数最大绝对差均为0。所有有效保存模型的1–4阶解码与inferred_structure逐键核对，目标/重复源排除规则保持；额外74项断言通过。

审计比较对浮点数预设了 `atol=1e−12, rtol=1e−10` 的系数上限，并独立要求支持集、计数和NA理由精确一致；本轮实际系数差为0。alpha、初始化和权重采用更严格的单独容差，实际差同样为0。验收阈值不随看到的结果改变。

不复制整个实验到新位置，不以一个“好seed”替代30-seed统计，不扫描阈值或调rho/gamma/eps。未来S2若任一关键支持集、NA状态、选择路径或解码变化，应停止并诊断，不能以效果提高为由接受行为变化。
