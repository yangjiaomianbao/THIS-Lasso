# THIS + LASSO 文档导航

当前唯一科研主报告：[R5 E1：30个独立初值seed正式报告](../reports/R5_report.md)。该报告的路径、结论与数字保持原样。

项目整理已完成S0–S6：[最终审计与变更总览](refactor/S6_final_report.md)、[复现命令与新实验边界](REPRODUCE.md)。

## 当前研究事实

E1响应噪声实验已完成：480条记录，420 OK、60个零噪声BIC NA、0 FAILED。当前结果不支持Adaptive LASSO普遍优于普通LASSO；20%噪声下M2三体F1低于M1。原始数据、诊断、警告和全部种子保留。

- [正式结果目录](../results/r5/e1_full_independent/)
- [逐trial原始指标](../results/r5/e1_full_independent/trials.csv)
- [配对统计及区间](../results/r5/e1_full_independent/paired_summary_30.csv)
- [弱边诊断](../results/r5/e1_full_independent/weak_edge_diagnostics.csv)
- [三体F1/AP图](../results/r5/e1_full_independent/figures/triple_f1_ap.png)
- [候选数值警告](../results/r5/e1_full_independent/candidate_warnings.csv)

## 理解实现与复现证据

- [数据与跨语言接口合同](data_contract.md)
- [R1固定数据及修订说明](r1_data_description.md)
- [R2求解器原理](../reports/R2_learning.md)、[R3桥接与解码验证](../reports/R3_report.md)
- [S0真实依赖图](refactor/S0_dependency_map.md)
- [S1冻结基线](refactor/S1_baseline.md)、[Golden字段及输入](refactor/S1_golden.md)
- [S2公共模块与接口](refactor/S2_report.md)、[回归测试记录](refactor/S2_test_results.md)
- [环境使用与重建依据](../environment/r0/README.md)

## 历史材料与未完成研究

- [历史归档总索引](../archive/INDEX.md)：R0–R4报告、smoke、aborted及旧README均可追溯。
- [原R4六方法对照](../reports/R4_report.md)：M0s和M1v保留；完整B阶段尚未运行。
- [后续噪声方案](r5_followup_noise_plan.md)：E2过程噪声、E3观测噪声尚未实现/运行，需新的科研授权。
- [原R0–R5任务包](../codex任务文件夹/00_开始前先读.md)、[重构任务包](../Codex_THIS_Refactor_Plan/00_START_HERE.md)

“归档”主要采用原位索引，原字节和路径保留。历史快照的原始相对链接通过归档伴随导航解释，不回写受保护记录。整理授权不等于启动R4 B、E2/E3或新的算法调参。
