# V2 正式实验与 GitHub 更新（2026-10-24）

本次发布新增15节点公平对照实验代码、锁定配置、测试日志及正式结果表。既有代码、原论文源码和旧实验结果保持原样。项目根README及V2初次交付说明中的阶段状态属于此前记录；当前正式实验已完成，以本页及正式运行的completion.json为准。

## 当前正式结果

- 10个网络，每个网络3个数据重复；5种动力学、2种噪声、3种方法，共900模型。
- 889 OK、0模型级NA、11 FAILED。失败全部发生于无噪声Lorenz的Adaptive，涉及14个目标方程的验证MSE一致性断言；该配置有效数19/30。失败没有填零或选择性重跑。
- 三体指标：F1、AP、MCC、Jaccard；二体另行统计。56条有效模型的二体MCC记录未定义，保留NA。
- 正式实验及S7分析于北京时间2026-10-24 08:23完成，实验耗时约10.15小时。
- 140项Python测试、141项新增Julia断言、40项原方程断言通过。正式实验后12,399个历史受保护文件哈希检查通过。
- 三方法共享标准化、自由截距及验证MSE选参；**预测MSE最优不保证结构F1最优**。没有利用测试结构真值调参。

## 阅读入口

- [正式实验概览](../results/dynamics15/fair_comparison_v2_full_run01/REPORT.md)
- [三体四指标表](../results/dynamics15/fair_comparison_v2_full_run01/analyses/formal_statistics01/tables/table1_threebody.md)
- [二体四指标表](../results/dynamics15/fair_comparison_v2_full_run01/analyses/formal_statistics01/tables/table2_pairwise.md)
- [配对差异、方差、稳定性与图表](../results/dynamics15/fair_comparison_v2_full_run01/analyses/formal_statistics01/REPORT.md)
- [逐实例原始指标](../results/dynamics15/fair_comparison_v2_full_run01/trials.csv)
- [失败诊断索引](../results/dynamics15/fair_comparison_v2_full_run01/failure_reasons.csv)（索引中的14份节点JSON一并上传）
- [实现说明](experiments/fair_comparison_v2/S2_S4_implementation.md)
- [运行与复现命令](experiments/fair_comparison_v2/S6_S7_runbook.md)（其中“等待确认”是启动前记录；本次正式运行已经获得授权并完成）
- [运行后历史保护审计](experiments/fair_comparison_v2/full_run01_startup/post_run_integrity.json)

## 上传范围与本地保留内容

上传V2源代码、测试、锁定网络/种子配置、审计和运行日志、正式与小试的轻量报告、原始trials、模型清单、选择锁、资源统计、配对统计、表格、图片及源码快照。文件级SHA256见[GITHUB_UPDATE_V2_MANIFEST.json](GITHUB_UPDATE_V2_MANIFEST.json)，该清单不包含自身。

以下内容仍保留在服务器原路径，本次不上传、不移动、不删除：

- Python/Julia本地环境、Julia depot及绘图缓存；
- 各实例的数据矩阵、特征矩阵、逐节点成功求解系数与完整诊断；
- `edge_scores.csv`（正式运行约116 MB），`selection_details.csv`（约62 MB），以及小试对应的上述文件；
- 指向本地逐模型数据的`native_decode_request.toml`与临时PID文件。

GitHub副本可以直接查看和使用已发布的结果表及统计，但不包含完整实验存档。`fair_v2_verify.py`、`fair_v2_analyse.py`的完整重新核验依赖连续边得分、选择路径及相关本地文件；仅克隆本仓库不能直接重新执行这部分核验。需要恢复服务器存档，或按锁定配置在新的输出标签下重新运行。环境和原论文依赖按原项目复现说明准备。

模型清单和完整性清单按原样上传，部分条目指向未上传的大文件；它们记录完整本地实验的来源，不表示GitHub副本包含每一个条目。
