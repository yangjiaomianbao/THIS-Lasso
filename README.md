# THIS + LASSO / Adaptive LASSO

保留 THIS 原始 Taylor 特征与高阶结构解码，比较 STLSQ、普通 LASSO 和 Ridge 初始化的 Adaptive LASSO。

最新采样诊断（2026-11-03）：[独立局部采样与连续轨迹秩亏综合报告](results/large_scale_screening_v1/sampling_diagnostics_v1/REPORT.md) · [实现与复现命令](experiments/large_scale_screening_v1/sampling_diagnostics_v1/README.md) · [GitHub 发布范围](experiments/large_scale_screening_v1/sampling_diagnostics_v1/PUBLICATION.md)。N15/N30 各30个配对拓扑，0%/5%噪声、K16/32、五种统计筛选，27,000个主实验拟合全部通过原 Adaptive BIC。5%噪声、K16时，FWL平均三体F1为0.952/0.901。B1–B4及追加激励控制表明数值可解与结构恢复质量需要分开；全部29,700个拟合中的1,260次BIC拒绝保留为NA。原求解器门槛、神经权重和历史结果保持不变，66项测试通过。原始数据和逐模型大数组保留本地，公开指标、选参记录及文件哈希支持审查与复现。

最新预训练相关内容（2026-11-03）：[大规模预筛选模块与复现命令](experiments/large_scale_screening_v1/README.md)、[P0/P1 交付报告](results/large_scale_screening_v1/p0_p1_20261103/DELIVERY.md)。包含轻量网络与 Transformer 的共享时序编码、对称三体评分、独立训练/验证/测试拓扑、验证损失选权重及后续正式训练 CLI。当前两个架构均仅完成 **3 步训练流程验证（smoke）**，正式神经预训练与 P2/P3/P4 扩量实验尚未运行，不能据此宣称学习优势或跨规模泛化。

有效权重与训练记录位于 [neural_final](results/large_scale_screening_v1/p0_p1_20261103/neural_final/)，N30 推断验证见 [neural_n30_final](results/large_scale_screening_v1/p0_p1_20261103/neural_n30_final/summary.json)，统计对照见 [statistical_final 报告](results/large_scale_screening_v1/p0_p1_20261103/statistical_final/REPORT.md)。本批 150 个统计回归均因库秩亏或病态返回 BIC NA，结构恢复指标为 NA。开发核验批次单独保留，不与有效结果合并；完整交付文件的 SHA-256 见 [清单](results/large_scale_screening_v1/p0_p1_20261103/delivery_artifact_sha256.json)。

最新完成（2026-10-26）：[三类动力学、10种子支持集与误检诊断](results/diagnostics/this_lasso_support_3systems_10seeds_v2/REPORT.md)。保留 Kuramoto、Lotka–Volterra、SIS，15个物理节点、0%/5%/10%响应噪声；固定网络10次数据重复，加5个额外网络各10次重复。共3,240条配置：3,000 OK、240原BIC不适用、0 FAILED；472.5万个候选边的原生结构和连续分数核验通过。

[固定网络报告](results/diagnostics/this_lasso_support_3systems_10seeds_v2/A_fixed/REPORT.md) · [多网络报告](results/diagnostics/this_lasso_support_3systems_10seeds_v2/B_networks/REPORT.md) · [GitHub交付范围与文件清单](docs/publication/support_v2_20261026/README.md)。GitHub保存代码、配置、指标表、图表与哈希；约61 GiB的完整本地产物包含大型系数路径和原始矩阵，未直接放入Git历史。

早期科研结果：[R5 E1正式报告](reports/R5_report.md)。30个独立初值seed、四档响应噪声，480条记录：420 OK、60零噪声BIC NA、0 FAILED。该批次不支持Adaptive LASSO普遍优于普通LASSO；20%噪声下M2三体F1低于M1。

从 [文档导航](docs/README.md) 开始。正式原始指标、诊断、配置、选择锁和图表完整保存在 [results/r5/e1_full_independent](results/r5/e1_full_independent/)。[archive/INDEX.md](archive/INDEX.md)说明历史报告、smoke、中止批次及旧README的来源和位置。

S0–S6文件治理与重构已完成：[最终审计报告](docs/refactor/S6_final_report.md)、[复现与使用](docs/REPRODUCE.md)。最终检查包括95项Python测试、179项Julia断言与六份正式统计文件的精确复算。仅清理了具有完整恢复备份的可选Python缓存，科研原始证据未删。

公共实现已按S2提取至src中的bridge、data、preprocessing、metrics及solvers模块，旧R阶段接口保留。Taylor、Adaptive核心与原论文源码保持原样；[S2接口与回归报告](docs/refactor/S2_report.md)记录实际变化和历史源码哈希保护的边界。

Python环境是项目内Conda prefix `.venv-r0`（Python3.11.17），Julia为1.9.4。激活及锁定重建说明见 [environment/r0/README.md](environment/r0/README.md)。主项目使用Git管理，两份原论文在upstream下各自保有独立Git仓库。

R4的M0s/M1v对照保留；R4完整B、R5 E2/E3尚未运行，仍需独立科研授权。本次文件整理不更改算法或正式实验结论。
