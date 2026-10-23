# THIS + LASSO / Adaptive LASSO

保留 THIS 原始 Taylor 特征与高阶结构解码，比较 STLSQ、普通 LASSO 和 Ridge 初始化的 Adaptive LASSO。

当前科研结果：[R5 E1正式报告](reports/R5_report.md)。30个独立初值seed、四档响应噪声，480条记录：420 OK、60零噪声BIC NA、0 FAILED。当前不支持Adaptive LASSO普遍优于普通LASSO；20%噪声下M2三体F1低于M1。

从 [文档导航](docs/README.md) 开始。正式原始指标、诊断、配置、选择锁和图表完整保存在 [results/r5/e1_full_independent](results/r5/e1_full_independent/)。[archive/INDEX.md](archive/INDEX.md)说明历史报告、smoke、中止批次及旧README的来源和位置。

S0–S6文件治理与重构已完成：[最终审计报告](docs/refactor/S6_final_report.md)、[复现与使用](docs/REPRODUCE.md)。最终检查包括95项Python测试、179项Julia断言与六份正式统计文件的精确复算。仅清理了具有完整恢复备份的可选Python缓存，科研原始证据未删。

公共实现已按S2提取至src中的bridge、data、preprocessing、metrics及solvers模块，旧R阶段接口保留。Taylor、Adaptive核心与原论文源码保持原样；[S2接口与回归报告](docs/refactor/S2_report.md)记录实际变化和历史源码哈希保护的边界。

Python环境是项目内Conda prefix `.venv-r0`（Python3.11.17），Julia为1.9.4。激活及锁定重建说明见 [environment/r0/README.md](environment/r0/README.md)。主项目尚未初始化Git，两份原论文在upstream下各自保有独立Git仓库。

R4的M0s/M1v对照保留；R4完整B、R5 E2/E3尚未运行，仍需独立科研授权。本次文件整理不更改算法或正式实验结论。
