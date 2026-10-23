# 15节点、五类高阶动力学

已完成120个模型：96 OK、24 BIC NA、0 FAILED，独立核验PASS。

五类为Kuramoto、Lorenz、Lotka–Volterra、Stuart–Landau、SIS，各15物理节点、45条成对边、15条三体边。完整三阶库，每seed1024/256/256独立初值，0%/5%响应噪声，2seed。

无噪声普通验证LASSO在四个多项式系统上二体/三体F1均为1；Adaptive并非普遍更好。部分组合AP=1但固定阈值F1低，主要表现为假阳性较多；Lorenz含噪声三体排序仍弱。

[完整报告、指标及图表](../results/dynamics15/five_systems_certified_20261022/REPORT.md)。

[方程和文献来源](../docs/dynamics15_five_systems_protocol.md)、[配置](../configs/dynamics15_five_systems_certified_v1_1.toml)。
