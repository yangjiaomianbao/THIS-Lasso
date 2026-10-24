# PILOT: triple reconstruction

均值 ± 样本标准差（ddof=1）；n为各指标有效重复数。粗体仅比较同系统/噪声的均值。
验证集MSE选参用于公平预测比较，不保证结构F1最优。NA不填零。

| Dynamics | Noise | Method | F1 ↑ | AP ↑ | MCC ↑ | Jaccard ↑ | OK/attempted |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Kuramoto | 0% | STLSQ_val | **1.0000 ± NA (n=1)** | **1.0000 ± NA (n=1)** | **1.0000 ± NA (n=1)** | **1.0000 ± NA (n=1)** | 1/1 |
| Kuramoto | 0% | LASSO_val | 0.0598 ± NA (n=1) | **1.0000 ± NA (n=1)** | 0.1415 ± NA (n=1) | 0.0308 ± NA (n=1) | 1/1 |
| Kuramoto | 0% | Adaptive_val | 0.0654 ± NA (n=1) | **1.0000 ± NA (n=1)** | 0.1518 ± NA (n=1) | 0.0338 ± NA (n=1) | 1/1 |
| Kuramoto | 5% | STLSQ_val | **0.8889 ± NA (n=1)** | 0.8022 ± NA (n=1) | **0.8934 ± NA (n=1)** | **0.8000 ± NA (n=1)** | 1/1 |
| Kuramoto | 5% | LASSO_val | 0.1302 ± NA (n=1) | **0.8653 ± NA (n=1)** | 0.2345 ± NA (n=1) | 0.0697 ± NA (n=1) | 1/1 |
| Kuramoto | 5% | Adaptive_val | 0.3117 ± NA (n=1) | 0.7826 ± NA (n=1) | 0.3820 ± NA (n=1) | 0.1846 ± NA (n=1) | 1/1 |
| SIS | 0% | STLSQ_val | **1.0000 ± NA (n=1)** | **1.0000 ± NA (n=1)** | **1.0000 ± NA (n=1)** | **1.0000 ± NA (n=1)** | 1/1 |
| SIS | 0% | LASSO_val | **1.0000 ± NA (n=1)** | **1.0000 ± NA (n=1)** | **1.0000 ± NA (n=1)** | **1.0000 ± NA (n=1)** | 1/1 |
| SIS | 0% | Adaptive_val | 0.8824 ± NA (n=1) | **1.0000 ± NA (n=1)** | 0.8872 ± NA (n=1) | 0.7895 ± NA (n=1) | 1/1 |
| SIS | 5% | STLSQ_val | **0.8000 ± NA (n=1)** | 0.6703 ± NA (n=1) | **0.8150 ± NA (n=1)** | **0.6667 ± NA (n=1)** | 1/1 |
| SIS | 5% | LASSO_val | 0.3093 ± NA (n=1) | **0.9958 ± NA (n=1)** | 0.4170 ± NA (n=1) | 0.1829 ± NA (n=1) | 1/1 |
| SIS | 5% | Adaptive_val | 0.3934 ± NA (n=1) | 0.7800 ± NA (n=1) | 0.4476 ± NA (n=1) | 0.2449 ± NA (n=1) | 1/1 |
