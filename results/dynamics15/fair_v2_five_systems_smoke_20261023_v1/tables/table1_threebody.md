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
| Lorenz | 0% | STLSQ_val | 0.0714 ± NA (n=1) | 0.0436 ± NA (n=1) | 0.0620 ± NA (n=1) | 0.0370 ± NA (n=1) | 1/1 |
| Lorenz | 0% | LASSO_val | **1.0000 ± NA (n=1)** | **1.0000 ± NA (n=1)** | **1.0000 ± NA (n=1)** | **1.0000 ± NA (n=1)** | 1/1 |
| Lorenz | 0% | Adaptive_val | 0.0650 ± NA (n=1) | 0.9341 ± NA (n=1) | 0.1439 ± NA (n=1) | 0.0336 ± NA (n=1) | 1/1 |
| Lorenz | 5% | STLSQ_val | **0.0541 ± NA (n=1)** | **0.0142 ± NA (n=1)** | **0.0423 ± NA (n=1)** | **0.0278 ± NA (n=1)** | 1/1 |
| Lorenz | 5% | LASSO_val | 0.0169 ± NA (n=1) | 0.0101 ± NA (n=1) | -0.0120 ± NA (n=1) | 0.0085 ± NA (n=1) | 1/1 |
| Lorenz | 5% | Adaptive_val | 0.0120 ± NA (n=1) | 0.0116 ± NA (n=1) | -0.0148 ± NA (n=1) | 0.0061 ± NA (n=1) | 1/1 |
| Lotka–Volterra | 0% | STLSQ_val | **1.0000 ± NA (n=1)** | **1.0000 ± NA (n=1)** | **1.0000 ± NA (n=1)** | **1.0000 ± NA (n=1)** | 1/1 |
| Lotka–Volterra | 0% | LASSO_val | **1.0000 ± NA (n=1)** | **1.0000 ± NA (n=1)** | **1.0000 ± NA (n=1)** | **1.0000 ± NA (n=1)** | 1/1 |
| Lotka–Volterra | 0% | Adaptive_val | 0.9091 ± NA (n=1) | **1.0000 ± NA (n=1)** | 0.9119 ± NA (n=1) | 0.8333 ± NA (n=1) | 1/1 |
| Lotka–Volterra | 5% | STLSQ_val | **0.9286 ± NA (n=1)** | 0.8681 ± NA (n=1) | **0.9303 ± NA (n=1)** | **0.8667 ± NA (n=1)** | 1/1 |
| Lotka–Volterra | 5% | LASSO_val | 0.2703 ± NA (n=1) | **1.0000 ± NA (n=1)** | 0.3832 ± NA (n=1) | 0.1562 ± NA (n=1) | 1/1 |
| Lotka–Volterra | 5% | Adaptive_val | 0.5172 ± NA (n=1) | 0.9922 ± NA (n=1) | 0.5845 ± NA (n=1) | 0.3488 ± NA (n=1) | 1/1 |
| Stuart–Landau | 0% | STLSQ_val | 0.6923 ± NA (n=1) | 0.6044 ± NA (n=1) | 0.6979 ± NA (n=1) | 0.5294 ± NA (n=1) | 1/1 |
| Stuart–Landau | 0% | LASSO_val | **1.0000 ± NA (n=1)** | **1.0000 ± NA (n=1)** | **1.0000 ± NA (n=1)** | **1.0000 ± NA (n=1)** | 1/1 |
| Stuart–Landau | 0% | Adaptive_val | 0.1538 ± NA (n=1) | **1.0000 ± NA (n=1)** | 0.2705 ± NA (n=1) | 0.0833 ± NA (n=1) | 1/1 |
| Stuart–Landau | 5% | STLSQ_val | **0.5294 ± NA (n=1)** | 0.6044 ± NA (n=1) | **0.5273 ± NA (n=1)** | **0.3600 ± NA (n=1)** | 1/1 |
| Stuart–Landau | 5% | LASSO_val | 0.0647 ± NA (n=1) | **1.0000 ± NA (n=1)** | 0.1506 ± NA (n=1) | 0.0334 ± NA (n=1) | 1/1 |
| Stuart–Landau | 5% | Adaptive_val | 0.1765 ± NA (n=1) | **1.0000 ± NA (n=1)** | 0.2945 ± NA (n=1) | 0.0968 ± NA (n=1) | 1/1 |
| SIS | 0% | STLSQ_val | **1.0000 ± NA (n=1)** | **1.0000 ± NA (n=1)** | **1.0000 ± NA (n=1)** | **1.0000 ± NA (n=1)** | 1/1 |
| SIS | 0% | LASSO_val | **1.0000 ± NA (n=1)** | **1.0000 ± NA (n=1)** | **1.0000 ± NA (n=1)** | **1.0000 ± NA (n=1)** | 1/1 |
| SIS | 0% | Adaptive_val | 0.8824 ± NA (n=1) | **1.0000 ± NA (n=1)** | 0.8872 ± NA (n=1) | 0.7895 ± NA (n=1) | 1/1 |
| SIS | 5% | STLSQ_val | **0.8000 ± NA (n=1)** | 0.6703 ± NA (n=1) | **0.8150 ± NA (n=1)** | **0.6667 ± NA (n=1)** | 1/1 |
| SIS | 5% | LASSO_val | 0.3093 ± NA (n=1) | **0.9958 ± NA (n=1)** | 0.4170 ± NA (n=1) | 0.1829 ± NA (n=1) | 1/1 |
| SIS | 5% | Adaptive_val | 0.3934 ± NA (n=1) | 0.7800 ± NA (n=1) | 0.4476 ± NA (n=1) | 0.2449 ± NA (n=1) | 1/1 |
