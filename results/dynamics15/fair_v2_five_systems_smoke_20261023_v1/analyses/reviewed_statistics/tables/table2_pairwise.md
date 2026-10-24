# PILOT: pairwise reconstruction

均值 ± 样本标准差（ddof=1）；n为各指标有效重复数。粗体仅比较同系统/噪声的均值。
验证集MSE选参用于公平预测比较，不保证结构F1最优。NA不填零。

| Dynamics | Noise | Method | F1 ↑ | AP ↑ | MCC ↑ | Jaccard ↑ | OK/attempted |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Kuramoto | 0% | STLSQ_val | **1.0000 ± NA (n=1)** | **1.0000 ± NA (n=1)** | **1.0000 ± NA (n=1)** | **1.0000 ± NA (n=1)** | 1/1 |
| Kuramoto | 0% | LASSO_val | 0.9574 ± NA (n=1) | **1.0000 ± NA (n=1)** | 0.9466 ± NA (n=1) | 0.9184 ± NA (n=1) | 1/1 |
| Kuramoto | 0% | Adaptive_val | 0.5556 ± NA (n=1) | **1.0000 ± NA (n=1)** | 0.4656 ± NA (n=1) | 0.3846 ± NA (n=1) | 1/1 |
| Kuramoto | 5% | STLSQ_val | 0.9677 ± NA (n=1) | **1.0000 ± NA (n=1)** | 0.9594 ± NA (n=1) | 0.9375 ± NA (n=1) | 1/1 |
| Kuramoto | 5% | LASSO_val | **0.9890 ± NA (n=1)** | **1.0000 ± NA (n=1)** | **0.9861 ± NA (n=1)** | **0.9783 ± NA (n=1)** | 1/1 |
| Kuramoto | 5% | Adaptive_val | 0.8571 ± NA (n=1) | **1.0000 ± NA (n=1)** | 0.8257 ± NA (n=1) | 0.7500 ± NA (n=1) | 1/1 |
| Lorenz | 0% | STLSQ_val | 0.0000 ± NA (n=1) | 0.2143 ± NA (n=1) | NA (n=0) | 0.0000 ± NA (n=1) | 1/1 |
| Lorenz | 0% | LASSO_val | **1.0000 ± NA (n=1)** | **1.0000 ± NA (n=1)** | **1.0000 ± NA (n=1)** | **1.0000 ± NA (n=1)** | 1/1 |
| Lorenz | 0% | Adaptive_val | 0.9773 ± NA (n=1) | 0.9651 ± NA (n=1) | 0.9717 ± NA (n=1) | 0.9556 ± NA (n=1) | 1/1 |
| Lorenz | 5% | STLSQ_val | 0.0000 ± NA (n=1) | 0.2143 ± NA (n=1) | NA (n=0) | 0.0000 ± NA (n=1) | 1/1 |
| Lorenz | 5% | LASSO_val | **0.9773 ± NA (n=1)** | **0.9651 ± NA (n=1)** | **0.9717 ± NA (n=1)** | **0.9556 ± NA (n=1)** | 1/1 |
| Lorenz | 5% | Adaptive_val | 0.6154 ± NA (n=1) | 0.5635 ± NA (n=1) | 0.6213 ± NA (n=1) | 0.4444 ± NA (n=1) | 1/1 |
| Lotka–Volterra | 0% | STLSQ_val | **1.0000 ± NA (n=1)** | **1.0000 ± NA (n=1)** | **1.0000 ± NA (n=1)** | **1.0000 ± NA (n=1)** | 1/1 |
| Lotka–Volterra | 0% | LASSO_val | **1.0000 ± NA (n=1)** | **1.0000 ± NA (n=1)** | **1.0000 ± NA (n=1)** | **1.0000 ± NA (n=1)** | 1/1 |
| Lotka–Volterra | 0% | Adaptive_val | 0.9375 ± NA (n=1) | **1.0000 ± NA (n=1)** | 0.9221 ± NA (n=1) | 0.8824 ± NA (n=1) | 1/1 |
| Lotka–Volterra | 5% | STLSQ_val | **1.0000 ± NA (n=1)** | **1.0000 ± NA (n=1)** | **1.0000 ± NA (n=1)** | **1.0000 ± NA (n=1)** | 1/1 |
| Lotka–Volterra | 5% | LASSO_val | 0.9890 ± NA (n=1) | **1.0000 ± NA (n=1)** | 0.9861 ± NA (n=1) | 0.9783 ± NA (n=1) | 1/1 |
| Lotka–Volterra | 5% | Adaptive_val | 0.8824 ± NA (n=1) | **1.0000 ± NA (n=1)** | 0.8556 ± NA (n=1) | 0.7895 ± NA (n=1) | 1/1 |
| Stuart–Landau | 0% | STLSQ_val | 0.9535 ± NA (n=1) | 0.9302 ± NA (n=1) | 0.9432 ± NA (n=1) | 0.9111 ± NA (n=1) | 1/1 |
| Stuart–Landau | 0% | LASSO_val | **1.0000 ± NA (n=1)** | **1.0000 ± NA (n=1)** | **1.0000 ± NA (n=1)** | **1.0000 ± NA (n=1)** | 1/1 |
| Stuart–Landau | 0% | Adaptive_val | **1.0000 ± NA (n=1)** | **1.0000 ± NA (n=1)** | **1.0000 ± NA (n=1)** | **1.0000 ± NA (n=1)** | 1/1 |
| Stuart–Landau | 5% | STLSQ_val | 0.9535 ± NA (n=1) | 0.9302 ± NA (n=1) | 0.9432 ± NA (n=1) | 0.9111 ± NA (n=1) | 1/1 |
| Stuart–Landau | 5% | LASSO_val | **0.9890 ± NA (n=1)** | **1.0000 ± NA (n=1)** | **0.9861 ± NA (n=1)** | **0.9783 ± NA (n=1)** | 1/1 |
| Stuart–Landau | 5% | Adaptive_val | **0.9890 ± NA (n=1)** | **1.0000 ± NA (n=1)** | **0.9861 ± NA (n=1)** | **0.9783 ± NA (n=1)** | 1/1 |
| SIS | 0% | STLSQ_val | **1.0000 ± NA (n=1)** | **1.0000 ± NA (n=1)** | **1.0000 ± NA (n=1)** | **1.0000 ± NA (n=1)** | 1/1 |
| SIS | 0% | LASSO_val | **1.0000 ± NA (n=1)** | **1.0000 ± NA (n=1)** | **1.0000 ± NA (n=1)** | **1.0000 ± NA (n=1)** | 1/1 |
| SIS | 0% | Adaptive_val | 0.9890 ± NA (n=1) | **1.0000 ± NA (n=1)** | 0.9861 ± NA (n=1) | 0.9783 ± NA (n=1) | 1/1 |
| SIS | 5% | STLSQ_val | **1.0000 ± NA (n=1)** | **1.0000 ± NA (n=1)** | **1.0000 ± NA (n=1)** | **1.0000 ± NA (n=1)** | 1/1 |
| SIS | 5% | LASSO_val | 0.9890 ± NA (n=1) | **1.0000 ± NA (n=1)** | 0.9861 ± NA (n=1) | 0.9783 ± NA (n=1) | 1/1 |
| SIS | 5% | Adaptive_val | 0.8911 ± NA (n=1) | **1.0000 ± NA (n=1)** | 0.8660 ± NA (n=1) | 0.8036 ± NA (n=1) | 1/1 |
