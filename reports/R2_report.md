# R2 执行报告

日期：2026-10-16（Asia/Shanghai）。用户明确授权清理 R1 后执行 R2；本阶段只实现独立 Python Adaptive LASSO。**30 项单元测试全部通过，固定 R1 数据演示通过，退出码均为 0；等待人工验收，未执行 R3。**

## 本次修改文件

| 范围 | 文件 | 作用 |
| --- | --- | --- |
| R2 新增 | `src/solvers/adaptive_lasso.py` | 训练集 Ridge/OLS 初始化、自适应权重、BIC/验证集/固定 alpha、目标与 KKT 诊断 |
| R2 新增 | `src/solvers/__init__.py` | 公开单节点接口、配置和异常 |
| R2 新增 | `tests/test_r2_adaptive_lasso.py` | 30 项数值与边界测试 |
| R2 新增 | `scripts/r2_solver_demo.py` | 只读 R1 修订矩阵，独立验证演示，不运行 Julia 或结构解码 |
| R2 新增 | `reports/R2_report.md`、`reports/R2_learning.md` | 完整输出、配置、结论及逐行学习说明 |
| 用户授权的 R1 清理 | `results/r1/history.log` | 合并 7 份旧文件，保留原文件名、SHA-256 和完整字节 |
| 用户授权的清理维护 | `scripts/r1_run.jl`、`reports/R1_report.md`、`docs/r1_data_description.md`、`results/r1/artifact_manifest.toml`、`README.md` | 移除旧 diagnostics 入口，修正链接、阶段状态和清单 |

没有增加依赖，没有改动 `data/r1/`、上游源码或人工进度勾选表。R2 日志直接嵌在本报告，避免扩展任务文件限定的输出边界。

## R1 清理结果与保留依据

原独立文件已删除：`scripts/r1_diagnostics.jl`，`results/r1/coefficient_diagnostics.csv`，`results/r1/diagnostics.toml`，以及旧 `results/r1/logs/` 内的 `test.log`、`baseline.log`、`replay.log`、`diagnostics.log`。它们是已用完的单次诊断和旧运行记录，已按原始字节合并到 [history.log](../results/r1/history.log)，因此净减少 6 个文件，并移除旧空日志目录。

保留 R1 两版固定数据、旧失败基线、修订版默认/选参结果、协议、完整候选表、锁定记录、验收结果、重放脚本、测试和当前修订日志。它们支撑复现、后续集成和公平比较，不能以“已完成 R1”为由删除。最终 [R1 产物清单](../results/r1/artifact_manifest.toml) 记录清理后的哈希和前版清单哈希。

## 实现与数值核验

接口：`fit_adaptive_lasso(X:[K,P], y:[K], config, ...) -> (coef:[P], diagnostics)`。预测为 `X @ coef + diagnostics['intercept']`。默认训练集标准化 Ridge/SVD，ρ=1、γ=1、ε=10^-6；可切换 OLS。权重由原始尺度初值定义，`w=(abs(beta_init)+eps)^(-gamma)`，`a=1/w`。

最终原始尺度目标为 `SSE/(2K)+alpha*sum(w*abs(beta))`，单个非零常数列不惩罚。训练标准化坐标 Z 的缩放显式补回尺度 s：`U=Z*(s*a)=(X-mean)*a`；在 U 上求标准 LASSO 后 `beta=a*theta`。常数列已承载偏置时，额外 intercept 返回 0，输出列数与顺序保持不变。学习笔记给出 Su 源码对照、目标推导及核心 28 行逐行说明。

默认 BIC 使用 `LassoLarsIC`，显式检查 K>P、残差自由度、满秩、条件数和可估计噪声方差；失败抛 `BICNotApplicable`，不自动改策略。验证模式要求训练/验证轨迹 ID 不交叉，以训练确定的 alpha_max 构造固定比例网格，用验证 MSE 选参；容差内取较大 alpha。固定/验证模式使用确定性的坐标下降。记录全部候选和警告；选中解收敛失败或 KKT 超标则报错。

30 项测试覆盖：已知稀疏含噪系数、合法 BIC、K≤P/重复/病态/无噪声 BIC 拒绝、显式高维与重复特征验证、强相关、微小初值、权重溢出、Ridge/OLS 切换、固定种子、无惩罚常数项、零列、常数模型、多常数拒绝、无截距模式、训练/验证轨迹隔离、训练变换与权重固定、输入不修改、原尺度权重、预测还原、加权目标等价和独立 KKT 检查。没有依靠“恢复几条边”代替优化正确性核验。

## 固定 R1 数据演示的实际结果

读取 `data/r1/revision/` 的既定矩阵，选 Julia 节点 7 的回归问题。训练 512 条轨迹/1536 样本，验证 128 条轨迹/384 样本，P=165，首列常数。数据文件含原定测试列，但拟合和选参只传入训练/验证切片，未计算测试指标；真值文件仅参与字节哈希核对，不用于拟合或选参。

| 检查 | 实测 |
| --- | --- |
| 默认 BIC | 预期拒绝：满模型残差方差低于可靠性下限 |
| 显式验证选择 alpha | 3.4631161526439488×10^-9，共 12 个候选 |
| 验证 MSE | 4.2027920581945406×10^-15 |
| 原加权目标 / 变换后目标 | 1.0463570206483078×10^-8 / 1.0463570206483077×10^-8 |
| 归一化 KKT 最大违背量 | 2.6888142607758932×10^-15 |
| 原尺度预测还原最大差 | 4.440892098500626×10^-16 |
| 警告 | 无 |

下方完整输出保存版本、所有配置、全部 alpha 候选、轨迹数、系数和两版 R1 数据哈希。展示系数采用 `abs(coef)>10^-9` 的打印过滤，包括约 -2.67×10^-9 的常数偏置；这不是结构推断阈值或已解码的网络边。

## 环境、限制及人工验收

使用已锁定的独立环境 `.venv-r0`：Python 3.11.17、NumPy 1.26.4、scikit-learn 1.6.1、SciPy 1.11.4；BLAS 单线程，禁写 pycache。源码与依赖锁定沿用 R0，本次没有环境变更。实际命令、时间、输出、退出码见下方原始记录；最后的文件审计另核对清理归档、固定数据和上游哈希。

无噪声且 alpha>0 时有收缩偏差，不能要求所有系数逐位等于真值。重复特征时系数本身未必唯一；验证模式可运行并不证明可识别性。标准化改善初始化，不能完全消除极端自适应权重导致的病态；当前用显式数值检查或异常报告，而不承诺任意输入都成功。BIC 采用实用适用性门槛，并非其统计理论在全部自适应情形都成立的证明。

当前证据支持求解器实现的数值正确性；未评价网络结构 F1/AUPRC、弱耦合优势、含噪鲁棒性或跨种子平均效果。没有接入 THIS Julia 解码，没有运行大规模网络比较。

机器完成状态：Su 对照、Ridge/OLS、自适应变换、原尺度还原、目标等价/KKT/边界测试、BIC 与验证策略、参数日志和学习笔记均已完成。人工清单保持未勾选。可在激活 `environment/r0/activate.sh` 后复跑下方两条 Python 命令进行验收。

R2 完成；等待人工验收；未执行下一阶段。

## 真实执行记录

### R2 单元测试（首次）

开始：2026-10-16T22:37:38.399428+08:00

命令：`/home/yg/THIS-Lasso/.venv-r0/bin/python -m unittest discover -s tests -p 'test_r2_*.py' -v`

环境：Python 3.11.17，OPENBLAS_NUM_THREADS=1，PYTHONDONTWRITEBYTECODE=1。

```text
test_bic_does_not_silently_fall_back (test_r2_adaptive_lasso.AdaptiveLassoTests.test_bic_does_not_silently_fall_back) ... ok
test_bic_on_valid_noisy_full_rank_data (test_r2_adaptive_lasso.AdaptiveLassoTests.test_bic_on_valid_noisy_full_rank_data) ... ok
test_bic_rejects_K_le_P (test_r2_adaptive_lasso.AdaptiveLassoTests.test_bic_rejects_K_le_P) ... ok
test_bic_rejects_duplicate_features (test_r2_adaptive_lasso.AdaptiveLassoTests.test_bic_rejects_duplicate_features) ... ok
test_bic_rejects_ill_conditioned_full_model (test_r2_adaptive_lasso.AdaptiveLassoTests.test_bic_rejects_ill_conditioned_full_model) ... ok
test_bic_rejects_unreliable_noise_variance (test_r2_adaptive_lasso.AdaptiveLassoTests.test_bic_rejects_unreliable_noise_variance) ... ok
test_constant_column_prediction_and_no_duplicate_intercept (test_r2_adaptive_lasso.AdaptiveLassoTests.test_constant_column_prediction_and_no_duplicate_intercept) ... ok
test_constant_only_problem_has_no_model_selection (test_r2_adaptive_lasso.AdaptiveLassoTests.test_constant_only_problem_has_no_model_selection) ... ok
test_duplicate_features_validation_is_explicit_and_valid (test_r2_adaptive_lasso.AdaptiveLassoTests.test_duplicate_features_validation_is_explicit_and_valid) ... ok
test_explicit_weight_formula_in_original_units (test_r2_adaptive_lasso.AdaptiveLassoTests.test_explicit_weight_formula_in_original_units) ... ok
test_extremely_small_initial_coefficients_remain_finite (test_r2_adaptive_lasso.AdaptiveLassoTests.test_extremely_small_initial_coefficients_remain_finite) ... ok
test_fixed_seed_and_cyclic_solver_are_repeatable (test_r2_adaptive_lasso.AdaptiveLassoTests.test_fixed_seed_and_cyclic_solver_are_repeatable) ... ok
test_free_constant_column_even_without_additional_intercept (test_r2_adaptive_lasso.AdaptiveLassoTests.test_free_constant_column_even_without_additional_intercept) ... ok
test_high_dimensional_validation_is_supported (test_r2_adaptive_lasso.AdaptiveLassoTests.test_high_dimensional_validation_is_supported) ... ok
test_highly_correlated_features_fixed_objective (test_r2_adaptive_lasso.AdaptiveLassoTests.test_highly_correlated_features_fixed_objective) ... ok
test_independent_kkt_check (test_r2_adaptive_lasso.AdaptiveLassoTests.test_independent_kkt_check) ... ok
test_inputs_are_not_mutated (test_r2_adaptive_lasso.AdaptiveLassoTests.test_inputs_are_not_mutated) ... ok
test_invalid_inputs_and_configuration (test_r2_adaptive_lasso.AdaptiveLassoTests.test_invalid_inputs_and_configuration) ... ok
test_known_sparse_noisy_coefficients (test_r2_adaptive_lasso.AdaptiveLassoTests.test_known_sparse_noisy_coefficients) ... ok
test_multiple_nonzero_constant_columns_rejected (test_r2_adaptive_lasso.AdaptiveLassoTests.test_multiple_nonzero_constant_columns_rejected) ... ok
test_no_bias_when_no_constant_and_intercept_disabled (test_r2_adaptive_lasso.AdaptiveLassoTests.test_no_bias_when_no_constant_and_intercept_disabled) ... ok
test_ols_initialization_switch (test_r2_adaptive_lasso.AdaptiveLassoTests.test_ols_initialization_switch) ... ok
test_scale_and_offset_prediction_restoration (test_r2_adaptive_lasso.AdaptiveLassoTests.test_scale_and_offset_prediction_restoration) ... ok
test_trajectory_leakage_is_rejected (test_r2_adaptive_lasso.AdaptiveLassoTests.test_trajectory_leakage_is_rejected) ... ok
test_unrepresentable_weights_are_rejected (test_r2_adaptive_lasso.AdaptiveLassoTests.test_unrepresentable_weights_are_rejected) ... ok
test_validation_cannot_change_training_preprocessing_or_weights (test_r2_adaptive_lasso.AdaptiveLassoTests.test_validation_cannot_change_training_preprocessing_or_weights) ... ok
test_validation_rejects_changed_constant_column (test_r2_adaptive_lasso.AdaptiveLassoTests.test_validation_rejects_changed_constant_column) ... ok
test_validation_requires_group_ids (test_r2_adaptive_lasso.AdaptiveLassoTests.test_validation_requires_group_ids) ... ok
test_weighted_and_transformed_objectives_are_equal (test_r2_adaptive_lasso.AdaptiveLassoTests.test_weighted_and_transformed_objectives_are_equal) ... ok
test_zero_column_retained_as_zero (test_r2_adaptive_lasso.AdaptiveLassoTests.test_zero_column_retained_as_zero) ... ok

----------------------------------------------------------------------
Ran 30 tests in 0.050s

OK
```

退出码：0

### 固定 R1 数据上的 R2 独立演示

开始：2026-10-16T22:39:31.146709+08:00

命令：`/home/yg/THIS-Lasso/.venv-r0/bin/python scripts/r2_solver_demo.py`

环境：`PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`

```text
EXPECTED_BIC_REJECTION: Full-model noise variance is zero or below its reliability floor. Explicitly use selection="validation" with independent trajectory groups.
{
  "algorithm_version": "r2-adaptive-lasso-1.0.0",
  "sklearn_version": "1.6.1",
  "config": {
    "initialization": "ridge",
    "ridge_rho": 1.0,
    "gamma": 1.0,
    "eps": 1e-06,
    "standardize": true,
    "fit_intercept": true,
    "selection": "validation",
    "alpha": null,
    "validation_alphas": null,
    "alpha_fractions": [
      1.0,
      0.3,
      0.1,
      0.03,
      0.01,
      0.003,
      0.001,
      0.0003,
      0.0001,
      1e-05,
      1e-06,
      1e-08
    ],
    "validation_tolerance": 1e-12,
    "constant_tol": 0.0,
    "bic_noise_floor": 1e-12,
    "bic_condition_limit": 100000000.0,
    "max_iter": 100000,
    "tol": 1e-10,
    "kkt_tolerance": 1e-06
  },
  "node_julia_index": 7,
  "train_shape": [
    1536,
    165
  ],
  "validation_shape": [
    384,
    165
  ],
  "train_trajectories": 512,
  "validation_trajectories": 128,
  "alpha": 3.4631161526439488e-09,
  "intercept": 0.0,
  "constant_index": 0,
  "nonzero_coefficients": [
    {
      "python_feature_index": 0,
      "julia_feature_index": 1,
      "coefficient": -2.665551135719313e-09
    },
    {
      "python_feature_index": 137,
      "julia_feature_index": 138,
      "coefficient": 0.19999982683956946
    },
    {
      "python_feature_index": 146,
      "julia_feature_index": 147,
      "coefficient": 0.11999990545252565
    },
    {
      "python_feature_index": 156,
      "julia_feature_index": 157,
      "coefficient": -1.099999995403465
    }
  ],
  "validation_MSE": 4.2027920581945406e-15,
  "objective": 1.0463570206483078e-08,
  "transformed_objective": 1.0463570206483077e-08,
  "kkt_scaled_violation": 2.6888142607758932e-15,
  "prediction_restore_max_abs": 4.440892098500626e-16,
  "weight_min": 0.920351728419369,
  "weight_max": 829267.3433210526,
  "warnings": [],
  "validation_path": [
    {
      "alpha": 0.34631161526439486,
      "MSE": 0.39202926501695634,
      "nonzero": 0,
      "n_iter": 1,
      "dual_gap": 0.0,
      "warnings": [],
      "convergence_warning": false
    },
    {
      "alpha": 0.10389348457931845,
      "MSE": 0.041083748174901395,
      "nonzero": 1,
      "n_iter": 2,
      "dual_gap": 5.551115123125783e-17,
      "warnings": [],
      "convergence_warning": false
    },
    {
      "alpha": 0.03463116152643949,
      "MSE": 0.010113473158130453,
      "nonzero": 1,
      "n_iter": 2,
      "dual_gap": 0.0,
      "warnings": [],
      "convergence_warning": false
    },
    {
      "alpha": 0.010389348457931845,
      "MSE": 0.006550363530761333,
      "nonzero": 1,
      "n_iter": 2,
      "dual_gap": 2.3129646346357427e-18,
      "warnings": [],
      "convergence_warning": false
    },
    {
      "alpha": 0.0034631161526439487,
      "MSE": 0.004202792159295628,
      "nonzero": 3,
      "n_iter": 7,
      "dual_gap": 1.0523989087592629e-16,
      "warnings": [],
      "convergence_warning": false
    },
    {
      "alpha": 0.0010389348457931847,
      "MSE": 0.0003782512943366228,
      "nonzero": 3,
      "n_iter": 7,
      "dual_gap": 9.880695798584564e-17,
      "warnings": [],
      "convergence_warning": false
    },
    {
      "alpha": 0.00034631161526439485,
      "MSE": 4.202792159295701e-05,
      "nonzero": 3,
      "n_iter": 8,
      "dual_gap": 1.3305871161828353e-16,
      "warnings": [],
      "convergence_warning": false
    },
    {
      "alpha": 0.00010389348457931845,
      "MSE": 3.7825129433663577e-06,
      "nonzero": 3,
      "n_iter": 10,
      "dual_gap": 2.3714663768594397e-17,
      "warnings": [],
      "convergence_warning": false
    },
    {
      "alpha": 3.463116152643949e-05,
      "MSE": 4.20279215928257e-07,
      "nonzero": 3,
      "n_iter": 13,
      "dual_gap": 1.19129890075397e-16,
      "warnings": [],
      "convergence_warning": false
    },
    {
      "alpha": 3.463116152643949e-06,
      "MSE": 4.202792158352932e-09,
      "nonzero": 3,
      "n_iter": 20,
      "dual_gap": 9.669567487380135e-16,
      "warnings": [],
      "convergence_warning": false
    },
    {
      "alpha": 3.4631161526439484e-07,
      "MSE": 4.202792158225547e-11,
      "nonzero": 3,
      "n_iter": 36,
      "dual_gap": 1.2665957370911662e-16,
      "warnings": [],
      "convergence_warning": false
    },
    {
      "alpha": 3.4631161526439488e-09,
      "MSE": 4.202792056121321e-15,
      "nonzero": 3,
      "n_iter": 69,
      "dual_gap": 2.7902298956855047e-16,
      "warnings": [],
      "convergence_warning": false
    }
  ],
  "data_sha256": {
    "data/r1/metadata.toml": "a147db7c2df4715df652eade5c8f8d5aad7c552170024231a7b99fc11bf4f345",
    "data/r1/true_coefficients.csv": "6ee5228a11e3507a7c1133bd386ec41c4acb760b0f78e01d24fcce52fd66fc19",
    "data/r1/initial_conditions.csv": "7eba13126a1bdb17abab096bc2d32b49c8ff59aa336a4759579fa4eccd6af16b",
    "data/r1/derivatives.csv": "b445db3939401d93c2619937e6f94ad3fc1b3f5a7c1c8168592578a10a2f7073",
    "data/r1/monomial_indices.csv": "c039ccfc464622c630602a426fc51dabe9cd81df367c86ffe3047772d8e3bbe2",
    "data/r1/true_structure.csv": "94fd84be4a7e33f36053385b37d99b76e35ad60f72db99ce89523250ab089bcd",
    "data/r1/sample_manifest.csv": "fda5183b4af3e2473c944927f74e569bf08a6878f9c2c87bfddddf6812cc3ee2",
    "data/r1/states.csv": "a709cc39c674246c3a8a7b445e56fcc513a16dd7fecf25a5ce2cf7d7f11ecb2e",
    "data/r1/theta.csv": "dcbac373c363df6edf618993b60c15f4a0cc44930b3d83893be18d55f3480a1f",
    "data/r1/revision/metadata.toml": "a5fdaa823d54981aaa5bd202714ad32386b6c534e03d6fb275721b4fd7b40d85",
    "data/r1/revision/true_coefficients.csv": "6ee5228a11e3507a7c1133bd386ec41c4acb760b0f78e01d24fcce52fd66fc19",
    "data/r1/revision/initial_conditions.csv": "1fc1368b8a7d0e81aa4abb3e7d545dc5e9fbece494bd8bb65322181ac2d6da63",
    "data/r1/revision/derivatives.csv": "81f74de439d1c5147d317c7fd2b7a3cd859692535627d9bb298591d5d8d99fff",
    "data/r1/revision/monomial_indices.csv": "c039ccfc464622c630602a426fc51dabe9cd81df367c86ffe3047772d8e3bbe2",
    "data/r1/revision/true_structure.csv": "94fd84be4a7e33f36053385b37d99b76e35ad60f72db99ce89523250ab089bcd",
    "data/r1/revision/sample_manifest.csv": "5cba93c2ac43fbe91b9d9d041de69042411e3de6754904ad4b5d4858acb2b9ce",
    "data/r1/revision/states.csv": "1881b7e6999f45d09e961117062f53866ba56e658989e8557c8112d88bbd7228",
    "data/r1/revision/theta.csv": "7ceacdefe5e09bd0062deb303bb6d34b3807960e00914b7aea3e988ca48195e6"
  }
}
R1_DATA_UNCHANGED; NO_TEST_EVALUATION; NO_JULIA_INVOCATION_OR_STRUCTURE_DECODING
R2_SOLVER_DEMO_PASS
```

退出码：0

### R1 清理后日志入口语法检查

开始：2026-10-16T22:42:57.787725+08:00

实际命令：

```bash
/home/yg/THIS-Lasso/environment/r0/julia.sh --startup-file=no -e 'Meta.parseall(read("scripts/r1_run.jl", String)); println("R1_LOGGER_PARSE_PASS")'
```

环境：`PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`

```text
R1_LOGGER_PARSE_PASS
```

退出码：0

### 清理归档、固定数据、上游源码与文档最终审计

开始：2026-10-16T22:42:57.946515+08:00

实际命令：

```bash
/home/yg/THIS-Lasso/.venv-r0/bin/python -c '
from pathlib import Path
import datetime, hashlib, json, re, subprocess, tomllib
r=Path.cwd()
hash_file=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=r/'"'"'results/r1/artifact_manifest.toml'"'"'
old_bytes=manifest.read_bytes();old=tomllib.loads(old_bytes.decode())
archive=(r/'"'"'results/r1/history.log'"'"').read_bytes()
pattern=rb'"'"'===== BEGIN ([^\n]+) SHA256=([0-9a-f]{64}) =====\n(.*?)\n===== END \1 ====='"'"'
parts=re.findall(pattern,archive,re.S)
assert len(parts)==7
removed=set()
for name,expected,body in parts:
    name=name.decode();expected=expected.decode()
    assert hashlib.sha256(body).hexdigest()==expected==old['"'"'file_sha256'"'"'][name], name
    assert not (r/name).exists(),name
    removed.add(name)
allowed={'"'"'scripts/r1_run.jl'"'"','"'"'reports/R1_report.md'"'"','"'"'docs/r1_data_description.md'"'"'}
for name,expected in old['"'"'file_sha256'"'"'].items():
    if name not in removed|allowed:
        assert hash_file(r/name)==expected,name
print('"'"'R1_ARCHIVE: 7 original byte streams and SHA256 values verified; old files absent'"'"')
print('"'"'R1_PRESERVED: all 18 data files and all retained results/source match previous manifest'"'"')
locks=json.loads((r/'"'"'environment/r0/upstream_lock.json'"'"').read_text())
count=0
for name,lock in locks.items():
    repo=r/'"'"'upstream'"'"'/name
    assert subprocess.check_output(['"'"'git'"'"','"'"'-C'"'"',str(repo),'"'"'rev-parse'"'"','"'"'HEAD'"'"'],text=True).strip()==lock['"'"'head'"'"']
    assert not subprocess.check_output(['"'"'git'"'"','"'"'-C'"'"',str(repo),'"'"'status'"'"','"'"'--porcelain'"'"'],text=True).strip()
    for path,expected in lock['"'"'tracked_file_sha256'"'"'].items():
        assert hash_file(repo/path)==expected,(name,path)
        count+=1
print(f'"'"'UPSTREAM_READ_ONLY: {count} tracked file hashes match R0 locks; both repositories clean'"'"')
for filename in ['"'"'README.md'"'"','"'"'reports/R1_report.md'"'"','"'"'reports/R2_report.md'"'"','"'"'reports/R2_learning.md'"'"','"'"'docs/r1_data_description.md'"'"']:
    p=r/filename
    for target in re.findall(r'"'"'\]\(([^)]+)\)'"'"',p.read_text()):
        if '"'"'://'"'"' in target or target.startswith('"'"'#'"'"'):
            continue
        assert (p.parent/target.split('"'"'#'"'"')[0]).exists(),(filename,target)
print('"'"'DOCUMENT_LINKS: all local links in 5 maintained documents resolve'"'"')
paths=(set(old['"'"'file_sha256'"'"'])-removed)|{'"'"'results/r1/history.log'"'"'}
files={name:hash_file(r/name) for name in sorted(paths)}
meta={k:v for k,v in old.items() if k not in ('"'"'file_sha256'"'"','"'"'file_count'"'"')}
meta.update(verified_at=datetime.datetime.now().astimezone().isoformat(),
 scope='"'"'R1 original and revised experiment, after user-authorized cleanup before R2'"'"',
 previous_manifest_sha256=hashlib.sha256(old_bytes).hexdigest(),
 cleanup_archive_byte_hashes_verified=True, r1_data_unchanged_during_r2=True,
 cleanup_removed_files=sorted(removed), file_count=len(files))
def encode(value):
    if isinstance(value,bool):return str(value).lower()
    return json.dumps(value,ensure_ascii=False)
text='"'"'\n'"'"'.join(f'"'"'{k} = {encode(v)}'"'"' for k,v in meta.items())+'"'"'\n\n[file_sha256]\n'"'"'
text+='"'"'\n'"'"'.join(f'"'"'{json.dumps(k)} = {json.dumps(v)}'"'"' for k,v in files.items())+'"'"'\n'"'"'
manifest.write_text(text)
new=tomllib.loads(manifest.read_text())
assert new['"'"'file_count'"'"']==len(new['"'"'file_sha256'"'"'])==54
assert all(hash_file(r/name)==expected for name,expected in new['"'"'file_sha256'"'"'].items())
print('"'"'R1_MANIFEST_REFRESHED: 54 active files; previous manifest SHA256='"'"'+new['"'"'previous_manifest_sha256'"'"'])
for name in ['"'"'src/solvers/adaptive_lasso.py'"'"','"'"'src/solvers/__init__.py'"'"','"'"'scripts/r2_solver_demo.py'"'"','"'"'tests/test_r2_adaptive_lasso.py'"'"']:
    print('"'"'R2_SOURCE_SHA256 '"'"'+name+'"'"' '"'"'+hash_file(r/name))
print('"'"'FINAL_AUDIT_PASS'"'"')
'
```

环境：`PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`

```text
R1_ARCHIVE: 7 original byte streams and SHA256 values verified; old files absent
R1_PRESERVED: all 18 data files and all retained results/source match previous manifest
UPSTREAM_READ_ONLY: 118 tracked file hashes match R0 locks; both repositories clean
DOCUMENT_LINKS: all local links in 5 maintained documents resolve
R1_MANIFEST_REFRESHED: 54 active files; previous manifest SHA256=2988027dbe8a3688b7676570d8f00a38abc8fba3620812aeb1d8716d85c657aa
R2_SOURCE_SHA256 src/solvers/adaptive_lasso.py 30aab9751a1981c2c2901b760ac16bc686bc1f77c81f2103eaffb2ee64e70a23
R2_SOURCE_SHA256 src/solvers/__init__.py 6c5c416eb14575477ee69c8220930fd58ae569f37b9075f2f3729ffca5a46f9d
R2_SOURCE_SHA256 scripts/r2_solver_demo.py 0b6d83ade4da276cf9e3e7dffdb2c518c1ba32dfb8a68595c6073b99147f61ef
R2_SOURCE_SHA256 tests/test_r2_adaptive_lasso.py 4d4b56a6773fc68b230c4699a74091f2d609a6373b0a7ac16377aacf0c896cfd
FINAL_AUDIT_PASS
```

退出码：0

R2 完成；等待人工验收；未执行下一阶段。
