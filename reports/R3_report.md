# R3 执行报告

日期：2026-10-16（Asia/Shanghai）。用户明确授权执行 R3。本阶段完成 CSV/TOML 融合接口，在同一个 R1 修订种子上运行 STLSQ、LASSO、Adaptive LASSO。**Python 15 项测试与 Julia 63 项断言通过；三种求解器演示通过。等待人工验收，未执行 R4。**

## 实际文件与最小修改范围

| 新增文件 | 作用 |
| --- | --- |
| [src/this_adapter.jl](../src/this_adapter.jl) | 原生 get_θd、STLSQ 直通、Python 调用、请求/返回校验、同规则解码 |
| [src/data_bridge.py](../src/data_bridge.py) | CSV/TOML 读写，调用只读 R2 求解器及普通 LASSO |
| [scripts/r3_bridge.py](../scripts/r3_bridge.py) | 桥接命令行入口 |
| [scripts/r3_run.py](../scripts/r3_run.py) | 记录真实命令、环境、时间、完整输出与退出码 |
| [scripts/r3_demo.jl](../scripts/r3_demo.jl) | 一个既定 N=8 种子的三种求解器接口演示 |
| [scripts/r3_audit.py](../scripts/r3_audit.py) | 数据/源码保护、请求/返回、记录与产物清单核对 |
| [tests/test_r3_bridge.py](../tests/test_r3_bridge.py)、[tests/test_r3_adapter.jl](../tests/test_r3_adapter.jl) | Python 单元测试与 Julia 跨语言数值测试 |
| `results/r3/` | 预检快照、协议、交换文件、系数、指标、完整日志及审计清单 |
| 本报告、[R3 学习笔记](R3_learning.md) | 验收依据、公式/维度与具体特征映射解释 |

仅新增上述 R3 允许范围内的文件。R1 数据/结果、R2 求解器、依赖锁定、上游文件及人工进度勾选表均未修改。未增加依赖，未修改 Pearson 筛选或新增 H/SC 分类器。

源码版本沿用 R0 锁定：THIS `865efc43fe8ec21966659b482ca5756396974425`（GPL-3.0），Su `fa0d662f68b7b41a1c6c3401af9504929126dafd`（MIT）。Julia 私有解码提取自锁定的 THIS `this.jl`，保留来源说明和原始判定；两份 upstream 仓库只读。

## 接口契约与原生等价性

入口为 `run_this_solver(X,Y,ooi,dmax;solver=...,...)`，返回命名元组 `Ainf, coeff, relerr, theta, d, diagnostics`。STLSQ 直接调用原始 `this`，并检查相同系数经私有解码后与原生 Ainf 精确相等；Python 路径只替换求系数步骤。

Julia 导出 `theta.csv:[P,K]`、`Y.csv:[N,K]`、`monomial_indices.csv:[P,dmax]` 和 `metadata.toml`，浮点格式为 `%.17g`。Python 用 `ndmin=2` 读取并取 `theta.T:[K,P]`，回写 `coefficients.csv:[N,P]`。metadata 保存 N/K/P/dmax、显式形状、1 起始节点号、0 填充、特征顺序 SHA-256 和索引条目总数。R1 的 P=165，dmax=3，索引条目数为 495。

特征顺序哈希定义为 d 的 CSV 原始字节 SHA-256，与 R1 完全一致；不会重排 d 或随意减 1。返回 response 保存调用前 metadata 的哈希、相同特征哈希、输出形状和系数/诊断哈希；Julia 拒绝旧响应、输入被改写、错位特征、错误行列数或非有限结果。response 最后写入，已有产物拒绝覆盖。

解码仍为严格 `abs(coef)>1e-8`，排除重复非零源节点，并排除目标位于源集合的项；源节点仅在输出边中排序。返回 o=1…4 保持原生行为，本轮评价明确限于 o=2/3。`relerr` 保持原生 `norm(Y-C*theta,1)/norm(Y,1)` 定义。

## 数据、预先固定的选择与阶段边界

使用 `data/r1/revision/` 唯一既定种子 **21261016**，N=8。训练 512 条独立轨迹/1536 样本，验证 128 条独立轨迹/384 样本；未生成新轨迹、未搜索种子。先逐项核对保存的 theta、d 与原生 `get_θd` 精确一致，再运行求解器。

[protocol.toml](../results/r3/protocol.toml) 在拟合前保存。STLSQ 沿用 R1 已锁定 λ=0.03、ρ=0、niter=10；普通 LASSO 和 Adaptive LASSO 都显式用独立验证轨迹选择 alpha，每节点 12 个训练 alpha_max 比例候选。容差内选择较大 alpha，保留全部路径。R2 的初值、标准化和权重始终只由训练数据确定；普通 LASSO 使用原始单位等权 L1 与无惩罚常数项。无噪声数据不满足可靠 BIC 方差估计，未强行使用 BIC。

所有模型完成选参后才读取结构真值用于简单 F1/MCC 检查。原 CSV 含既定测试列，但不把其切片传给拟合/选参，也不计算测试指标。这里的 12 点 L1 路径是接口内的参数选择，不是 R4 的网络对照网格。

## 实际接口检查结果

| 求解器 | 成对 TP/FP/FN | 成对 F1/MCC | 三体 TP/FP/FN | 三体 F1/MCC | 验证导数 MSE |
| --- | --- | --- | --- | --- | --- |
| STLSQ | 10/0/0 | 1 / 1 | 3/0/0 | 1 / 1 | 4.656088493363959×10^-31 |
| LASSO | 10/0/0 | 1 / 1 | 3/0/0 | 1 / 1 | 2.511969281818678×10^-14 |
| Adaptive LASSO | 10/0/0 | 1 / 1 | 3/0/0 | 1 / 1 | 3.108299350960535×10^-14 |

候选全集沿用 R1：56 条有向成对边、168 条有向目标/无序双源三体边。详细计数、训练/验证误差和忽略阶次行数见 [summary.toml](../results/r3/summary.toml)；全部非零 Taylor 项和解码边已打印在 [demo.log](../results/r3/logs/demo.log)，并分别保存每种求解器的 `nonzero_taylor.csv`、`inferred_structure.csv`。

STLSQ 的全部系数与 R1 修订选参结果精确一致。按原阈值计，三种方法分别有 21、22、22 个 Taylor 系数；其中 8 个自衰减项不解码为自环。两个 L1 方法各有一个约 1.6161×10^-8 的常数项，原生规则返回为 o=1，均完整保留且不计入 o=2/3 指标；三种方法 o=4 行数均为 0。没有提高阈值来隐藏这些输出。

普通 LASSO 最大归一化 KKT 违背量为 1.3273308075192856×10^-13；Adaptive LASSO 为 2.2303611906142563×10^-14，均无收敛或其他警告。Adaptive LASSO 最大原尺度预测还原差 4.440892098500626×10^-16，八个节点的原目标与变换后目标之差在本次记录中均为 0。

[feature_trace.toml](../results/r3/feature_trace.toml) 给出具体实例：Julia 第 138 行、Python 第 137 列，d=[0,6,5]，对目标 7 解码为 7←{5,6}；重复源 x₅² 对应第 133 行 d=[0,5,5]，不作为两个不同源的三体边。学习笔记逐步解释文件出入和系数还原。

上述结果仅验证本次接口能正确连接。无噪声且 alpha>0 的 L1 解有收缩偏差，这里不以系数逐位等于真值为要求；也不据此比较方法优劣、弱耦合优势或含噪鲁棒性。

## 实际命令、测试与完整日志

环境：独立 Python 3.11.17、NumPy 1.26.4、scikit-learn 1.6.1；Julia 1.9.4，使用 R0 Project/Manifest 和项目 depot。日志器设置 `PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_NUM_THREADS=1`。从项目根目录执行：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-r0/bin/python scripts/r3_run.py python_test
PYTHONDONTWRITEBYTECODE=1 .venv-r0/bin/python scripts/r3_run.py julia_test
PYTHONDONTWRITEBYTECODE=1 .venv-r0/bin/python scripts/r3_run.py python_test --label python_test_final
PYTHONDONTWRITEBYTECODE=1 .venv-r0/bin/python scripts/r3_run.py julia_test --label julia_test_final
PYTHONDONTWRITEBYTECODE=1 .venv-r0/bin/python scripts/r3_run.py demo
PYTHONDONTWRITEBYTECODE=1 .venv-r0/bin/python scripts/r3_run.py audit
```

| 验证 | 结果 | 完整日志 |
| --- | --- | --- |
| 首轮 Python 测试 | 13 项通过，退出 0 | [python_test.log](../results/r3/logs/python_test.log) |
| 首轮 Julia 测试 | 63 项断言通过，退出 0 | [julia_test.log](../results/r3/logs/julia_test.log) |
| 补充常数响应/单系数列后的 Python 测试 | 15 项通过，退出 0 | [python_test_final.log](../results/r3/logs/python_test_final.log) |
| 最终 Julia 跨语言测试 | 63 项断言通过，退出 0 | [julia_test_final.log](../results/r3/logs/julia_test_final.log) |
| 单种子三方法演示 | 完成，退出 0 | [demo.log](../results/r3/logs/demo.log) |
| 最终文件与边界审计 | 读取固定产物核对，无重新选参 | [audit.log](../results/r3/logs/audit.log) |

没有失败的测试或演示运行。首轮通过后补充了常数响应的解析处理和单系数列的返回形状测试，并重新验证受影响的 Python/Julia 路径；保留两轮日志。负面测试中的哈希错位、旧响应、维度错误及轨迹混用均是预期拒绝，未静默跳过。

Julia 断言分组为：原生解码/阈值/排除规则 11 项，跨语言精度和单轴形状 18 项，Python 回归系数到 Julia 特征对应 20 项，R1 全部特征/真值/原生系数一致性 14 项。单行、单列、1×1 矩阵往返最大误差均为 0。Python 测试还覆盖双求解器输出轴、无惩罚常数、输入校验、不可覆盖和独立轨迹限制。

日志器保留每条内部命令、起止时间、环境、全部输出和退出码。再次执行测试应换 `--label`，避免覆盖旧证据；正式 demo 与桥接目录同样拒绝覆盖。需要另做小型复查时，可使用适配器的一个新空 work_dir；不要删除已验收数据来重跑。

## 产物、已知限制与进度

[preflight.toml](../results/r3/preflight.toml) 保存开始时的 140 个保护文件哈希，包括两版 R1 数据、R2 求解器/测试/演示与上游文件。[artifact_manifest.toml](../results/r3/artifact_manifest.toml) 保存 R3 源码、报告和产物哈希；正在写入的审计日志与清单自身不纳入自指哈希。最终审计同时核对 R1 原有 54 项清单、R0 的上游 118 项锁定哈希和两个仓库的干净状态，并确认两种 Python 方法看到完全相同的训练/验证矩阵。

桥接采用多个可读文件并保留正式中间矩阵，适合当前小网络，未优化大量文件或大规模进程调用；当前磁盘约 14 MB。普通 LASSO 暂提供固定 alpha/独立验证选择，Adaptive LASSO 沿用 R2 的 BIC/固定/验证功能。对于测试响应 Y 全零的情形，原生 `relerr` 定义可能为 NaN；本轮实际数据没有该情况，未擅自修改其定义。

机器完成状态：维度/索引合同已核对；CSV/TOML 精度、形状和哈希测试通过；同一 R1 数据三种路径运行；STLSQ 解码精确一致；具体三体特征追踪已保存；完整配置/系数/警告/路径/日志可复查。人工阶段检查表保持未勾选。未启动 R4、未运行批量试验、未实现下一阶段代码。

R3 完成；等待人工验收；未执行下一阶段。

## 2026-10-17：按六项清单独立复核

本次按用户列出的六项重新检查，**六项自动核查均通过**，没有代勾任务文件的人工验收清单，也没有启动 R4。复核开始前先逐项验证原 R3 清单的 51 个文件哈希，全部匹配；原清单另存为 [original_manifest.toml](../results/r3/review_20261017/original_manifest.toml)。

| 核查项 | 实际检查与结论 |
| --- | --- |
| Julia/Python 维度、索引、输出契约 | 训练 theta=165×1536，Python design=1536×165，Y=8×1536，三种回传系数均为 8×165；d=165×3，节点号 1…8、0 填充，检查通过。 |
| 可重复、具备精度检查的 CSV/TOML 交换 | 重新将完整 N=8 的训练和验证 theta/Y/d 经 Julia→Python→Julia 往返；theta 和 Y 的最大绝对误差均为 0。单行、单列、1×1 边界测试也再次通过。 |
| 三种 solver 作用于同一 R1 数据 | 同一个种子 21261016、相同训练/验证切片重新运行三种求解器；各自与已保存系数的最大差均为 0，解码边身份与系数核对通过，未覆盖旧实验文件。 |
| 特征顺序与 monomial indices | 全部 165 行、495 个节点索引往返逐项精确相等；索引 CSV SHA-256 与 R1 的 feature_order_hash 一致；重排索引的负面测试也通过。 |
| 原始 STLSQ 系数解码一致 | 对 N=8 原生 STLSQ 系数，私有解码与原始 this 返回的全部 o=1…4 Ainf 矩阵精确一致；与保存的 STLSQ 系数、规范排序后的输出边也精确一致。 |
| 已打印“特征→系数→边”实例 | 重新打印 Julia 特征 138 / Python 列 137、d=[0,6,5]，对目标 7 得到 x₅*x₆ 的系数并回到 7←{5,6}；额外给 x₅² 人工非零测试系数，确认它不会被解码成双源三体边。 |

完整动态核查见 [checklist.log](../results/r3/review_20261017/checklist.log)，结构化结果见 [checklist_results.toml](../results/r3/review_20261017/checklist_results.toml)。本次实际重新执行：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-r0/bin/python scripts/r3_run.py python_test --label review_20261017_python
PYTHONDONTWRITEBYTECODE=1 .venv-r0/bin/python scripts/r3_run.py julia_test --label review_20261017_julia
environment/r0/julia.sh --startup-file=no --project=environment/r0/julia scripts/r3_checklist.jl
PYTHONDONTWRITEBYTECODE=1 .venv-r0/bin/python scripts/r3_run.py audit --label review_20261017_audit
```

核查命令日志记录完整绝对路径及 `PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_NUM_THREADS=1` 环境。Python 再次 **15 项通过**，见 [review_20261017_python.log](../results/r3/logs/review_20261017_python.log)；Julia 再次 **63 项断言通过**，见 [review_20261017_julia.log](../results/r3/logs/review_20261017_julia.log)；独立六项核查脚本退出码为 0。最终文件审计记录在 [review_20261017_audit.log](../results/r3/logs/review_20261017_audit.log)，并刷新本阶段产物清单。

本次只新增 `scripts/r3_checklist.jl`、本次日志/结果/原清单快照，并在本报告追加核查结论、刷新 R3 产物清单；求解器、适配器、原测试代码、原实验数据与结果保持不变。临时桥接目录由测试自动删除；没有遗留重复的完整矩阵。所有运行均为同一个既定种子的接口复查，没有测试集评价或新的超参数策略，不据此作方法性能结论。

R3 完成；等待人工验收；未执行下一阶段。
