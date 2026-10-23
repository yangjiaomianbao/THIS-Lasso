# Kuramoto 与 Lorenz：首次 THIS + LASSO 动力学扩展

2026-10-21。两个系统已经实际跑通，48个模型记录为44 OK、4 NA、0拟合FAILED。结果表明共享求解器可用于新动力学；Adaptive LASSO是否改善恢复，取决于动力学、噪声及选参方式。

模型形式依据[THIS原论文的合成模型](https://arxiv.org/html/2402.00078v2)。固定上游仓库没有两个系统的生成脚本，因此本次新增了生成器；原THIS核心用于特征、STLSQ和解码。本次使用新拓扑、非零展开中心、短轨迹及解析导数，是工程试跑，不是论文原图复现。

## 设计

- Kuramoto：4个相位节点、5条有向二体边、2条有向三体边，完整三阶Taylor库35列。
- Lorenz：3个物理振子、9个状态变量，3条二体边、1条三体边，三阶库220列。三个状态变量聚合回同一个物理振子，内部项不算网络边。
- 每系统2个独立seed，每seed128/32/32条训练/验证/测试轨迹，每条3点，共384个训练点。解析导数，响应噪声0%/5%，未使用观测噪声、差分导数或长时吸引子数据。
- 同一系统同一seed和噪声下，六种方法共享全部数据。训练集计算标准化及Ridge权重；所有选参和系数先锁定，再计算结构真值与测试误差。
- AP使用全部候选边连续得分；F1严格使用原1e−8支持阈值，没有按真值调整阈值。两seed汇总只做描述，不做显著性结论。

完整设计见[预声明协议](../docs/dynamics_pilot_protocol.md)，参数见[配置](../configs/dynamics_pilot.toml)。三体作用对应的目标←源二体边在这两个拓扑中都存在，这是一项有利条件，不能外推到无二体边界的任意超图。

## 实际三体恢复效果

下表为两个seed的平均F1，噪声比例指训练导数总体标准差的倍数。

| 方法 | Kuramoto 0% | Kuramoto 5% | Lorenz 0% | Lorenz 5% |
| --- | ---: | ---: | ---: | ---: |
| M0：原默认THIS/STLSQ | 0.000 | 0.000 | 0.500 | 0.500 |
| M0s：共同标准化STLSQ | 0.000 | 0.000 | 0.000 | 0.000 |
| M1：普通LASSO+BIC | 0.450 | 0.667 | NA | 0.000 |
| M2：Adaptive+BIC | 1.000 | 0.733 | NA | 0.500 |
| M1v：普通LASSO+验证集 | 0.382 | 0.400 | 1.000 | 0.500 |
| M3：Adaptive+验证集 | 0.556 | 0.500 | 0.667 | 0.667 |

Kuramoto的Adaptive+BIC在这次局部试跑中改善了支持恢复。无噪声时，M1和M2的三体AP均为1，但F1分别为0.45和1：普通LASSO将真实边排在前列，同时保留了阈值以上的小假阳性。因此，F1差异不等同于排序能力的同等差异。

Lorenz零噪声可由三阶多项式库精确表达，BIC的残差方差可靠性门槛不满足，M1/M2在两个seed中均为NA，理由完整保留。验证集选参的普通LASSO恢复了本试验全部二体/三体边，三体F1=1，干净测试导数NMSE约3.80e−13；Adaptive三体F1约0.667、NMSE约8.33e−8，说明其在该情形下没有优势。

5%噪声Lorenz中，Adaptive+BIC三体F1为0.5、普通+BIC为0；验证集路径Adaptive的F1为0.667、普通为0.5。但验证集路径的AP反而是Adaptive 0.5、普通0.75。不能只挑一个指标宣称普遍更好。该系统仅3个物理三体候选、1条真边，结果对单条边非常敏感。

原默认STLSQ在Kuramoto上未恢复三体边。这是固定默认参数在当前局部特征尺度上的结果，不能解释成STLSQ经验证选参后的性能上限。本轮没有针对它补做调参。

![三体F1与AP](../results/dynamics_pilot/kuramoto_lorenz_20261021/triple_comparison.png)

## 验证、修复与保留证据

103项Python测试通过；34项新数据/解析检查和6项Julia聚合断言通过。检查涵盖混合导数与解析Taylor系数、Lorenz精确多项式表达、数据分割，以及物理自边/重复物理源排除。Julia解码与Python独立聚合的支持集和分数一致。

首次后处理遇到上游one2dim的`sort(keys(...))`在Julia1.9报错。新增适配层只将键收集成向量后排序，原文件和原函数保持；失败日志保留。48个模型和208份节点诊断保持，0次重新拟合，没有更换seed、alpha或系数来改善结果。

原S6交付清单25,368个条目前后哈希均一致，其中包括两篇论文原始仓库和正式R5的4,113个文件；旧实验结果、配置、报告和算法未改。

- [本批完整报告](../results/dynamics_pilot/kuramoto_lorenz_20261021/REPORT.md)
- [逐trial指标](../results/dynamics_pilot/kuramoto_lorenz_20261021/trials.csv)、[均值及可用数](../results/dynamics_pilot/kuramoto_lorenz_20261021/summary.csv)
- [全部物理边得分](../results/dynamics_pilot/kuramoto_lorenz_20261021/edge_scores.csv)、[警告及NA理由](../results/dynamics_pilot/kuramoto_lorenz_20261021/warnings_and_unavailable.csv)
- [后处理修复证据](../results/dynamics_pilot/kuramoto_lorenz_20261021/postprocess_recovery.json)、[原文件完整性](../results/dynamics_pilot/kuramoto_lorenz_20261021/legacy_integrity_after.json)
- [Python测试日志](../results/dynamics_pilot/python_tests_20261021.log)、[Julia解析检查](../results/dynamics_pilot/kuramoto_lorenz_20261021/logs/julia_prepare.log)、[聚合检查](../results/dynamics_pilot/kuramoto_lorenz_20261021/logs/physical_decode_tests.log)

## 重新运行

在项目根目录执行以下命令，必须使用尚不存在的新标签；修复后的普通入口会完成全部阶段。

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv-r0/bin/python -B scripts/dynamics_pilot.py --label 自定义新标签
```

本次证据支持“新动力学实验入口可用”，尚不支持大规模、长时混沌、数值微分/观测噪声条件下的正式优劣结论。后续扩大seed、网络与采样范围时，应单独版本化协议和输出。
