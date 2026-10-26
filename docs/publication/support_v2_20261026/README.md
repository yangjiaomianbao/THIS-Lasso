# 2026-10-26 支持集与结构误检诊断：GitHub交付

执行范围：Kuramoto、Lotka–Volterra、SIS，15物理节点，每方程完整816项三阶Taylor库；0/5/10%导数响应噪声，六种原流程。A固定原网络10数据种子，B另加5网络各10数据种子。3240配置：3000 OK、240原BIC不适用、0 FAILED。

- [总报告](../../../results/diagnostics/this_lasso_support_3systems_10seeds_v2/REPORT.md)
- [固定网络报告](../../../results/diagnostics/this_lasso_support_3systems_10seeds_v2/A_fixed/REPORT.md)
- [多网络报告](../../../results/diagnostics/this_lasso_support_3systems_10seeds_v2/B_networks/REPORT.md)
- [本次公开文件清单](published_files.csv)
- [3240模型状态与系数哈希](model_manifest.csv)
- [完整本地文件清单与哈希](../../../results/diagnostics/this_lasso_support_3systems_10seeds_v2/artifact_manifest.csv)

## 发布范围

GitHub包括本次新增代码、测试、实验方案审阅、同步核查、原始与分阶段指标表、配对比较、相对alpha路径、逐方程支持统计、解码阈值诊断、有限冗余诊断、PNG/PDF图、固定种子与实例配置、数据元信息、运行记录及源码快照。

约61 GiB完整产物保留在本地。未放入Git历史：原始状态/导数与特征大矩阵、全部节点候选系数NPZ、每个BIC折点的压缩详细表、Julia包缓存、可再生预检数据。A/B的节点原生路径表分别约242 MiB/1190 MiB；公开的04_alpha_path.csv保留逐模型共同12点相对alpha诊断，05参数表保留逐方程选择信息。完整原生BIC折点数据仍在本地，不能将公开12点表误称为全部BIC折点。

published_files.csv列的是本次明确加入Git的文件；artifact_manifest.csv列的是完整本地产物，包括未上传文件。后者的某条目存在不代表该文件可从GitHub下载。

## 数据与核验

每个样本是独立t=0初始状态及解析瞬时导数，没有时间积分或同步预热。训练/验证/测试为1024/256/256，六方法共享数据和噪声底样本。三类系统沿用原五类配置的固定编号，配置中保留其他系统定义只为保持随机流命名，未运行Lorenz或Stuart–Landau。

原生THIS严格按abs(coef)>1e-8输出结构，AP使用不截断的连续分数。两者分别完成4725000项精确Julia/Python核对；6个非零但不超过阈值的分数保留用于AP。核验修复没有更改模型系数、参数或指标。

新增8项诊断测试通过。原全量Python检查144项通过，1项因本地历史R5 fixture缺失报错，见已提交验证日志。统计区间按固定网络数据种子或多网络网络均值重采样，不把同网络10次数据准备视为10个独立网络。

## 代码与复现条件

主入口为scripts/support_run.py，分析入口为scripts/support_analyse.py。Python/Julia依赖、上游源码获取方式见environment/r0/README.md。运行脚本沿用完整工作区中的既有基线目录，用它核对并复用原两个种子；本GitHub交付不包含这些大型基线输入，因此仅克隆公开表格不是完整的原始工作区。

在已配置环境、已保留原基线输入的完整工作区中，可用新的输出目录重放：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv-r0/bin/python -B -u scripts/support_run.py --out results/diagnostics/support_reproduce --workers 8
```

不要覆盖本次冻结结果目录。运行时源码快照及source_sha256记录对应实际实验；维护修复的before/after哈希与原因保存在logs中的回执。公开源码包含这些已验证的核验和恢复修复。
