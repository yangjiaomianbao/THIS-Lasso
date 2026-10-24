# S6正式运行准备与S7统计流程

状态：正式配置和入口已实现；30组正式实验尚未运行。小试通过不代表三种算法的科学性能已经得到充分验证。

## 已锁定配置

- [完整配置](../../../configs/fair_comparison_v2/config_locked.toml)
- [30组种子](../../../configs/fair_comparison_v2/seed_manifest.csv)
- [10个网络及边权](../../../configs/fair_comparison_v2/network_manifest.csv)
- [配置与清单哈希](../../../configs/fair_comparison_v2/plan_lock.json)

每网络3个数据种子，每实例5系统、2噪声、3方法，总计900个完整网络模型、1,800条分阶指标。方法为STLSQ val、LASSO val、Adaptive val；不混入旧BIC或STLSQ raw结果。全部网络、数据种子预先固定，不根据小试表现更换。

## 复现小规模检查

以下命令应在项目根目录执行，标签必须尚不存在。`reproduce01`只用于新的独立输出，不覆盖已交付的小试。

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv-r0/bin/python -B -m unittest discover -s tests

# 1网络、1采样、Kuramoto/SIS、2噪声、3方法：12个完整模型。
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv-r0/bin/python -B scripts/fair_v2.py run \
  --mode pilot --label fair_v2_pilot_reproduce01 --workers 8

# 全45维Lorenz、完整17296特征，5%噪声，第1目标方程的3种方法资源探针。
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv-r0/bin/python -B scripts/fair_v2.py run \
  --mode probe --label fair_v2_lorenz_probe_reproduce01 --workers 3

# 五系统、1网络、1采样：30个完整模型，仍是小试。
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv-r0/bin/python -B scripts/fair_v2.py run \
  --mode coverage --label fair_v2_five_systems_reproduce01 --workers 8
```

Julia解析数据测试命令记录在`julia_data_tests.log`对应本目录3份data_test_config中。完整计划已有锁定文件，无需再次运行`plan`覆盖它；该命令会拒绝覆盖。

## 正式实验命令：等待用户进一步确认

任务书要求“未经用户进一步确认，不启动全部30组正式实验”。以下仅提供命令，不表示已执行。

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  .venv-r0/bin/python -B scripts/fair_v2.py audit

PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv-r0/bin/python -B scripts/fair_v2.py run \
  --mode full --label fair_comparison_v2_full_run01 --workers 8 --confirm-full \
  > docs/experiments/fair_comparison_v2/full_run01_console.log 2>&1

PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  .venv-r0/bin/python -B scripts/fair_v2.py audit
```

进程中断后，仅在同一代码和锁定计划下，用原命令增加`--resume`继续未完成方程；已记录FAILED不会被选择性重试。已完成运行拒绝续跑和覆盖。保留完整控制台日志，节点异常还会写入各node JSON和failure_reasons.csv。

## 正式统计分析

900模型完成后，运行入口已生成原始trials、summary、selection_details、paired_differences、paired_summary、stability、edge_scores和两张主表。以下S7命令再执行独立得分/选择锁核验，并在新子目录生成完整四指标图、配对差异区间图、跨网络稳定性图和显式方差表，不拟合任何模型：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  .venv-r0/bin/python -B scripts/fair_v2_analyse.py \
  results/dynamics15/fair_comparison_v2_full_run01 --label formal_statistics01
```

所有主要指标先按完整网络计算。NA/FAILED不填零；各指标有效数独立记录；样本标准差和样本方差采用ddof=1。MCC分母为0保留NA。

配对差异为Adaptive−LASSO、LASSO−STLSQ、Adaptive−STLSQ，逐实例严格匹配数据/噪声哈希。网络bootstrap采用2000次重采样，区间针对网络等权的配对均值；同时记录pooled pair mean和有效配对数。网络数少于2时区间NA。只有10个独立网络，区间可能不稳定，不宣称总体显著优越性。

显式方差表区分总体实例样本方差、网络均值的样本方差、网络内样本方差平均值。后两项是描述性统计，不能解释为已经分离的纯拓扑随机效应；本设计同时改变拓扑与权重。

**统一验证集MSE选参，是为了公平比较三种回归方法，不保证能得到结构恢复F1最优的模型。**

## 资源估计

资源以实际五系统小试和Lorenz探针测量为依据。`S6_readiness.json`提供小试墙钟耗时、进程树RSS峰值及粗略30倍外推；它不保证正式运行时间，因为不同网络上的LARS回退路径可能明显更慢。默认8个worker，每个BLAS单线程，矩阵内存映射；不通过删特征或放宽容差缩短运行。
