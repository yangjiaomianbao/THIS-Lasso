# THIS-Lasso 15节点公平对照实验 V2：交付记录

交付日期：2026-10-23。所有实现、配置、小试及审计输出均位于 THIS-Lasso 项目内。旧文件未修改；本次仅新增 V2 文件，未提交或推送 GitHub。

**IMPLEMENTED · TESTED · PILOT VERIFIED · FULL EXPERIMENT NOT RUN**

## 阶段状态

|阶段|实际完成内容|状态|
|---|---|---|
|S1–S3|原实现审计、三方法共享特征标准化、自由截距、验证集MSE选参|IMPLEMENTED / TESTED|
|S4–S5|独立随机命名空间、10网络×3数据清单、共享实例检查、单元测试和五系统小试|TESTED / PILOT VERIFIED|
|S6|锁定30组正式配置、900模型入口、断点续跑和资源预检|IMPLEMENTED，等待启动确认|
|S7|逐实例指标、方差、配对差异、网络bootstrap、稳定性分析及图表；小试全流程核验|IMPLEMENTED / PILOT VERIFIED，正式统计尚未运行|

任务书第0节明确要求：**“未经用户进一步确认，不启动全部 30 组正式实验。”** 因此没有启动900模型正式实验，也没有将单实例小试当作30组独立重复。

## 算法与数据约定

三种方法使用同一训练/验证数据、Taylor完整特征及列顺序。非恒定项按训练集均值与中心化RMS统一标准化；验证/测试使用同一变换。常数项不参与惩罚。STLSQ对训练响应中心化，将响应均值作为自由截距，与LASSO/Adaptive相同；导出时统一恢复原尺度系数。原STLSQ数学程序保持不变，新增适配器保留其Ridge初始化、严格小于阈值剪枝和停止规则。

每种方法均以验证集MSE从12个相对参数候选中选择，采用相同的确定性并列规则。STLSQ的阈值比例与LASSO的惩罚比例意义不同，不宣称数值相同即正则强度相同。Adaptive权重仅由训练数据Ridge初估得到。测试数据和结构真值不参与拟合、参数网格构造或选择。

系数锁定后，按原始解码与固定1e-8阈值输出物理二体/三体边；原生Julia边集合核对一致。评价使用210个二体候选/45条真边、1365个三体候选/15条真边。AP使用全部连续得分，MCC未定义保留NA。

网络种子、拓扑子种子、边权子种子、数据种子、噪声种子分别建立随机命名空间。10个网络各配3个数据种子；五系统共享网络拓扑和绝对边权，保留各动力学原本的符号约定。噪声按训练集干净导数标准差定义，并区分系统、数据划分和坐标。三方法共享同一实例的数据/噪声哈希。细节见[实现说明](S2_S4_implementation.md)及[锁定种子](../../../configs/fair_comparison_v2/seed_manifest.csv)。

**统一验证集MSE选参，是为了公平比较三种回归方法，不保证能得到结构恢复F1最优的模型。**

## 测试与小试证据

|检查|结果|证据|
|---|---|---|
|Python全套测试|140通过，0失败；包含24个新增测试|[日志](all_python_tests.log)|
|Julia新增数据/种子检查|141断言通过|[日志](julia_data_tests.log)|
|Julia原五系统方程/契约检查|40断言通过|[日志](legacy_julia_dynamics_tests.log)|
|Kuramoto/SIS小试|12完整模型全部OK，180目标方程|[独立核验](pilot_independent_validation.json)|
|Lorenz资源探针|3个单方程求解全部OK；不作为完整网络模型评价|[控制台](lorenz_probe_console.log)|
|五系统覆盖小试|30完整模型全部OK，720目标方程，60条分阶指标|[独立核验](five_systems_independent_validation.json)|
|跨运行重现|重叠12模型的原尺度系数、选参、输入指纹和指标完全一致|[记录](replay_validation.json)|
|历史文件保护|12,399文件哈希不变，原论文源码和旧R5/15节点结果未修改|[审计](final_integrity.json)|

没有失败的上述测试；900模型正式运行及其最终统计结论尚未执行。原测试日志中的ResourceWarning保留，不将其隐藏。

五系统覆盖小试仅含**1个网络、1个数据实例**；不能估计跨网络稳定性或宣称方法显著优越。30模型=5动力学×2噪声×3方法。模型级NA=0、FAILED=0；Lorenz两种噪声的STLSQ二体MCC共2项未定义，保留NA。所有三体指标可定义。单实例标准差、方差和网络bootstrap区间均为NA，不填0。

覆盖小试生成7,680个互不重复的数据样本/RNG名，原生Julia与适配器解码的4,745条预测边完全一致。所有候选得分、混淆矩阵、AP、MCC、F1、Jaccard、选参锁均由独立脚本复核。

四指标完整小试表和图：

- [三体恢复表](../../../results/dynamics15/fair_v2_five_systems_smoke_20261023_v1/analyses/reviewed_statistics/tables/table1_threebody.md)
- [二体恢复表](../../../results/dynamics15/fair_v2_five_systems_smoke_20261023_v1/analyses/reviewed_statistics/tables/table2_pairwise.md)
- [图表与统计报告](../../../results/dynamics15/fair_v2_five_systems_smoke_20261023_v1/analyses/reviewed_statistics/REPORT.md)
- [逐实例原始指标](../../../results/dynamics15/fair_v2_five_systems_smoke_20261023_v1/trials.csv)

当前小试中，无噪声LASSO在Lorenz、GLV、Stuart–Landau、SIS三体F1为1；STLSQ在无噪声Kuramoto、GLV、SIS三体F1为1。Lorenz 5%噪声三方法三体F1均低于0.06。部分配置AP很高但F1较低，说明排序表现与固定阈值下的结构恢复不同；保留这一结果，不按真值调整阈值。

## 网格边界与资源

每方法240个目标方程中，STLSQ有27次选中最大候选阈值；LASSO、Adaptive分别有105、95次选中最小惩罚候选。见[边界诊断](grid_boundary_diagnostics.csv)。这提示当前候选路径的覆盖范围可能影响结果，不等于结构指标最优，也没有据此改动已锁定网格。STLSQ采用原始严格小于规则，最大阈值不保证产生纯截距模型。

五系统覆盖小试墙钟时间1,086.58秒（18.11分钟），8个worker、每个BLAS单线程；进程树采样RSS峰值6,216.47MiB（约6.07GiB，共享页可能重复计数）。Lorenz单方程5%噪声探针：STLSQ约6.76秒、LASSO约25.74秒、Adaptive约24.68秒。不同目标及网络的困难路径可能明显更慢，保留原KKT检查和工作集回退。

按覆盖小试简单乘30约9.05小时，**仅为单网络外推，不保证正式耗时**。硬件/代码/配置检查见[S6预检](S6_readiness.json)，复现、正式运行、续跑及统计命令见[运行手册](S6_S7_runbook.md)。

## 新增文件清单与保持不变的内容

|新增路径|用途|
|---|---|
|`src/solvers/stlsq_validation.py`|自由截距STLSQ验证选参适配器|
|`src/experiments/fair_v2.py`|锁定配置、独立种子、共享预处理与三solver接口|
|`src/experiments/fair_v2_statistics.py`|完整网络指标汇总、有效数、配对和网络重采样|
|`scripts/fair_v2_prepare.jl`|复用原方程和Taylor库的数据准备|
|`scripts/fair_v2.py`|计划、小试、正式运行、续跑、模型锁定与审计入口|
|`scripts/fair_v2_verify.py`|独立指标与锁定文件核验|
|`scripts/fair_v2_analyse.py`|S7统计、方差与图表输出|
|`scripts/fair_v2_preflight.py`|正式启动预检与候选网格边界诊断|
|`tests/test_fair_v2.py`、`tests/test_fair_v2_data.jl`|新增公平性、随机性、统计和数据测试|
|`configs/fair_comparison_v2/`|锁定配置、30组种子、10网络边权及哈希|
|`docs/experiments/fair_comparison_v2/`|审计、设计说明、运行手册、日志和交付记录|
|`results/dynamics15/fair_v2_pilot_20261023_v1/`|12模型先行小试|
|`results/dynamics15/fair_v2_lorenz_probe_20261023_v1/`|Lorenz资源探针|
|`results/dynamics15/fair_v2_five_systems_smoke_20261023_v1/`|30模型覆盖小试及独立统计分析|

**既有文件修改：0。** 原始论文源码、既有solver、原方程、原解码、R0–R5及以前15节点正式结果保持不变。V2调用既有实现，不修改其源文件。所有V2输出进入新的独立目录，已有运行拒绝覆盖。新增代码、测试、配置与审计文件校验清单见`delivery_manifest.json`（不包含运行缓存和清单自身）；实验结果的来源与选择锁由各运行目录单独记录并核验。

正式设计为10独立网络×3数据重复×5系统×2噪声×3方法=900完整网络模型。统计程序分别报告均值、样本标准差/方差、有效数、网络间/网络内波动及严格配对性能差异。bootstrap以网络均值为重采样单位，不能将30数据实例视为30独立网络；网络变化同时含拓扑和边权，描述性方差不代表纯拓扑效应。
