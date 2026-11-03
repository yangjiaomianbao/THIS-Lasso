# P0 + P1 交付报告（2026-11-03）

**当前有效结果：statistical_final/。** statistical/、statistical_verified/为开发核验过程保留的独立运行，不能合并进最终样本。neural_final/与neural_n30_final/为本轮有效神经smoke。

## 1. 已完成的范围

已完成仓库审查、统一数据/筛选/三阶函数库/原Adaptive BIC集成、测试、N30五统计方法全候选执行及两种神经架构的训练/未见规模推断smoke。**没有启动N50/100/200/500/1000正式实验，也没有运行30种子扩量。**
代码入口：experiments/large_scale_screening_v1/；审查与实施计划为该目录AUDIT_AND_PLAN.md，复现及P2/P3/P4完整命令为README.md。

## 2. 六种方法状态

|方法|实现状态|P1状态|
|---|---|---|
|Pearson|目标导数边际相关，候选对几何均值；与旧状态Pearson分开|30目标×406候选，完成|
|SCC|共同OLS残差、组内相关差、对称化|完成|
|SCC-OLS|同残差/分组，全局训练标准化、带截距组斜率差|完成；不覆盖旧同名相关差|
|FWL|SVD投影，显式分块残差，增量RSS；无重复求逆|完成，并通过近共线OLS恒等测试|
|GC-CMI|秩高斯化+收缩协方差+条件信息差|完成，始终experimental；纯乘积盲区和非线性假阳性均已有反例测试|
|Triadic-NDformer|共享时序编码+对称融合；轻量与Transformer；验证拓扑checkpoint|两个模型各3步smoke完成，N8训练→N30推断接口完成；未完成正式预训练或泛化效果检验|

## 3. 数学/代码审查的重要结论

- 原生THIS完整库P=C(N+3,3)，原Pearson是状态—状态全局排序；本任务指定导数边际相关，所以独立实现并标明新定义。
- 三阶候选组是4列，不是一列；统一保留低阶及目标混合项，P=6N−2+4K，原生d映射已与真实Julia核对。
- 原Adaptive：统一外部训练标准化坐标下的Ridge初值、w=(abs(init)+eps)^(-gamma)、原LARS/BIC路径及全部NA/KKT门槛均复用公共M2，没有复制求解器或回退验证集。
- 初稿FWL用范数平方差会在候选接近低阶空间时相消；最终版显式计算每块投影残差，有独立的近共线数值测试。这项修正在新目录内，旧算法文件未改。
- 拓扑身份哈希仅由节点数和二/三体边集决定，排除ω和初值；不让不同频率伪装成不同拓扑。神经train/validation/test/benchmark图哈希均不重叠。
- 原THIS三体读出只认二次xj*xk：混合正弦的二阶系数是−(K3/d3_i)sin(θ0j+θ0k−2θ0i)，可为0而三次项仍非零。保留三阶库不等于消除了该相位盲区；本轮不擅自换成组系数范数解码。
- N1000强制列5998已经超过4096训练样本；不改样本/库协议，原BIC必然NA。dry_run_n1000.json只是规模估算，不是N1000运行结果。

## 4. 已通过/未通过

25项新增科学测试全部通过、无skip，包括真实Julia Taylor/解码对照、FWL完整OLS恒等、近投影残差、纯乘积、线性负例、代理变量、尺度/周期/退化组、GCMI失败边界、块等价、神经对称性、库映射、公共M2一致性、拆分与连续AP。
历史Adaptive 30项测试在本次含torch环境及原.venv-r0环境各通过一次。日志已保存。
需要区别：测试全部通过不代表GC-CMI具有纯乘积识别能力；其失败边界是测试预期。N30全部BIC NA是本次真实数值结果，不是被隐藏的测试成功或伪造零F1。

## 5. N30实际结果

固定拓扑种子2026110310、数据种子2026110410、导数噪声5%、K16、每轨迹256点、4/1/1条train/validation/test轨迹；平均二元度4、三体关联度6。pilot冻结K2=0.3、K3=0.6。以下为目标级宏平均筛选指标，非全网络结构F1。

|方法|Recall@16|连续筛选AP|筛选秒数(30目标)|回归秒数|NA目标|
|---|---|---|---|---|---|
|fwl|0.020000|0.024291|0.22441|1.44559|30/30|
|gc_cmi|0.051190|0.024030|0.02363|1.47119|30/30|
|pearson|0.033783|0.025511|0.01092|1.44544|30/30|
|scc|0.014815|0.019596|0.53348|1.44550|30/30|
|scc_ols|0.014484|0.027139|0.51623|1.43590|30/30|

**所有最终结构F1/AP/MCC/Jaccard为NA。** 原因是所有150个库触发原BIC的秩/病态检查，未生成可接受的系数模型，不能把缺失系数当零模型。FWL修正前的开发结果不用于该表。
代表性目标节点1的矩阵诊断如下（非恒定列241）：

|方法|数值秩|条件数|
|---|---|---|
|pearson|95|8.6684e+16|
|scc|96|8.0955e+16|
|scc_ols|95|8.6264e+16|
|fwl|96|8.2665e+16|
|gc_cmi|95|8.236e+16|

**实测解释：** 同一公共低阶投影秩为31，节点1低阶解释率约99.746511%；四条平滑短轨迹仅形成有限变化，三阶矩阵无法支撑241个独立方向。n行数大于p只是必要条件，不保证满秩。
轨迹干净导数lag-1相关中位数为0.9999999509，全局相位同步序参量最大0.109298。低全局序参量不足以证明没有局部同步；本次秩缺陷有直接矩阵证据，无需将其归因于全局同步。
独立性解释须区分：上述lag-1是干净导数的相关性，不是回归噪声或模型误差的独立性检验；它不能单独证明BIC的独立误差假设被违反。本轮加入的响应噪声按定义是独立的，高阶截断误差是否产生相关残差仍未验证；数值秩亏则已可由矩阵直接确定。
单拓扑的筛选召回不支持方法优越性判断，尤其当前动态采样造成强相关/低激励；不应拿它与旧独立快照结果直接比较。

## 6. 神经smoke与资源

|架构|参数量|训练步骤|训练/验证/测试流程耗时(s)|N30推断(s，406候选)|源交换误差|
|---|---|---|---|---|---|
|light|1681|3|0.04537|0.04147|0|
|transformer|3905|3|0.10843|0.16744|0|

统计首轮调度区间墙钟11.747秒（含数据生成和报告，不含Python导入/协议初始化），峰值RSS约175.39 MiB；神经训练整体峰值RSS约601.31 MiB。GPU没有使用，显存NA；无OOM/超时/数值求解异常，BIC NA单列统计。
这不是有效全链条结构恢复的速度基准：回归在数值前提不满足时提前拒绝。不能从约十秒外推N1000成本。
断点重跑已验证：602个科学artifact哈希保持不变，新增后端拟合0个。原始墙钟保存在statistical_first_completion.json；statistical_final/completion.json为最近一次resume调用的完成状态/耗时。

## 7. 文件清单和复现

全部新增代码文件：

- experiments/large_scale_screening_v1/AUDIT_AND_PLAN.md
- experiments/large_scale_screening_v1/README.md
- experiments/large_scale_screening_v1/__init__.py
- experiments/large_scale_screening_v1/backend.py
- experiments/large_scale_screening_v1/candidates.py
- experiments/large_scale_screening_v1/cli.py
- experiments/large_scale_screening_v1/common.py
- experiments/large_scale_screening_v1/configs/default.json
- experiments/large_scale_screening_v1/data.py
- experiments/large_scale_screening_v1/library.py
- experiments/large_scale_screening_v1/metrics.py
- experiments/large_scale_screening_v1/neural.py
- experiments/large_scale_screening_v1/report.py
- experiments/large_scale_screening_v1/screeners.py
- experiments/large_scale_screening_v1/tests/test_system.py

没有修改历史生产文件；数据和结果仅写在新版本目录。
有效结果入口：statistical_final/REPORT.md、network_trials.csv、target_trials.csv、summary.csv；新增rank诊断为matrix_diagnostics.csv。测试/完整性证据为verification.json及三份测试日志。
精确CLI及默认参数见experiments/large_scale_screening_v1/README.md。复现请为输出目录选择新名称；本交付实际使用statistical_final、neural_final与neural_n30_final。

## 8. 下一步建议（本轮未执行）

1. **先处理N30数据激励与有效轨迹数，再进入P2。** 在新版本协议中评估增加独立初值轨迹、降低同轨迹冗余或增加总预算；保留原BIC门槛，先确认库的秩/条件，不能为得到F1而放宽门槛。
2. 对N1000先明确原BIC所需样本/库政策。当前条件数学上不适用；只运行筛选资源诊断不等于完整结构恢复可用。
3. Uniform Shared Pool在N1000的目标—候选级期望池召回约2048/498501=0.41%，必须先实测池召回；否则最终结果主要反映池丢失而非六种评分能力。本版作为无标签探索下界，不能声称高召回池已经实现。
4. P2前完成正式神经训练及同预算轻量架构对照；3步smoke不能证明学习优势或跨规模泛化。
5. 满足前提后再按README的两种子→30配对拓扑→大规模流程执行；P4以拓扑为bootstrap簇，CI和失败率分开报告。当前一个拓扑没有有效CI。
