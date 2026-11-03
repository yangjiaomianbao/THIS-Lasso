# 大规模筛选系统：本次实际运行报告

只汇总输入目录中真实存在的完整目标记录；未运行规模不补点，NA不记为零F1。Full Scan、Shared Pool、固定/递增样本预算分别分组。

|N|noise|K|mode|方法|候选召回均值|筛选AP均值|全网三体F1|全网三体AP|后端NA率|时间(s)|峰值RSS(MiB)|
|---|---|---|---|---|---|---|---|---|---|---|---|
|30|0.05|16|full|fwl|0.0388889|0.0147783|NA|NA|1|1.65557|174.16|
|30|0.05|16|full|gc_cmi|0.0511905|0.0240298|NA|NA|1|1.54435|172.949|
|30|0.05|16|full|pearson|0.0337831|0.0255115|NA|NA|1|1.56653|174.191|
|30|0.05|16|full|scc|0.0148148|0.0195955|NA|NA|1|1.95853|174.273|
|30|0.05|16|full|scc_ols|0.0144841|0.0271389|NA|NA|1|1.97368|174.234|

## 数值状态及解释

- 150 个目标拟合：Rank-deficient or numerically unstable full design. Explicitly use selection="validation" with independent trajectory groups.

统计筛选得分是训练数据统计量，不能作为最终结构分数。最终AP使用连续原尺度系数；未选择项作为完整候选空间中的零分数组精确加权计算。
若某目标BIC NA，不将缺失系数当作零模型；只要全网有目标无可接受模型，全网结构指标就是NA。
默认每条轨迹256点，P1训练只有4条独立轨迹，不能把1024行当成1024次独立制备；这可能造成三阶设计矩阵秩亏/病态。时间自相关也限制原BIC的统计解释。
N1000默认4096训练行小于强制库列数5998，故原后端必然NA；内存可运行不等于BIC数学条件成立。
GC-CMI标记experimental：Gaussian-copula协同评分可漏掉对称纯乘积，也可能给可加非线性或代理节点非零分数，不是因果证明。
神经网络若未作为benchmark方法运行，仅有训练/推断smoke结果，不能据此参与上述正式性能比较。

## 资源与统计边界

时间按候选生成、公共预处理/数据生成按目标摊销、筛选、函数构造、回归累计，每个K作为独立完整pipeline计费；跨K缓存复用使实际总墙钟更低，见completion.json。
公共投影只计算一次。峰值为10ms采样的进程绝对RSS，包含导入及共享缓存基线；短暂峰值可能漏采。GPU未使用，显存NA。
每个规模只有一拓扑时不输出置信区间；summary.csv给出原因。P4先在同拓扑内平均轨迹重复，再对拓扑做1000次bootstrap；paired_summary.csv使用匹配的拓扑/轨迹差值。

## 文件

- target_trials.csv：按目标候选召回、连续AP、状态、时间与资源。
- network_trials.csv：无向方向最大值聚合后的二体/三体指标；NA保持缺失。
- summary.csv、paired_differences.csv、paired_summary.csv：拓扑簇均值、CI及配对差异。
- scaling.png / scaling.pdf：实际运行规模的六类曲线，没有生成未运行数据。
- 每个目标筛选npz含候选对、连续得分和排名；成功拟合的npz含系数、原生d及训练尺度；selection_locked.json在测试评价前写入。
