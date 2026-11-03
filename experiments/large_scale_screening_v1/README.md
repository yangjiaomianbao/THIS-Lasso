# 大规模 THIS 预筛选对比系统 v1

本目录为独立新增实现，先读 [审查与实施协议](AUDIT_AND_PLAN.md)。原有算法、结果、upstream、环境锁不修改。第一次只运行P0/P1；这里不宣称N1000已完成实验。

## 环境与启动

从仓库根目录执行，选择已安装NumPy/SciPy/scikit-learn/matplotlib/torch的现有Python。当前CPU即可运行；不需要安装/升级任何依赖。若默认Python不含torch，请先激活已有含torch环境。

```bash
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHON=python
EXP=experiments.large_scale_screening_v1.cli
BASE=results/large_scale_screening_v1/p0_p1_20261103
```

所有命令默认拒绝覆盖。运行同一冻结实验使用原命令追加`--resume`，源码、参数或checkpoint变化必须使用新目录。`run`不传out时自动生成时间标签。

## P0 / P1 复现命令

```bash
"$PYTHON" -B -m unittest discover -s experiments/large_scale_screening_v1/tests -v
# 如Julia未在PATH，给THIS_JULIA指定已有Julia可执行文件；否则原生parity测试会明确SKIP。
"$PYTHON" -B -m unittest discover -s tests -p 'test_r2_adaptive_lasso.py' -v
"$PYTHON" -B -m "$EXP" pilot --out "$BASE/coupling"
"$PYTHON" -B -m "$EXP" run --stage p1 --out "$BASE/statistical" --calibration "$BASE/coupling/coupling_pilot.json" --dry-run
"$PYTHON" -B -m "$EXP" run --stage p1 --out "$BASE/statistical" --calibration "$BASE/coupling/coupling_pilot.json"
"$PYTHON" -B -m "$EXP" train-neural --steps 3 --sizes 8 --topologies 2 --out "$BASE/neural"
"$PYTHON" -B -m "$EXP" neural-smoke --models "$BASE/neural" --data "$BASE/statistical/n30/topology_2026110310_trajectory_2026110410/data" --out "$BASE/neural_n30"
```

P1：N30，一拓扑种子/一轨迹种子，5%响应噪声，K16，五统计筛选方法对全部30目标的全部406候选评分；并用同一Adaptive BIC后端。神经3步只是训练/推断smoke，轻量/Transformer均跑，不作为正式预训练效果比较。`test_actual_native_julia_library_and_decode`调用真正的upstream库及原生解码，临时depot写到临时目录。

## P2 / P3 / P4 的准确 CLI（本轮不运行）

先训练正式的两个架构对照，checkpoint按独立验证拓扑的损失选择；此命令不包含任何benchmark数据。

```bash
MODEL=results/large_scale_screening_v1/neural_formal_v1
"$PYTHON" -B -m "$EXP" train-neural --steps 200 --sizes 30 50 --topologies 8 --out "$MODEL"
CAL="$BASE/coupling/coupling_pilot.json"
# 先两次配对重复，全噪声和K网格；dry-run不训练、不构造设计矩阵。
"$PYTHON" -B -m "$EXP" run --stage p2 --repeats 2 --checkpoint "$MODEL/transformer.pt" --calibration "$CAL" --out results/large_scale_screening_v1/p2_two --dry-run
"$PYTHON" -B -m "$EXP" run --stage p2 --repeats 2 --checkpoint "$MODEL/transformer.pt" --calibration "$CAL" --out results/large_scale_screening_v1/p2_two
# 确认后使用新的目录运行30个配对拓扑；不要把two与thirty重复样本一起汇总。
"$PYTHON" -B -m "$EXP" run --stage p2 --repeats 30 --checkpoint "$MODEL/transformer.pt" --calibration "$CAL" --out results/large_scale_screening_v1/p2_thirty
# 同一模型、同一coupling，N500/1000默认共享无标签探索池M2048。
"$PYTHON" -B -m "$EXP" run --stage p3 --repeats 2 --checkpoint "$MODEL/transformer.pt" --calibration "$CAL" --out results/large_scale_screening_v1/p3_pool --dry-run
"$PYTHON" -B -m "$EXP" run --stage p3 --repeats 2 --checkpoint "$MODEL/transformer.pt" --calibration "$CAL" --out results/large_scale_screening_v1/p3_pool
# 可选资源探测：只N500 Full Scan，与Shared Pool分开输出和统计。
"$PYTHON" -B -m "$EXP" run --stage p3 --sizes 500 --mode full --repeats 1 --checkpoint "$MODEL/transformer.pt" --calibration "$CAL" --out results/large_scale_screening_v1/p3_full500 --dry-run
# 固定样本预算对照，使用新目录；数据拆分仍为完整轨迹。
"$PYTHON" -B -m "$EXP" run --stage p2 --repeats 2 --fixed-samples --checkpoint "$MODEL/transformer.pt" --calibration "$CAL" --out results/large_scale_screening_v1/p2_fixed
# P4：只传真实已经完成的目录，缺失规模不会补画/伪造。
"$PYTHON" -B -m "$EXP" report --inputs results/large_scale_screening_v1/p2_thirty results/large_scale_screening_v1/p3_pool --out results/large_scale_screening_v1/p4_report
```

`--trajectories-per-topology 2`允许每个拓扑两组不同数据种子；ω与图在同一拓扑内不变。各方法使用相同拓扑、轨迹、噪声实现、候选池。`--mode full|pool|auto`和`--fixed-samples`属于实验条件，不能混为相同条件对照。`--config`可指定复制后的配置，但已开始的run拒绝配置变化。coupling文件必须与P1同源，跨N不重新调K2/K3；不同架构正式比较需用各自checkpoint的新run，并在版本/报告中区别架构。

## 数据/方法合同

- Python数组[T,N]；原相位不wrap，另存sin/cos、解析导数、组ID、时间、无向真实结构、有向ODE边列表、度与ω。d索引为原生Julia1起、0填充，局部列位置0起。
- 训练圆中心产生局部主值坐标；相位跨分支风险与每节点训练圆集中度记录。统计方法使用训练标准化局部坐标，神经用sin/cos；低阶残差B=[1,所有一次项]。
- `Screener.score(X,dX,target_node,candidate_pairs)`不接受真值。返回连续分数、canonical候选、排名、参数、时间和采样RSS；交换j/k不变。Context仅缓存训练统计量。
- SCC-OLS在各组进行带截距斜率计算，与SCC相关差严格区分；均按训练尺度标准化。GC-CMI始终experimental；通过纯乘积失败边界测试不等于该方法能识别纯乘积。
- 候选到三阶函数组4列：xj*xk、xj²*xk、xj*xk²、xi*xj*xk；统一保留所有单节点1–3次项及目标-源1–3次混合项，P=6N−2+4K。不使用已知正弦耦合关系、真值特征或Oracle。
- 后端只调用公共`fit_node('M2')`，外部训练列标准化，复用`restore_coefficients`，不改权重/BIC/KKT。每一列及组关系显式保存；只有原生eligible二次共源项直接解码三体。
- 目标方向保留连续abs系数，无向边/超边取各方向max。完整空间中未选择项为0，稀疏指标函数用加权零分组计算精确AP；pool内AP与全空间pipeline AP分开。
- 默认Shared Pool为**均匀随机无标签探索**，不同方法完全共享，不依赖Pearson；N1000池召回先验约2048/498501≈0.0041，不能把低池召回误归因于后续方法。

## 运行状态与限制

每目标返回OK、NA、NUMERICAL_FAILURE、OOM或TIMEOUT；绝不静默改为验证MSE。原BIC的n<=P、秩/条件数和方差底线全部保留。N1000默认必然n<=P；N30仅4条train轨迹可能使库秩亏/病态。因此这版可运行到大N的**筛选和资源诊断**，并不承诺原BIC在所有规模返回有效结构。

每次run包含源/配置/checkpoint哈希、源码快照、数据哈希、单元状态、逐目标候选npz、库映射、可接受模型系数、选参锁定记录、分层CSV、报告、六类曲线。中断保留已完成文件，`--resume`复用已完成目标，不重写旧科研结果。结构全网指标仅在所有目标均OK时计算；NA不能作为零F1模型纳入均值。

`limits`固定单任务120秒、地址空间12GiB、BLAS1线程；超时是软信号中断，正在执行的底层LAPACK可能延迟响应。RSS每10ms采样，包含公共缓存和导入基线，短峰值可能漏采。GPU不使用，显存NA。初始化/公共投影按目标摊销计费；跨K缓存命中节省的实际墙钟见completion。

连续轨迹残差有时间相关，数值BIC通过也不自动等于其独立残差统计假设通过。当前代码没有为得到非NA而偷偷改变默认点数或后端。P2前应先评估有效独立轨迹数及库的秩/条件；若修改样本预算，应预注册新版本。FWL精确投影和每目标多次SVD/LARS是主要规模瓶颈，分块避免的是内存爆炸，不是总计算量。
