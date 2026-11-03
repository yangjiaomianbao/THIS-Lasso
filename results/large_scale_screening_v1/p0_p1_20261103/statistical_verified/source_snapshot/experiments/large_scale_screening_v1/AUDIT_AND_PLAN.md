# P0：仓库审查与实施协议 v1（2026-11-03）

本文件先于实现生成。本轮仅 P0/P1；不启动 P2/P3 的正式扩量。用户任务原文授权本独立实验系统及必要新数据生成，旧结果、旧实现、upstream、锁文件均只读。

## 代码审查

|文件/位置|实际行为|复用/限制|
|---|---|---|
|src/preprocessing/pearson_filter.jl:candidate_neighborhoods|调用原生 keep_correlated，对状态—状态 Pearson 作全局排序，再形成局部邻域|不是本任务指定的目标导数边际相关。新版本标明 Pearson derivative-geomean，不覆盖旧实现|
|upstream/THIS/this-tools.jl:get_θd|递归生成0至dmax全部重复组合，含常数；完整三阶列数 C(N+3,3)|原生完整库在N=1000有167668501列；不能物化。新稀疏索引库保留显式monomial映射，并与原生小N列值逐项比对|
|src/preprocessing/taylor.py:restore_coefficients|只负责从训练均值/尺度恢复系数和偏置；不是构造Taylor库|直接复用|
|src/solvers/adaptive_lasso.py:_prepare/fit_adaptive_lasso|训练中心化；可选内部标准化；Ridge(svd)初值，w=(abs(beta_init)+eps)^(-gamma)，变换L1路径后恢复系数|直接调用公共fit_node('M2')，不复制回归代码；统一外部训练标准化、内部standardize=false，参数对全部筛选器相同|
|src/solvers/adaptive_lasso.py:_bic_variance/_select|全库OLS训练残差估计方差；LassoLarsIC原生BIC路径，原BIC df为活动系数数|n<=P、秩亏、条件数>1e8、方差低于相对底线均NA；数值/KKT失败单独FAILED，不回退验证MSE|
|src/solvers/methods.py:fit_node|M2为Adaptive+BIC稳定入口|直接复用；不调用本轮排除的STLSQ或普通LASSO|
|src/solvers/large_library.py|dual/primal Ridge、STLSQ及验证LASSO工作集；不是可替换的Adaptive BIC实现|只审查；不把其验证后端冒充BIC|
|src/this_adapter.jl:decode_coefficients|原生解码：源索引互异、目标不在源内、abs(coef)>1e-8；线性=>二体，源交叉二次=>三体|新实验读出保持此规则；三次重复或含目标项参与拟合但不直接解码三体|
|src/bridge/physical_decode.jl|仅为upstream one2dim的keys排序兼容补丁|不是任意Taylor项分组聚合器，不能把所有三次项都聚合为三体系数|
|scripts/dynamics30.py|硬编码30节点、配置种子和机内Julia路径，完整三阶库；K<=P时BIC NA|不能作为N=1000调度器。新目录统一N接口、稀疏边ODE、独立轨迹拆分|
|docs/data_contract.md、docs/REPRODUCE.md|Julia原始数组[N,T]，设计[T,P]；索引1起0填充；拆分按独立轨迹|新数据API明确[T,N]；导出d仍1起0填充。所有训练中心/尺度只看train|
|src/preprocessing/audited_screening.py|旧scc_ols实际为OLS残差后的组内相关差；FWL旧版为残差相关；旧pearson状态相关|保留。新SCC-OLS是组斜率差；FWL用增量RSS。命名带独立版本、不可合并旧结果|
|src/preprocessing/triadic_ndformer.py|旧适配网络含固定三阶矩加分且跨节点attention，训练数据为snapshot|不覆盖/复用旧checkpoint。新共享时序编码+对称融合；轻量与Transformer两架构，各用独立拓扑验证checkpoint|

审查文件没有可复用的N无关稀疏三阶候选组函数库。部分脚本固定15/30、原生解码本身N通用。公共solver、scale restoration、split验证可直接复用。现有工作区已有大量旧文件删除，原样保留，不恢复、不stage、不提交。

## 固定科学定义

- 无向稀疏二体图平均度4，无向三元超图平均关联度6；生成唯一边集O(N)，对ODE展开为目标有向边列表O(N)。每节点归一化，零度安全。ω在拓扑内固定。RK45，解析导数，原始相位与sin/cos保存。
- 每轨迹256点；不同初值轨迹整体分配train/validation/test。默认样本预算按用户表，另有固定1024/256/256预算模式。初相位围绕拓扑内固定随机中心独立扰动，避免把Taylor近似与任意长时间相位绕回混淆；记录同步序参量、局部坐标的分支风险、积分误差及轨迹自相关诊断。
- 统计输入为相位相对**训练圆均值**的主值偏移，训练标准化；神经输入为相位sin/cos及训练标准化目标导数。没有动力学参数/拓扑真值进入筛选函数。
- 公共低阶投影B=[1,所有一次状态偏移]，SVD精确列空间投影；不把平方/候选高阶作用先拟合走。SCC/SCC-OLS/FWL/GC-CMI复用同一残差。FWL score=(rᵀz_perp)^2/(z_perpᵀz_perp)，即单特征OLS的RSS减少量。
- Pearson= sqrt(abs(corr(y_i,Z_j))*abs(corr(y_i,Z_k)))。SCC=双向组内相关差绝对值均值；SCC-OLS=全局统一标准化后双向**带截距组斜率**差绝对值均值。中位分组，退化组返回0并计数。
- GC-CMI：先对残差和状态做秩高斯化，收缩相关矩阵，再计算对称 max(0,mean[I_G(r;Zj|Zk)-I_G(r;Zj)]))。该Gaussian-copula协同近似对独立对称纯乘积可能完全失败，且可加非线性/代理也可产生分数，故始终experimental；不是非参数CMI一致估计器，不采用I(Y;ZjZk|Zj,Zk)，不复用旧三阶矩法。
- 稀疏三阶THIS库：常数；所有节点一次/平方/立方；每个目标-源对的全部二变量混合项到3次（xi*xj, xi²*xj, xi*xj²）；对候选(j,k)追加(xj*xk,xj²*xk,xj*xk²,xi*xj*xk)。去重并映射原生单项式，候选组含4列，维数P=6N−2+4K。没有强加Kuramoto正弦系数约束；保留目标混合项与重复变量项以拟合低阶干扰。
- 解码仍只以xi响应的xj*xk系数判三体边，其余组列是拟合干扰项；不能把一组系数范数作为原THIS解码。二体取xj系数；无向边按各目标方向的连续绝对系数最大值聚合，额外保存每个方向。
- N<=200 Full Scan，按目标/候选块处理；N>=500默认Shared Pool，统一无标签均匀探索池M=2048，所有方法共享同一池。纯探索池不是性能优化承诺，预期池Recall=M/C(N−1,2)很低，必须实测报告，不能把条件池内AP冒充全空间AP。
- 全空间结构AP基于连续abs系数；未保留候选为0，并用零分数组的加权计数精确表示巨大候选全集，避免枚举N³；池内筛选AP另列，不混Full/Shared。

## 先验瓶颈与NA风险（不绕过）

1. N=1000、训练4096时，强制低阶/目标混合库已有5998列，任何K都触发原BIC K<=P，**本协议预期N1000后端NA**。可完成筛选/资源实验，但不能声称完成非NA结构恢复。若未来改变样本预算/库规则必须另版本，不能默改本轮。
2. 4条train轨迹×256点虽有1024行，却未必给三阶库足够秩；连续轨迹及相近初值可能令N30也NA。原BIC假设逐样本残差独立，轨迹时间相关时其统计解释有限，即使数值条件通过也不是已验证结构选择保证。
3. FWL精确投影约O(T*N*M)每目标，SVD/BIC在每个库的成本可能高于筛选；分块只保证内存有界，不承诺N1000快速。全扫描共N*C(N−1,2)候选。
4. 本机GPU驱动不可用，P1在已有含torch的Python环境CPU执行；不安装依赖、不升级环境。显存记录NA并注明原因。

## 实施顺序与文件

1. 本审查文档 -> configs/default.json、数据合同与API -> data.py/candidates.py/library.py/screeners.py -> backend.py复用M2。
2. neural.py：轻量/Transformer，对称融合，独立训练/验证/测试拓扑，checkpoint按验证损失；P1仅几步smoke，不声称已完成预训练。
3. tests：用户13类验收，特别是FWL-OLS恒等、GCMI失败边界、组斜率/相关区别、native Taylor/decode映射、原后端回归一致性、按轨迹拆分与块等价。
4. cli.py：dry-run、资源上限、唯一结果目录、哈希冻结/断点恢复；P1只N30一个拓扑/数据种子、5%响应噪声、K16，五统计方法全候选/全部30目标；神经接口独立smoke。完整噪声/K网格仅注册不启动。
5. coupling pilot只按数据动力学同步/相位偏移选择K2/K3（不看结构F1），冻结后N30运行；report.py输出实测CSV、NA/失败原因、资源、图与报告；未来P2/P3/P4提供CLI但不启动。
