"""Actual-data summaries, topology bootstrap, paired contrasts and diagnostics plots."""
from collections import defaultdict,Counter
import csv
import json
import math
from pathlib import Path
import numpy as np
from ..common import save,table,digest


def mean(rows,key):
    vals=[r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])]
    return float(np.mean(vals)) if vals else None


def summarize(rows,keys,metrics):
    groups=defaultdict(list)
    for r in rows:groups[tuple(r[k] for k in keys)].append(r)
    result=[];rng=np.random.default_rng(2026110399)
    for key,rs in sorted(groups.items(),key=lambda item:str(item[0])):
        rec=dict(zip(keys,key));rec.update(topologies=len({r['seed_index'] for r in rs}),networks=len(rs),valid_networks=sum(r['status']=='OK' for r in rs))
        for metric in metrics:
            clusters=[mean([r for r in rs if r['seed_index']==s],metric) for s in sorted({r['seed_index'] for r in rs})]
            clusters=np.array([v for v in clusters if v is not None]);rec[metric]=float(np.mean(clusters)) if len(clusters) else None
            rec[metric+'_CI95_low']=rec[metric+'_CI95_high']=None
            if len(clusters)>=2:
                boot=clusters[rng.integers(0,len(clusters),size=(2000,len(clusters)))].mean(axis=1)
                rec[metric+'_CI95_low'],rec[metric+'_CI95_high']=np.quantile(boot,[.025,.975]).tolist()
        result.append(rec)
    return result


def markdown(rows,columns):
    def fmt(v):
        if v is None:return 'NA'
        if isinstance(v,float):return f'{v:.5g}'
        return str(v)
    return ['|'+'|'.join(columns)+'|','|'+'|'.join(['---']*len(columns))+'|']+['|'+'|'.join(fmt(r.get(k)) for k in columns)+'|' for r in rows]


def report(out):
    out=Path(out);net=[];targets=[];state=[];jobs=[];blind=[]
    for folder in sorted((out/'jobs').iterdir()):
        if not (folder/'completion.json').exists():continue
        job=json.loads((folder/'job.json').read_text());jobs.append(job)
        nr=json.loads((folder/'network_rows.json').read_text());tr=json.loads((folder/'target_rows.json').read_text())
        sd=json.loads((folder/'state_diagnostics.json').read_text());sd={k:v for k,v in sd.items() if k!='state_quantiles'}
        sd.update({k:job[k] for k in ['name','kind','N','seed_index','sampling_mode','delta','train_trajectories','duration']})
        if job['sampling_mode']=='local_iid':sd['train_trajectories']=0;sd['duration']=None
        state.append(sd)
        for r in nr:r.update({k:v for k,v in sd.items() if k not in r and k!='name'})
        net.extend(nr);targets.extend(tr)
        with (folder/'phase_blindness.csv').open() as f:
            for r in csv.DictReader(f):blind.append(dict(job=job['name'],seed_index=job['seed_index'],N=job['N'],**r))
    table(out/'target_trials.csv',targets);table(out/'network_trials.csv',net);table(out/'state_diagnostics.csv',state)
    table(out/'phase_blindness.csv',blind)
    metrics=['mean_Recall_at_K','mean_screen_AP','F1','AP','MCC','Jaccard','FP','FN','BIC_success_rate','rank_mean','condition_median','smin_median','pipeline_seconds']
    main=[r for r in net if r['kind']=='A'];primary=summarize(main,['N','noise','K','method'],metrics)
    table(out/'local_iid_summary.csv',primary)
    b1=[r for r in net if r['kind'] in ['B1','B1_extension']]
    b2=[r for r in net if r['kind']=='B2']+[r for r in b1 if r['trajectories']==4]
    b3=[r for r in net if r['kind']=='B3']+[r for r in main if r['N']==30 and r['seed_index']<2 and r['noise']==.05 and r['K']==16 and r['method'] in ['pearson','scc_ols','fwl']]
    ablation_metrics=metrics+['state_std_mean','state_stable_rank','covariance_min_eigenvalue','coordinate_range_mean','max_abs_fixed_center_offset','global_order_max','edge_temporal_PLV_mean','edge_phase_cos_mean','edge_frequency_gap_mean','derivative_lag1_within_trajectory','low_order_response_R2_mean','taylor3_relative_RMSE_mean']
    ts=summarize(b1,['trajectories','method'],ablation_metrics);ds=summarize(b3,['delta','method'],ablation_metrics);times=summarize(b2,['duration','method'],ablation_metrics)
    table(out/'trajectory_ablation.csv',ts);table(out/'delta_ablation.csv',ds);table(out/'time_ablation.csv',times)
    core=[r for r in main if r['N']==30 and r['seed_index']<2 and r['noise']==.05 and r['K']==16 and r['method'] in ['pearson','scc_ols','fwl']]+[r for r in b1 if r['trajectories']==4]
    cs=summarize(core,['sampling_mode','method'],ablation_metrics+['train_samples','independent_trajectories','independent_preparations'])
    table(out/'sampling_comparison.csv',cs)
    # Scale/SNR analysis is descriptive, never used to tune a model or select a center.
    excitation=[]
    from ..data import load_data
    from .sampling import coordinates,triadic_coefficients
    for folder in sorted((out/'jobs').iterdir()):
        if not (folder/'completion.json').exists():continue
        job=json.loads((folder/'job.json').read_text())
        if job['N']!=30 or job['seed_index']>=2 or job['sampling_mode']!='local_iid':continue
        d=load_data(folder/'data');x,_=coordinates(d);mask=d['splits']=='train';x=x[mask]
        noise_sd=.05*d['derivative'][mask].std(axis=0);ratios=[];vis=[];undirected=defaultdict(list)
        for c in triadic_coefficients(d['net'],json.loads((out/'protocol.json').read_text())['config']['data']):
            i,j,k=c['target'],c['j'],c['k'];signal=abs(c['quadratic_cross'])*np.std(x[:,j]*x[:,k])
            ratios.append(signal/noise_sd[i]);vis.append(c['relative_quadratic_visibility'])
            undirected[tuple(sorted((i,j,k)))].append(c['relative_quadratic_visibility'])
        excitation.append(dict(seed_index=job['seed_index'],delta=job['delta'],
            median_directed_quadratic_term_SD_to_noise_SD=float(np.median(ratios)),
            directed_relative_visibility_below_0_1=float(np.mean(np.array(vis)<.1)),
            undirected_all_directions_relative_visibility_below_0_1=float(np.mean([max(v)<.1 for v in undirected.values()]))))
    table(out/'quadratic_excitation.csv',excitation)
    full_quadratic(out)
    matrix_summary=[];center_summary=[]
    if (out/'matrix_spectrum.csv').exists():
        with (out/'matrix_spectrum.csv').open() as f:mr=list(csv.DictReader(f))
        groups=defaultdict(list)
        for r in mr:
            if r['center_mode']=='fixed' and r['scale_mode']=='standard':
                groups[tuple(r[k] for k in ['sampling_mode','trajectories','duration','delta','degree','columns'])].append(r)
        for key,rs in groups.items():
            rec=dict(zip(['sampling_mode','trajectories','duration','delta','degree','columns'],key))
            for metric in ['rank','condition','entropy_effective_rank','stable_rank','maximum_abs_term_correlation','low_order_response_R2','nonlinear_terms_low_order_R2']:
                values=[float(r[metric]) for r in rs if r.get(metric)]
                rec[metric]=float(np.median(values)) if values else None
            matrix_summary.append(rec)
        groups=defaultdict(list)
        for r in mr:
            if r['degree']=='3' and (r['job'].startswith('A_') or r['job'].startswith('B1_') and r['trajectories']=='4'):
                groups[(r['sampling_mode'],r['center_mode'],r['scale_mode'])].append(r)
        for key,rs in groups.items():
            rec=dict(zip(['sampling_mode','center_mode','scale_mode'],key))
            for metric in ['rank','condition','rank_rtol_1e-08','rank_rtol_1e-12','rank_rtol_1e-14']:
                rec[metric]=float(np.median([float(r[metric]) for r in rs]))
            center_summary.append(rec)
        table(out/'matrix_summary.csv',matrix_summary);table(out/'center_scale_sensitivity.csv',center_summary)
    baseline={(r['N'],r['seed_index'],r['noise'],r['K']):r for r in main if r['method']=='pearson'};paired=[]
    for r in main:
        if r['method']=='pearson':continue
        b=baseline[(r['N'],r['seed_index'],r['noise'],r['K'])]
        rec={k:r[k] for k in ['N','seed_index','noise','K','method','status']}
        for metric in ['mean_Recall_at_K','mean_screen_AP','F1','AP']:
            rec[metric+'_difference']=r[metric]-b[metric] if r.get(metric) is not None and b.get(metric) is not None else None
        paired.append(rec)
    table(out/'paired_differences.csv',paired)
    table(out/'paired_summary.csv',summarize(paired,['N','noise','K','method'],[k+'_difference' for k in ['mean_Recall_at_K','mean_screen_AP','F1','AP']]))
    plot(out,primary,cs,ts,times,ds)
    lines=['# THIS 采样机制与秩亏诊断 v1','',
        f'实际完成 {len(jobs)} 个数据任务、{len(targets)} 个目标拟合；状态为 {dict(Counter(r["status"] for r in targets))}。',
        '本报告只读真实完成记录。原库、五统计筛选器、Adaptive BIC配置/接受条件和结构解码全部复用；神经模型未参与。',
        '固定随机网络中心、频率、K2=.3/K3=.6；1024/256/256行。主实验采用固定net.center展开，旧接口保持训练圆均值；B4单独检查中心。']
    lines+=conclusions(primary,ts,times,ds,out)
    lines += [
        '## A：独立局部采样的完整对照','',
        'seed 0/1先作数值试验，满足预先写明的每个N/noise/K目标成功率≥95%才扩展到总计30个配对拓扑。前两个seed只计入一次。表内结构指标为全网指标，任何目标NA则该网络NA；均值及区间只能使用有效网络，valid_networks给出分母。',
        '置信区间按拓扑簇进行2000次bootstrap；两个诊断种子的区间只能描述这两个网络，不能作为充分总体证据。GC-CMI仍为experimental。','']
    lines+=markdown(primary,['N','noise','K','method','topologies','valid_networks','mean_Recall_at_K','mean_screen_AP','F1','AP','MCC','Jaccard','FP','FN','BIC_success_rate'])
    lines+=['','## 核心采样机制对照（N30、5%噪声、K16、同两网络）','',
        'local_iid为1024次独立制备，无真实时间或轨迹。trajectory为4条独立初值的256点RK45轨迹。实际分布同时变化，不能把两者性能差异全部归因于时间相关性；state_distribution_comparison.csv另给逐节点Wasserstein和KS距离。','']
    lines+=markdown(cs,['sampling_mode','method','train_samples','independent_trajectories','state_std_mean','covariance_min_eigenvalue','rank_mean','condition_median','mean_Recall_at_K','F1','BIC_success_rate'])
    lines+=['','## B1：固定1024行和T=.5，增加独立轨迹数','']
    lines+=markdown(ts,['trajectories','method','rank_mean','condition_median','smin_median','mean_Recall_at_K','F1','BIC_success_rate','state_stable_rank'])
    lines+=['','## B2：固定4条轨迹，改变演化时间','']
    lines+=markdown(times,['duration','method','max_abs_fixed_center_offset','rank_mean','condition_median','mean_Recall_at_K','F1','BIC_success_rate','global_order_max','edge_temporal_PLV_mean','edge_frequency_gap_mean','taylor3_relative_RMSE_mean'])
    lines+=['','## B3：固定底层抽样与单位噪声，改变局部宽度','']
    lines+=markdown(ds,['delta','method','rank_mean','condition_median','mean_Recall_at_K','F1','BIC_success_rate','taylor3_relative_RMSE_mean'])
    if (out/'row_shuffle_control.csv').exists():
        with (out/'row_shuffle_control.csv').open() as f:permutation=list(csv.DictReader(f))
        lines+=['','## 行打乱负控制','']+markdown(permutation,['seed_index','rank_before','rank_after','lag1_original_row_order','lag1_shuffled_row_order','spectrum_difference_relative_smax'])
        lines+=['','行置换不改变矩阵的奇异值，数值误差以最大奇异值为参照报告；尾部接近机器精度的相对误差或秩阈值跨越不表示新的信息。lag-1变化只是行序统计变化，不能证明激励改变。']
    lines+=['','## B4：嵌套函数库、中心、列尺度和秩阈值','',
        'matrix_spectrum.csv中固定原4轨迹FWL候选，分别分析常数+一阶、至二阶、至三阶；比较固定/训练圆均值/训练算术均值中心和原始/标准化尺度。每个条件检查目标0/1/2及两个网络。逐一奇异值保存在matrix_singular_values.npz；主拟合的每个目标完整谱保存在jobs/*/arrays.npz。',
        '默认数值秩阈值max(n,p)*eps*smax，另给1e-4至1e-14相对阈值下的秩、谱熵有效秩、stable rank、项相关性。训练列标准化与原backend一致，原BIC同时约束自适应加权库，单凭未加权满秩不保证通过。',
        '另补充不经过候选筛选的完整一阶+二阶库（N30共495个非恒定列），见full_quadratic_spectrum.csv及full_quadratic_singular_values.npz，覆盖全部两种子诊断数据。此库比当前稀疏三阶库还宽，因此作为独立诊断，不用于主回归；matrix_spectrum.csv的二阶条目指同一候选集合的嵌套二阶子库。',
        '完整二阶库在独立采样下满秩，而原4轨迹下同样严重退化。这支持数据激励与多项式函数族相互作用的解释，而不是库索引实现错误；仅减少最高阶数也未必解决问题，更宽的完整二阶库不能与较窄的稀疏三阶库只按阶数比较。',
        '低阶对响应的解释率与低阶对非线性函数项的解释率分别列出。解析完整动力学三阶截断误差独立于候选筛选和拟合估算；不把候选遗漏误差叫作Taylor截断误差。',
        '以下为固定候选、标准化、目标0/1/2×两种子的中位数，补充全部条件见matrix_summary.csv。','']
    representative=[r for r in matrix_summary if (r['sampling_mode']=='trajectory' and r['duration']=='0.5' and r['trajectories'] in ['4','16','32','64','512']) or (r['sampling_mode']=='local_iid' and r['delta']=='0.6')]
    lines+=markdown(representative,['sampling_mode','trajectories','delta','degree','columns','rank','condition','entropy_effective_rank','nonlinear_terms_low_order_R2'])
    lines+=['','中心/尺度敏感性（同候选；各值为三目标×两网络中位数）：','']
    lines+=markdown(center_summary,['sampling_mode','center_mode','scale_mode','rank','condition','rank_rtol_1e-08','rank_rtol_1e-12','rank_rtol_1e-14'])
    lines+=['','## 相位盲区与结构指标边界','',
        'phase_blindness.csv保存所有真实有向三体交互的局部二阶交叉系数；无向实际读出只有三方向二次系数的max，固定中心可能使某些交互弱可见。不能用真值系数挑中心、改候选或挑权重。',
        '原1e-8系数阈值未调优；未被选中的完整空间候选评分为0。筛选AP来自连续统计分数，结构AP来自原尺度连续二次系数，二者分开。',
        '局部同步使用真实边的组内时间相位锁定值和相位/频率差；短轨迹PLV高也可能只是缓慢演化，不能单独作为同步因果证据。',
        '干净导数的lag-1不是回归误差独立性检验。响应噪声按定义独立，但Taylor截断误差的相关性尚未验证；通过原BIC的数值门槛不能自动证明其误差统计假设成立。网络间bootstrap也不能替代这项验证。',
        '512×2条件每轨迹只有两个点，不能估计组内lag-1；新state_diagnostics字段对此明确保存NA。data/metadata.json保留原RK45生成器的旧诊断，其中两点轨迹的derivative_lag1_median=0是旧实现的占位结果，不能解读成无时间相关；本报告不使用它。',
        '## 复现与文件','',
        '命令见实现目录README.md；protocol.json、runtime.json、source_snapshot保存完整配置、环境和科研源码。每任务data/保留原始相位、解析导数、独立sample/trajectory ID、拆分、中心、真实图及频率；数据重建噪声使用冻结SeedSequence。',
        'jobs/*/arrays.npz保留所有评分/候选/库索引/奇异值/接受系数/路径；selection_locks/中在测试打分前保存模型。target_trials.csv、network_trials.csv可逐项复算，sampling_comparison/trajectory_ablation/time_ablation/delta_ablation为核心表，local_iid_summary/paired_summary带拓扑区间。',
        'pipeline_seconds累计筛选、函数构造、公共预处理和原后端拟合；各K按独立pipeline计费，共用评分时不能相加当作实际墙钟。该字段不含独立谱诊断、解析截断分析、测试打分和磁盘保存；每个jobs/*/completion.json的seconds另记录包含这些开销的任务墙钟。进程并行、每进程BLAS单线程，未使用GPU。',
        '本轮11项新增采样测试、25项历史科学测试（含Julia原生对照）、30项原锁定环境Adaptive测试均通过。independent_recalculation.json另记录294个全空间sklearn指标核对、1170个模型锁核对、1260个谱秩核对；原RK45数组和一次原后端复算系数均逐元素一致。',
        'verification.json记录已完成任务哈希和旧1993项交付文件的保护核验。旧历史文件的既有未提交删除不属于本实验。','']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    save(out/'report_manifest.json',dict(completed_jobs=len(jobs),target_fits=len(targets),
        main_topologies={str(n):len({r['seed_index'] for r in main if r['N']==n}) for n in [15,30]},
        status_counts=dict(Counter(r['status'] for r in targets)),report_source_sha256=digest(Path(__file__))))
    print('REPORT',out/'REPORT.md',flush=True)


def conclusions(primary,trajectory,times,delta,out):
    def find(rows,**criteria):return next((r for r in rows if all(r[k]==v for k,v in criteria.items())),None)
    def fmt(value):return 'NA' if value is None else f'{value:.4g}'
    lines=['','## Q1：独立局部采样能否恢复结构推断？','']
    for n in [15,30]:
        rs=[find(primary,N=n,noise=.05,K=16,method=m) for m in ['pearson','scc','scc_ols','fwl','gc_cmi']]
        if not all(rs):continue
        lines.append(f'N={n}、5%噪声、K16，已完成{rs[0]["topologies"]}个拓扑，五方法的三体F1依次为 '+
            '、'.join(f'{r["method"]}={fmt(r["F1"])}' for r in rs)+'。')
        f=rs[3]
        lines.append(f'FWL F1拓扑bootstrap 95%区间 [{fmt(f["F1_CI95_low"])}, {fmt(f["F1_CI95_high"])}]；其候选召回={fmt(f["mean_Recall_at_K"])}，BIC目标成功率={fmt(f["BIC_success_rate"])}。完整网的F1/AP用原二次系数读出，没有改库或BIC门槛。')
    lines+=['','结论是独立局部采样在本次网络/参数和样本预算下恢复了可求解性与较好的结构恢复；这是新基准的实测结果，不是与旧独立快照实验的逐文件等价复现，也不代表所有统计方法同样有效。',
        '','## Q2：连续轨迹的数值病态主要来自什么？','',
        '同图、同中心、同总行数的B1表明增加独立初值能改善设计；满秩和可接受条件数必须区分：']
    for g in [4,16,32,64,512]:
        r=find(trajectory,trajectories=g,method='fwl')
        if r:lines.append(f'- {g}条轨迹：目标平均秩={fmt(r["rank_mean"])}/241，目标条件数中位数的网络均值={fmt(r["condition_median"])}，BIC成功率={fmt(r["BIC_success_rate"])}，全网F1={fmt(r["F1"])}。')
    lines+=['','增加初值多样性的控制支持“可访问状态/多项式方向激励不足”这一机制。四条光滑短轨迹的高阶函数项近乎落入低阶空间；B4固定候选后仍然如此，因此不能把问题全推给筛选选错候选。少数曲线在精确数学中并不必然给出低代数秩，此处结论是浮点数值病态。',
        '行打乱使行序lag-1接近零而谱不变，排除了“lag-1数值本身直接决定矩阵秩”的解释。lag-1是平滑/冗余的伴随诊断，不是秩亏的直接数学原因。']
    t=find(times,duration=5.,method='fwl');small=find(delta,delta=.2,method='fwl');large=find(delta,delta=1.5,method='fwl')
    if t:lines.append(f'把4条轨迹延长到T=5仍为BIC成功率{fmt(t["BIC_success_rate"])}，解析三阶截断相对RMSE增至{fmt(t["taylor3_relative_RMSE_mean"])}。演化时间能增加某些方向，却没有解决高阶病态，还明显偏离有效局部近似；全局序参量和局部边PLV/频率差不足以支持“同步是唯一原因”。')
    if small and large:lines.append(f'独立采样δ=.2至1.5的库均满秩且标准条件数基本不变；FWL F1从δ=.2时{fmt(small["F1"])}变化到δ=1.5时{fmt(large["F1"])}，同时大宽度解析截断相对RMSE达{fmt(large["taylor3_relative_RMSE_mean"])}。因此小δ的性能下降不能称作数值秩亏；quadratic_excitation.csv给出二次信号相对5%导数噪声的尺度，大δ则有明确截断误差代价。固定中心、固定候选、相同底层均匀样本下，纯单项式各列随δ按其次数齐次缩放，逐列标准化会消除这个缩放；二次项信号却约随δ平方变化，而定义中的响应噪声主要随一次响应尺度变化。这个尺度解释与实测信噪比和谱一致，不能把它等同于所有小范围采样都不会病态。')
    lines+=['','这些是受控参数组合下的机制证据。B1同时改变每轨迹点数与独立初值数，特别512×2是两端点网格；不同机制的实际边际/联合状态分布也不同，不能从这些试验宣称一个与分布无关的普适因果定律。',
        '','## Q3：不放宽BIC，真实轨迹需要怎样的激励？','',
        '在本次两个N30网络、T=.5、δ=.6初值宽度、1024行、5%导数噪声、K16条件下，测试网格中64条×16点是首次B1三种统计方法的全部目标通过原BIC的点，32条×32点仍拒绝。它只是“数值可解”的观察边界，结构F1仍低，不能当作可接受结构恢复。']
    r=find(trajectory,trajectories=512,method='fwl')
    if r:lines.append(f'追加控制中512条×2点时FWL的平均F1={fmt(r["F1"])}（两个网络各自值见network_trials.csv）；这给出已实测的较好恢复条件，不能把512称为理论最小值。128/256也完整保留，包括其较差结果。')
    lines+=['达到可解需要使实际采样的高阶库及自适应加权库都通过原1e8条件数等门槛；达到较好恢复还需要足够独立初值、目标外二次信号相对噪声可见、候选召回和局部Taylor误差受控。仅增加同曲线采样密度或仅延长T没有得到这类结果。实际应用应增加独立制备/扰动初值，在原门槛下检查谱和候选信噪比，而不是只追求行数或较低lag-1。','']
    return lines


def full_quadratic(out):
    import itertools
    from ..data import load_data
    from .sampling import coordinates
    from .diagnostics import design_diagnostics
    from threadpoolctl import threadpool_limits
    rows=[];arrays={}
    with threadpool_limits(limits=1):
        for folder in sorted((out/'jobs').iterdir()):
            if not (folder/'completion.json').exists():continue
            job=json.loads((folder/'job.json').read_text())
            if job['N']!=30 or job['seed_index']>=2:continue
            d=load_data(folder/'data');x,_=coordinates(d);terms=[()]+[(i,) for i in range(30)]+list(itertools.combinations_with_replacement(range(30),2))
            info,s,_=design_diagnostics(x[d['splits']=='train'],terms)
            rows.append(dict(job=job['name'],seed_index=job['seed_index'],sampling_mode=job['sampling_mode'],
                trajectories=job['train_trajectories'] if job['sampling_mode']=='trajectory' else 0,
                delta=job['delta'] if job['sampling_mode']=='local_iid' else None,
                duration=job['duration'] if job['sampling_mode']=='trajectory' else None,
                library='constant + ALL linear + ALL quadratic, not used in production fits',**info))
            arrays[job['name']]=s
    table(out/'full_quadratic_spectrum.csv',rows);np.savez_compressed(out/'full_quadratic_singular_values.npz',**arrays)


def plot(out,primary,comparison,trajectories,times,deltas):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for method in ['pearson','scc_ols','fwl']:
        for ax,rows,xkey,ykey in [(axes[0,0],trajectories,'trajectories','rank_mean'),(axes[0,1],trajectories,'trajectories','BIC_success_rate'),(axes[0,2],trajectories,'trajectories','F1'),(axes[1,0],times,'duration','condition_median'),(axes[1,1],deltas,'delta','F1'),(axes[1,2],deltas,'delta','taylor3_relative_RMSE_mean')]:
            rs=sorted([r for r in rows if r['method']==method and r.get(ykey) is not None],key=lambda r:r[xkey])
            if rs:ax.plot([r[xkey] for r in rs],[r[ykey] for r in rs],'.-',label=method)
            ax.set(xlabel=xkey,ylabel=ykey);ax.grid(alpha=.2)
    axes[1,0].set_yscale('log');axes[1,2].set_yscale('log');axes[0,0].legend();fig.tight_layout()
    for ax in axes[0]:
        ax.set_xscale('log',base=2);ax.set_xticks([4,8,16,32,64,128,256,512]);ax.set_xticklabels(['4','8','16','32','64','128','256','512'])
    fig.suptitle('N30, 1024 train samples, 5% noise, K16; two paired topologies (descriptive)',fontsize=11)
    fig.tight_layout(rect=[0,0,1,.95])
    fig.savefig(out/'diagnostics.png',dpi=160);fig.savefig(out/'diagnostics.pdf');plt.close(fig)
    if (out/'matrix_singular_values.npz').exists():
        with np.load(out/'matrix_singular_values.npz') as arrays:
            fig,axes=plt.subplots(1,2,figsize=(12,4))
            for seed,ax in enumerate(axes):
                for key in arrays.files:
                    if f'_s{seed:02d}_' not in key or not key.endswith('_i0_fixed_standard_degree3'):continue
                    if key.startswith('A_') or ('_g4_t0.5' in key and key.startswith('B1_')) or '_g64_' in key:
                        s=arrays[key];ax.semilogy(np.arange(1,len(s)+1),s/s[0],label=key.split('_i0_')[0])
                ax.axhline(1024*np.finfo(float).eps,color='k',linestyle=':',label='default relative rank threshold')
                ax.set(xlabel='singular value index',ylabel='s / smax',title=f'paired seed {seed}');ax.legend(fontsize=6);ax.grid(alpha=.2)
            fig.tight_layout();fig.savefig(out/'singular_spectra.png',dpi=160);fig.savefig(out/'singular_spectra.pdf');plt.close(fig)
