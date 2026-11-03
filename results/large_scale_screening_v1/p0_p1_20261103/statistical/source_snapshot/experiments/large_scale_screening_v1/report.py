"""Network-level, topology-clustered reports; unavailable fits stay unavailable."""
import collections
import json
import math
from pathlib import Path
import numpy as np
from .common import save,table,digest
from .metrics import sparse_metrics


def trial_rows(inputs):
    rows={}
    for source in inputs:
        for p in Path(source).glob('n*/topology_*/noise_*/*/k*/target_*.json'):
            if p.stem.endswith('_selection_locked'):continue
            r=json.loads(p.read_text());r['source_run']=str(source)
            key=tuple(r.get(k) for k in ['N','topology_seed','trajectory_seed','noise','method','K','mode','sample_budget','target'])
            if key in rows:raise ValueError('Duplicate paired trial; do not mix pilot and formal reruns')
            rows[key]=r
    return list(rows.values())


def report(inputs,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=trial_rows(inputs)
    table(out/'target_trials.csv',({k:v for k,v in r.items() if k not in ['directed_scores','candidate_group_columns','target_structure']} for r in rows))
    groups=collections.defaultdict(list)
    for r in rows:groups[tuple(r.get(k) for k in ['N','topology_seed','trajectory_seed','noise','method','K','mode','sample_budget'])].append(r)
    networks=[]
    for key,values in sorted(groups.items(),key=str):
        base=dict(zip(['N','topology_seed','trajectory_seed','noise','method','K','mode','sample_budget'],key));n=base['N']
        metadata=Path(values[0]['source_run'])/f'n{n}'/f'topology_{base["topology_seed"]}_trajectory_{base["trajectory_seed"]}'/'data'
        with np.load(metadata/'data.npz',allow_pickle=False) as d:
            truth={2:set(map(tuple,d['net_pair_edges'])),3:set(map(tuple,d['net_triple_edges']))}
        ok=[r for r in values if r['status']=='OK'];status_counts=dict(collections.Counter(r['status'] for r in values))
        base.update(status_counts=json.dumps(status_counts,sort_keys=True),available_targets=len(ok),expected_targets=n,
            mean_Recall_at_K=float(np.mean([r['Recall_at_K'] for r in values if r.get('Recall_at_K') is not None])) if any(r.get('Recall_at_K') is not None for r in values) else None,
            mean_Precision_at_K=float(np.mean([r['Precision_at_K'] for r in values if r.get('Precision_at_K') is not None])) if any(r.get('Precision_at_K') is not None for r in values) else None,
            mean_pool_recall=float(np.mean([r['pool_recall'] for r in values if r.get('pool_recall') is not None])) if any(r.get('pool_recall') is not None for r in values) else None,
            mean_screen_AP=float(np.mean([r['full_pipeline_screen_AP'] for r in values if r.get('full_pipeline_screen_AP') is not None])) if any(r.get('full_pipeline_screen_AP') is not None for r in values) else None,
            pipeline_seconds=sum(r.get('pipeline_seconds',0) for r in values),
            screening_seconds=sum(r.get('screening_seconds',0) for r in values),
            feature_seconds=sum(r.get('feature_seconds',0) for r in values),
            regression_seconds=sum(r.get('regression_seconds',0) for r in values),
            cpu_peak_rss_bytes=max([r.get('cpu_peak_rss_bytes',0) for r in values]+[0]),
            failure_rate=sum(r['status'] in ['OOM','TIMEOUT','NUMERICAL_FAILURE'] for r in values)/len(values),
            NA_rate=sum(r['status']=='NA' for r in values)/len(values))
        for order in [2,3]:
            rec=dict(base,order=order)
            if len(ok)!=n:
                rec.update(status='NA',reason='Full-network structure unavailable: not every target returned an accepted BIC model',
                    F1=None,AP=None,MCC=None,Jaccard=None)
            else:
                scores={}
                for r in ok:
                    for item in r['directed_scores'][str(order)]:
                        edge=tuple(sorted(item['key']));scores[edge]=max(scores.get(edge,0),item['score'])
                rec.update(status='OK',reason='',**sparse_metrics(truth[order],scores,math.comb(n,order)))
            networks.append(rec)
    table(out/'network_trials.csv',networks)
    summary=[];rng=np.random.default_rng(2026110309)
    aggregate=collections.defaultdict(list)
    for r in networks:
        if r['order']==3:aggregate[tuple(r[k] for k in ['N','noise','method','K','mode','sample_budget'])].append(r)
    columns=['mean_Recall_at_K','mean_screen_AP','mean_pool_recall','F1','AP','MCC','Jaccard','pipeline_seconds','cpu_peak_rss_bytes','failure_rate','NA_rate']
    for key,records in sorted(aggregate.items(),key=str):
        result=dict(zip(['N','noise','method','K','mode','sample_budget'],key));topos=sorted(set(r['topology_seed'] for r in records));result['topologies']=len(topos)
        for metric in columns:
            # Mean across trajectory repeats within each topology; bootstrap topologies.
            cluster=[np.mean([r[metric] for r in records if r['topology_seed']==t and r.get(metric) is not None]) for t in topos if any(r.get(metric) is not None for r in records if r['topology_seed']==t)]
            result[metric+'_mean']=float(np.mean(cluster)) if cluster else None
            if len(cluster)>=2:
                draws=np.asarray(cluster)[rng.integers(0,len(cluster),(1000,len(cluster)))].mean(axis=1)
                result[metric+'_CI95_low'],result[metric+'_CI95_high']=np.quantile(draws,[.025,.975]).tolist()
            else:
                result[metric+'_CI95_low']=result[metric+'_CI95_high']=None
            result[metric+'_CI_reason']='' if len(cluster)>=2 else 'fewer than 2 valid topology clusters; no CI'
        summary.append(result)
    table(out/'summary.csv',summary)
    # Pair before reducing trajectory repeats; never compare unmatched seeds.
    baseline={tuple(r[k] for k in ['N','topology_seed','trajectory_seed','noise','K','mode','sample_budget']):r for r in networks if r['order']==3 and r['method']=='pearson'}
    paired=[]
    for r in networks:
        if r['order']!=3 or r['method']=='pearson':continue
        key=tuple(r[k] for k in ['N','topology_seed','trajectory_seed','noise','K','mode','sample_budget']);b=baseline.get(key)
        for metric in ['mean_Recall_at_K','F1','AP','pipeline_seconds']:
            valid=b is not None and b.get(metric) is not None and r.get(metric) is not None
            paired.append(dict(**{k:r[k] for k in ['N','topology_seed','trajectory_seed','noise','K','mode','sample_budget','method']},metric=metric,
                difference_vs_Pearson=r[metric]-b[metric] if valid else None,reason='' if valid else 'unavailable or unmatched paired observation'))
    table(out/'paired_differences.csv',paired)
    paired_groups=collections.defaultdict(list)
    for r in paired:paired_groups[tuple(r[k] for k in ['N','noise','K','mode','sample_budget','method','metric'])].append(r)
    comparisons=[]
    for key,records in paired_groups.items():
        cluster=[np.mean([r['difference_vs_Pearson'] for r in records if r['topology_seed']==t and r['difference_vs_Pearson'] is not None]) for t in set(r['topology_seed'] for r in records) if any(r['difference_vs_Pearson'] is not None for r in records if r['topology_seed']==t)]
        rec=dict(zip(['N','noise','K','mode','sample_budget','method','metric'],key));rec['paired_topologies']=len(cluster);rec['mean_difference']=float(np.mean(cluster)) if cluster else None
        if len(cluster)>=2:
            boot=np.array(cluster)[rng.integers(0,len(cluster),(1000,len(cluster)))].mean(axis=1);rec['CI95_low'],rec['CI95_high']=np.quantile(boot,[.025,.975]).tolist();rec['reason']=''
        else:rec.update(CI95_low=None,CI95_high=None,reason='fewer than 2 valid paired topology clusters')
        comparisons.append(rec)
    table(out/'paired_summary.csv',comparisons)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    metrics=['mean_Recall_at_K','F1','AP','pipeline_seconds','cpu_peak_rss_bytes']
    for ax,metric in zip(axes.flat,metrics):
        curves=collections.defaultdict(list)
        for r in summary:curves[(r['method'],r['noise'],r['K'],r['mode'],r['sample_budget'])].append(r)
        for key,rs in curves.items():
            rs=sorted(rs,key=lambda x:x['N']);valid=[r for r in rs if r[metric+'_mean'] is not None]
            if valid:ax.plot([r['N'] for r in valid],[r[metric+'_mean'] for r in valid],marker='o',label=str(key))
        ax.set(xlabel='N',ylabel=metric);ax.grid(alpha=.2)
        if not ax.lines:ax.text(.5,.5,'NA: no accepted full-network fits',ha='center',transform=ax.transAxes)
    for method in sorted(set(r['method'] for r in rows)):
        rs=[r for r in summary if r['method']==method]
        if rs:axes.flat[5].scatter([r['N'] for r in rs],[min(r['K'],math.comb(r['N']-1,2))/math.comb(r['N']-1,2) for r in rs],label=method)
    axes.flat[5].set(xlabel='N',ylabel='selected / full candidates')
    if axes.flat[0].lines:axes.flat[0].legend(fontsize=5)
    fig.tight_layout();fig.savefig(out/'scaling.png',dpi=160);fig.savefig(out/'scaling.pdf');plt.close(fig)
    lines=['# 大规模筛选系统：本次实际运行报告','',
        '只汇总输入目录中真实存在的完整目标记录；未运行规模不补点，NA不记为零F1。Full Scan、Shared Pool、固定/递增样本预算分别分组。',
        '', '|N|noise|K|mode|方法|候选召回均值|筛选AP均值|全网三体F1|全网三体AP|后端NA率|时间(s)|峰值RSS(MiB)|',
        '|---|---|---|---|---|---|---|---|---|---|---|---|']
    def fmt(v):return 'NA' if v is None else f'{v:.6g}'
    for r in summary:
        lines.append('|'+ '|'.join([str(r[k]) for k in ['N','noise','K','mode','method']]+[fmt(r[k+'_mean']) for k in ['mean_Recall_at_K','mean_screen_AP','F1','AP','NA_rate','pipeline_seconds']]+[fmt(None if r['cpu_peak_rss_bytes_mean'] is None else r['cpu_peak_rss_bytes_mean']/2**20)])+'|')
    reasons=collections.Counter(r.get('reason','') for r in rows if r['status']!='OK')
    lines+=['','## 数值状态及解释','']+[f'- {num} 个目标拟合：{reason}' for reason,num in reasons.items()]
    lines+=['','统计筛选得分是训练数据统计量，不能作为最终结构分数。最终AP使用连续原尺度系数；未选择项作为完整候选空间中的零分数组精确加权计算。',
        '若某目标BIC NA，不将缺失系数当作零模型；只要全网有目标无可接受模型，全网结构指标就是NA。',
        '默认每条轨迹256点，P1训练只有4条独立轨迹，不能把1024行当成1024次独立制备；这可能造成三阶设计矩阵秩亏/病态。时间自相关也限制原BIC的统计解释。',
        'N1000默认4096训练行小于强制库列数5998，故原后端必然NA；内存可运行不等于BIC数学条件成立。',
        'GC-CMI标记experimental：Gaussian-copula协同评分可漏掉对称纯乘积，也可能给可加非线性或代理节点非零分数，不是因果证明。',
        '神经网络若未作为benchmark方法运行，仅有训练/推断smoke结果，不能据此参与上述正式性能比较。',
        '', '## 资源与统计边界','',
        '时间按候选生成、公共预处理/数据生成按目标摊销、筛选、函数构造、回归累计，每个K作为独立完整pipeline计费；跨K缓存复用使实际总墙钟更低，见completion.json。',
        '公共投影只计算一次。峰值为10ms采样的进程绝对RSS，包含导入及共享缓存基线；短暂峰值可能漏采。GPU未使用，显存NA。',
        '每个规模只有一拓扑时不输出置信区间；summary.csv给出原因。P4先在同拓扑内平均轨迹重复，再对拓扑做1000次bootstrap；paired_summary.csv使用匹配的拓扑/轨迹差值。',
        '', '## 文件','',
        '- target_trials.csv：按目标候选召回、连续AP、状态、时间与资源。',
        '- network_trials.csv：无向方向最大值聚合后的二体/三体指标；NA保持缺失。',
        '- summary.csv、paired_differences.csv、paired_summary.csv：拓扑簇均值、CI及配对差异。',
        '- scaling.png / scaling.pdf：实际运行规模的六类曲线，没有生成未运行数据。',
        '- 每个目标筛选npz含候选对、连续得分和排名；成功拟合的npz含系数、原生d及训练尺度；selection_locked.json在测试评价前写入。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    save(out/'report_manifest.json',{p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name not in ['report_manifest.json','.lock','completion.json']})
    return summary
