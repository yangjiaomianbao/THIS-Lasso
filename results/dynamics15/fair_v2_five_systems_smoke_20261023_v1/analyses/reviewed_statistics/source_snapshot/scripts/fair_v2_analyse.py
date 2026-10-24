"""S7: analyse completed V2 runs in a new directory, without regenerating data or fits."""
import argparse
import json
import os
import shutil
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from src.experiments.fair_v2 import load_csv,sha,save_json,NAMES,METHODS
from src.experiments.fair_v2_statistics import export_statistics,METRICS,COMPARISONS
from src.bridge.numeric_io import write_rows
import tomllib
from fair_v2_verify import verify


def analyse(run,label):
    assert Path(label).name==label and label not in ('.','..')
    complete=json.loads((run/'completion.json').read_text())
    if complete['mode']=='probe':raise ValueError('Single-equation resource probes have no network statistics')
    audit=verify(run)
    config=tomllib.loads((run/'config_locked.toml').read_text())
    rows=load_csv(run/'trials.csv')
    for r in rows:
        r['noise_ratio']=float(r['noise_ratio'])
        for m in METRICS:
            r[m]=float(r[m]) if r[m] else None
    out=run/'analyses'/label;out.mkdir(parents=True,exist_ok=False)
    summary=export_statistics(out,rows,config,'FORMAL' if complete['mode']=='full' else 'PILOT')
    variance=[]
    for r in load_csv(out/'stability.csv'):
        if r['metric'] not in METRICS:continue
        s=next(x for x in summary if (x['system'],x['noise_ratio'],x['method'],x['order'])==
               (r['system'],float(r['noise_ratio']),r['method'],r['order']))
        metric=r['metric'];sd=s[metric+'_std']
        variance.append(dict(system=r['system'],noise_ratio=r['noise_ratio'],method=r['method'],order=r['order'],metric=metric,
            valid_instances=s[metric+'_n'],valid_networks=r['valid_networks'],
            overall_sample_variance=sd**2 if sd is not None else None,
            between_network_means_variance=float(r['between_network_mean_sd'])**2 if r['between_network_mean_sd'] else None,
            mean_within_network_sample_variance=float(r['within_network_rms_sd'])**2 if r['within_network_rms_sd'] else None))
    write_rows(out/'metric_variances.csv',variance)
    os.environ['MPLCONFIGDIR']=str(out/'mpl_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    systems=[s for s in config['systems'] if any(r['system']==s for r in rows)]
    for order in ('triple','pairwise'):
        fig,axes=plt.subplots(2,2,figsize=(15,9));groups=[(s,n) for s in systems for n in config['noise_ratios']]
        for ax,metric in zip(axes.ravel(),METRICS):
            for j,method in enumerate(METHODS):
                means=[];std=[]
                for s,n in groups:
                    r=next(r for r in summary if (r['system'],r['noise_ratio'],r['method'],r['order'])==(s,n,method,order))
                    means.append(r[metric+'_mean'] if r[metric+'_mean'] is not None else np.nan)
                    std.append(r[metric+'_std'] if r[metric+'_std'] is not None else np.nan)
                x=np.arange(len(groups))+(j-1)*.25
                ax.bar(x,means,.25,label=method)
                valid=np.isfinite(std)&np.isfinite(means)
                if valid.any():
                    ax.errorbar(x[valid],np.asarray(means)[valid],yerr=np.asarray(std)[valid],fmt='none',ecolor='black',capsize=2)
            ax.set_xticks(range(len(groups)),[f'{NAMES[s]} {n:.0%}' for s,n in groups],rotation=35,ha='right')
            ax.set_ylabel(metric+' (mean ± sample SD)');ax.legend(fontsize=7)
        fig.suptitle(f'{order}; validation-MSE selection; unavailable SD has no error bar')
        fig.tight_layout();fig.savefig(out/f'{order}_metrics.png',dpi=160);plt.close(fig)
    pairs=load_csv(out/'paired_summary.csv')
    fig,axes=plt.subplots(2,2,figsize=(15,9))
    for ax,metric in zip(axes.ravel(),METRICS):
        for j,(left,right) in enumerate(COMPARISONS):
            means=[];lo=[];hi=[]
            for s,n in groups:
                r=next(r for r in pairs if r['system']==s and float(r['noise_ratio'])==n and r['order']=='triple' and r['metric']==metric and r['comparison']==left+' - '+right)
                mean=float(r['cluster_mean']) if r['cluster_mean'] else np.nan
                means.append(mean);lo.append(mean-float(r['ci_low']) if r['ci_low'] else 0.)
                hi.append(float(r['ci_high'])-mean if r['ci_high'] else 0.)
            # Percentile intervals need not surround the sample effect; draw endpoints independently.
            x=np.arange(len(groups))+(j-1)*.20
            ax.scatter(x,means,label=left+' − '+right,s=15)
            for xv,m,l,h in zip(x,means,lo,hi):
                if l!=0 or h!=0:ax.vlines(xv,m-l,m+h,linewidth=1)
        ax.axhline(0,color='gray',lw=.7)
        ax.set_xticks(range(len(groups)),[f'{NAMES[s]} {n:.0%}' for s,n in groups],rotation=35,ha='right')
        ax.set_title(metric+' paired differences');ax.legend(fontsize=7)
    fig.suptitle('Triple: equally weighted network means; 95% network-bootstrap intervals where defined')
    fig.tight_layout();fig.savefig(out/'paired_differences.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(len(systems),2,figsize=(12,3*len(systems)),squeeze=False)
    for i,s in enumerate(systems):
        for j,n in enumerate(config['noise_ratios']):
            ax=axes[i,j]
            for method in METHODS:
                rs=[r for r in rows if r['system']==s and r['noise_ratio']==n and r['order']=='triple' and r['method']==method and r['status']=='OK']
                network_ids=sorted({int(r['network_id']) for r in rs})
                means=[np.mean([r['F1'] for r in rs if int(r['network_id'])==net]) for net in network_ids]
                ax.plot(network_ids,means,'o-',label=method)
            ax.set_title(f'{NAMES[s]} {n:.0%}');ax.set_xlabel('Network realization');ax.set_ylabel('Triple F1: data-replicate mean');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'network_stability.png',dpi=160);plt.close(fig)
    sources=[Path(__file__),ROOT/'src/experiments/fair_v2_statistics.py',ROOT/'scripts/fair_v2_verify.py']
    for source in sources:
        dest=out/'source_snapshot'/source.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    save_json(out/'analysis_provenance.json',dict(source_run=str(run),source_trials_sha256=sha(run/'trials.csv'),
        source_lock_sha256=sha(run/'selection_locked.json'),verification=audit,
        analysis_source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
        independent_networks=complete['independent_networks'],full_experiment_run=complete['full_experiment_run'],
        bootstrap_seed=config['bootstrap_seed'],bootstrap_resamples=config['bootstrap_resamples']))
    (out/'REPORT.md').write_text('\n'.join([
        '# V2统计分析：'+('正式实验' if complete['mode']=='full' else '小规模验证'),'',
        f'独立网络实现数：{complete["independent_networks"]}；数据重复数：{complete["independent_data_repetitions"]}。',
        '统一验证集MSE选择是为了公平预测比较，不保证结构F1最优。',
        'NA/FAILED不填0；先按完整网络计算指标，再按有效实例汇总，标准差ddof=1。',
        '配对比较匹配同一网络、数据种子及噪声；bootstrap重采样网络均值，网络等权。有效配对不足或某指标未定义时单独报告数量和原因。',
        '网络同时包含拓扑与边权随机性，网络间/网络内波动仅为描述统计，不能当作纯拓扑因果效应。',
        '正式设计只有10个独立网络，置信区间仍可能不稳定；当前若只有1个网络，则所有网络bootstrap区间为NA。','',
        '[三体表](tables/table1_threebody.md) · [二体表](tables/table2_pairwise.md) · [配对差异](paired_summary.csv) · [波动统计](stability.csv) · [显式方差](metric_variances.csv)','',
        '![三体指标](triple_metrics.png)','![二体指标](pairwise_metrics.png)',
        '![配对差异](paired_differences.png)','![网络稳定性](network_stability.png)','',
        '均值图的误差条仅在样本标准差有定义时显示；n=1缺少误差条不代表方差为0。',
        'FULL EXPERIMENT NOT RUN' if complete['mode']!='full' else '正式实验已完成；结论需结合有效网络数、失败比例和区间宽度。'])+'\n')
    print('ANALYSIS_COMPLETE',out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--label',required=True)
    a=p.parse_args();analyse(a.run,a.label)
