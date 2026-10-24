"""Network-level summaries and paired, network-clustered uncertainty for V2."""
from collections import defaultdict, Counter
from itertools import combinations
from pathlib import Path
import math
import numpy as np
from src.bridge.numeric_io import write_rows
from .fair_v2 import METHODS,NAMES,save_json

METRICS=['F1','AP','MCC','Jaccard']
COMPARISONS=[('Adaptive_val','LASSO_val'),('LASSO_val','STLSQ_val'),('Adaptive_val','STLSQ_val')]


def value(row,metric):
    if row['status']!='OK':return None
    if metric=='MCC' and row.get('MCC_status')!='OK':return None
    try:x=float(row[metric])
    except (TypeError,ValueError,KeyError):return None
    return x if math.isfinite(x) else None


def describe(values):
    a=np.asarray([x for x in values if x is not None],float)
    return dict(n=len(a),mean=float(a.mean()) if len(a) else None,
                std=float(a.std(ddof=1)) if len(a)>1 else None)


def clustered_bootstrap(groups, seed, resamples=2000):
    """Resample NETWORKS, each contributing its mean of valid within-network pairs.

    Inference targets an equally weighted network-realization effect. Separate
    pooled pair means are also reported when missingness creates unequal sizes.
    """
    keys=sorted(k for k,v in groups.items() if len(v))
    means=np.array([np.mean(groups[k]) for k in keys])
    if len(keys)<2:
        return dict(cluster_mean=float(means[0]) if len(keys) else None,
                    ci_low=None,ci_high=None,networks=len(keys),bootstrap_status='NA_less_than_two_networks',resamples=resamples)
    rng=np.random.default_rng(seed)
    draws=rng.integers(0,len(keys),size=(resamples,len(keys)))
    bootstrap=means[draws].mean(axis=1)
    lo,hi=np.quantile(bootstrap,[.025,.975])
    return dict(cluster_mean=float(means.mean()),ci_low=float(lo),ci_high=float(hi),
                networks=len(keys),bootstrap_status='OK',resamples=resamples)


def summarise(trials,seed=2026102302,resamples=2000):
    grouped=defaultdict(list);unique=set()
    for r in trials:
        uid=(r['system'],float(r['noise_ratio']),r['method'],r['order'],r['instance_id'])
        if uid in unique:raise ValueError('Duplicate experimental instance')
        unique.add(uid);grouped[uid[:4]].append(r)
    summary=[];stability=[]
    for (system,ratio,method,order),rows in grouped.items():
        states=Counter(r['status'] for r in rows)
        s=dict(system=system,noise_ratio=ratio,method=method,order=order,
               attempted=len(rows),OK=states['OK'],NA=states['NA'],FAILED=states['FAILED'],
               independent_networks=len({r['network_id'] for r in rows}))
        for m in METRICS+['TP','FP','FN','TN']:
            s.update({m+'_'+k:v for k,v in describe([value(r,m) for r in rows]).items()})
            bynet=defaultdict(list)
            for r in rows:
                v=value(r,m)
                if v is not None:bynet[r['network_id']].append(v)
            between=describe([np.mean(v) for v in bynet.values()])
            within=[float(np.var(v,ddof=1)) for v in bynet.values() if len(v)>1]
            stability.append(dict(system=system,noise_ratio=ratio,method=method,order=order,metric=m,
                valid_networks=len(bynet),between_network_mean_sd=between['std'],
                within_network_rms_sd=float(np.sqrt(np.mean(within))) if within else None,
                networks_with_repeated_data=len(within),
                interpretation='Descriptive spread; network varies topology AND weights; not pure variance components'))
        summary.append(s)
    pair_rows=[];pair_summary=[]
    outer=sorted({(r['system'],float(r['noise_ratio']),r['order']) for r in trials})
    index={(r['system'],float(r['noise_ratio']),r['order'],r['instance_id'],r['method']):r for r in trials}
    for group in outer:
        instances=sorted({r['instance_id'] for r in trials if (r['system'],float(r['noise_ratio']),r['order'])==group})
        for left,right in COMPARISONS:
            for metric in METRICS:
                valid=[];bynet=defaultdict(list);missing=Counter()
                for iid in instances:
                    a=index.get((*group,iid,left));b=index.get((*group,iid,right))
                    if a is None or b is None:raise ValueError('Missing model status row, cannot silently drop a pair')
                    assert all(a[k]==b[k] for k in ['network_seed','data_seed','noise_seed','dataset_sha256','noise_sha256'])
                    av,bv=value(a,metric),value(b,metric)
                    reason='' if av is not None and bv is not None else f'{left}:{a["status"]}/{a.get(metric+"_status", "")};{right}:{b["status"]}/{b.get(metric+"_status", "")};metric_undefined_or_model_unavailable'
                    delta=av-bv if not reason else None
                    if delta is not None:valid.append(delta);bynet[a['network_id']].append(delta)
                    else:missing[reason]+=1
                    pair_rows.append(dict(system=group[0],noise_ratio=group[1],order=group[2],
                        comparison=left+' - '+right,metric=metric,instance_id=iid,network_id=a['network_id'],
                        network_seed=a['network_seed'],data_seed=a['data_seed'],difference=delta,
                        status='OK' if delta is not None else 'NA',reason=reason))
                desc=describe(valid);boot=clustered_bootstrap(bynet,seed,resamples)
                pair_summary.append(dict(system=group[0],noise_ratio=group[1],order=group[2],
                    comparison=left+' - '+right,metric=metric,attempted_pairs=len(instances),
                    valid_pairs=desc['n'],missing_pairs=len(instances)-desc['n'],
                    pooled_pair_mean=desc['mean'],paired_sd=desc['std'],**boot,
                    missing_reasons=str(dict(missing))))
    return summary,stability,pair_rows,pair_summary


def display(mean,std,n):
    if mean is None:return f'NA (n={n})'
    return f'{mean:.4f} ± '+(f'{std:.4f}' if std is not None else 'NA')+f' (n={n})'


def export_statistics(out,trials,config,scope):
    out=Path(out)
    summary,stability,pairs,paired_summary=summarise(trials,config['bootstrap_seed'],config['bootstrap_resamples'])
    write_rows(out/'summary.csv',summary);write_rows(out/'stability.csv',stability)
    write_rows(out/'paired_differences.csv',pairs);write_rows(out/'paired_summary.csv',paired_summary)
    tables=out/'tables';tables.mkdir(exist_ok=True)
    for order,stem in [('triple','table1_threebody'),('pairwise','table2_pairwise')]:
        rows=[r for s in config['systems'] for noise in config['noise_ratios'] for m in METHODS
              for r in summary if (r['system'],r['noise_ratio'],r['method'],r['order'])==(s,noise,m,order)]
        write_rows(tables/(stem+'.csv'),rows)
        lines=[f'# {scope}: {order} reconstruction','',
            '均值 ± 样本标准差（ddof=1）；n为各指标有效重复数。粗体仅比较同系统/噪声的均值。',
            '验证集MSE选参用于公平预测比较，不保证结构F1最优。NA不填零。','',
            '| Dynamics | Noise | Method | F1 ↑ | AP ↑ | MCC ↑ | Jaccard ↑ | OK/attempted |',
            '| --- | --- | --- | --- | --- | --- | --- | --- |']
        for r in rows:
            vals=[]
            for metric in METRICS:
                label=display(r[metric+'_mean'],r[metric+'_std'],r[metric+'_n'])
                group=[a[metric+'_mean'] for a in rows if (a['system'],a['noise_ratio'])==(r['system'],r['noise_ratio']) and a[metric+'_n']]
                if r[metric+'_n'] and abs(r[metric+'_mean']-max(group))<=1e-12:label='**'+label+'**'
                vals.append(label)
            lines.append('| '+' | '.join([NAMES[r['system']],f'{r["noise_ratio"]:.0%}',r['method'],*vals,f'{r["OK"]}/{r["attempted"]}'])+' |')
        (tables/(stem+'.md')).write_text('\n'.join(lines)+'\n')
    return summary
