"""Explicit staged CLI; process parallelism uses one BLAS thread per worker."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
import numpy as np
from threadpoolctl import threadpool_limits
from ..common import ROOT,save,table,digest,load_config,runtime_info,timeout
from ..candidates import blocks,count
from ..library import terms_and_groups,decode,evaluate,native_indices
from ..backend import fit
from ..screeners import Context,Screener,STAT_METHODS
from ..metrics import sparse_metrics,screening_metrics
from .sampling import generate,coordinates,triadic_coefficients,noisy_derivative,save_data
from .diagnostics import design_diagnostics,state_diagnostics,spectrum,lag1

HERE=Path(__file__).resolve().parent
DEFAULT_OUT=ROOT/'results/large_scale_screening_v1/sampling_diagnostics_v1'


def sources():
    paths=list(HERE.glob('*.py'))+list((HERE/'tests').glob('*.py'))+[HERE/'AUDIT.md']
    paths += [HERE.parent/name for name in ['data.py','library.py','screeners.py','backend.py','metrics.py','candidates.py','common.py','configs/default.json']]
    paths += [ROOT/name for name in ['src/solvers/adaptive_lasso.py','src/solvers/methods.py','src/preprocessing/taylor.py','src/data/contracts.py']]
    # Report generation is independently versioned and does not influence any fit.
    return {str(p.relative_to(ROOT)):digest(p) for p in sorted(paths) if p.name!='report.py'}


def protocol(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    config=load_config();config.update(version='sampling-diagnostics-v1',methods=list(STAT_METHODS))
    state=dict(config=config,science_sources=sources(),counts=[1024,256,256],
        center_mode='fixed net.center for both sampling mechanisms',
        primary_seeds=list(range(30)),pilot_seed_indices=[0,1],
        formal_gate='>=95% target BIC OK in every N/noise/K cell of the two-seed pilot',
        A=dict(sizes=[15,30],noise=[0.,.05],K=[16,32],delta=.6,methods=list(STAT_METHODS)),
        B=dict(N=30,seeds=[0,1],noise=[.05],K=[16],methods=['pearson','scc_ols','fwl'],
            trajectories=[4,8,16,32,64],durations=[.5,1.,2.,5.],delta=[.2,.4,.6,1.,1.5]),
        neural='excluded; local_iid never passed to temporal windows',
        statistics='topology is experimental unit; first two seeds are included once in 30',
        unchanged_BIC_config=config['backend'])
    p=out/'protocol.json'
    if p.exists():
        if json.loads(p.read_text())!=state:raise ValueError('Frozen science/config changed; use a new version/output')
    else:
        save(p,state);save(out/'runtime.json',runtime_info())
        for name in state['science_sources']:
            dest=out/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    return state


def make_job(kind,n,seed,mode='local_iid',delta=.6,trajectories=4,duration=.5,methods=None,noises=None,budgets=None):
    name=f'{kind}_n{n}_s{seed:02d}_{mode}_d{delta:g}_g{trajectories}_t{duration:g}'
    return dict(name=name,kind=kind,N=n,seed_index=seed,sampling_mode=mode,delta=delta,
        train_trajectories=trajectories,duration=duration,
        methods=list(methods or STAT_METHODS),noises=noises if noises is not None else [0.,.05],budgets=budgets or [16,32])


def verified_job(folder):
    marker=folder/'completion.json'
    if not marker.exists():return False
    state=json.loads(marker.read_text())
    for name,expected in state['hashes'].items():
        if not (folder/name).is_file() or digest(folder/name)!=expected:raise ValueError('Completed artifact changed: '+str(folder/name))
    return True


def execute(payload):
    job,out,config=payload;out=Path(out);folder=out/'jobs'/job['name']
    if verified_job(folder):return dict(name=job['name'],resumed=True)
    folder.mkdir(parents=True,exist_ok=True);save(folder/'job.json',job);started=time.perf_counter()
    with threadpool_limits(limits=1):
        n=job['N'];seed=job['seed_index'];top=config['seeds']['topology']+seed;sample=config['seeds']['trajectory']+100*seed
        data=generate(n,top,sample,config['data'],sampling_mode=job['sampling_mode'],delta=job['delta'],
            train_trajectories=job['train_trajectories'],duration=job['duration'])
        save_data(folder/'data',data);x,center=coordinates(data);train=data['splits']=='train';test=data['splits']=='test'
        save(folder/'state_diagnostics.json',state_diagnostics(data,config['data']))
        table(folder/'phase_blindness.csv',triadic_coefficients(data['net'],config['data']))
        np.save(folder/'expansion_center.npy',center)
        rows=[];networks=[];arrays={};model_locks=[]
        for ratio in job['noises']:
            y=noisy_derivative(data,ratio,config['seeds']['noise'],sample);context=Context(x[train],y[train]);combo={}
            for target in range(n):
                pairs=np.concatenate(list(blocks(n,target,config['screening']['block_size'])))
                truth={tuple(t[1:]) for t in data['net']['triples'] if t[0]==target}
                for method in job['methods']:
                    scored=Screener(method,config['screening'],context).score(x[train],y[train],target,pairs)
                    tag=f'r{ratio:g}_{method}_i{target}';arrays[tag+'_pairs']=pairs;arrays[tag+'_scores']=scored['scores'];arrays[tag+'_ranking']=scored['ranking']
                    for k in job['budgets']:
                        selected=pairs[scored['ranking'][:k]];terms,groups=terms_and_groups(n,target,selected)
                        diag,s,_=design_diagnostics(x[train],terms)
                        key=tag+f'_k{k}';arrays[key+'_spectrum']=s;arrays[key+'_d']=native_indices(terms);arrays[key+'_selected']=selected
                        row=dict(job=job['name'],kind=job['kind'],N=n,seed_index=seed,topology_seed=top,sample_seed=sample,
                            sampling_mode=job['sampling_mode'],delta=job['delta'] if job['sampling_mode']=='local_iid' else None,
                            trajectories=data['diagnostics']['train_trajectories'],duration=data['diagnostics']['duration'],
                            target=target,noise=ratio,method=method,K=k,
                            **screening_metrics(truth,pairs,scored['scores'],scored['ranking'],k,count(n)),
                            **diag,screening_seconds=scored['seconds'],common_preprocess_seconds=context.seconds/n)
                        try:
                            with timeout(config['limits']['task_seconds']):result=fit(x[train],y[train,target],terms,config['backend'])
                        except (TimeoutError,MemoryError,np.linalg.LinAlgError) as exc:
                            result=dict(status='TIMEOUT' if isinstance(exc,TimeoutError) else 'OOM' if isinstance(exc,MemoryError) else 'NUMERICAL_FAILURE',reason=str(exc))
                        row.update({key:value for key,value in result.items() if key not in ('coefficient','diagnostics','means','scales')})
                        decoded={2:{},3:{}}
                        if result['status']=='OK':
                            coef=result['coefficient'];details=result['diagnostics']
                            for name in ['coefficient','means','scales']:arrays[key+'_'+name]=result[name]
                            for name in ['initial_coef','weights','adaptive_scales','alpha_path','bic_path']:arrays[key+'_'+name]=np.asarray(details[name])
                            # Lock model identity and all selection evidence before held-out/truth evaluation.
                            lock=dict(key=key,coefficient_sha256=hashlib.sha256(coef.tobytes()).hexdigest(),
                                alpha=details['alpha'],objective=details['objective'],kkt_scaled_violation=details['kkt_scaled_violation'],
                                full_design_condition=details['full_design_condition'],weighted_design_condition=details['weighted_design_condition'],
                                full_model_residual_MSE=details['full_model_residual_MSE'],test_used_for_selection=False,
                                warnings=details['warnings'])
                            model_locks.append(lock);row.update(alpha=details['alpha'],weighted_condition=details['weighted_design_condition'],kkt=details['kkt_scaled_violation'])
                            # Each accepted model is persisted before evaluation, not merely described afterwards.
                            lockfile=folder/'selection_locks'/f'{key}.npz';lockfile.parent.mkdir(exist_ok=True)
                            np.savez_compressed(lockfile,coefficient=coef,d=native_indices(terms),means=result['means'],scales=result['scales'])
                            decoded=decode(coef,terms,target)
                            row['test_MSE_clean']=float(np.mean((evaluate(x[test],terms)@coef-data['derivative'][test,target])**2))
                            row.update({'target_'+m:v for m,v in sparse_metrics({(target,*p) for p in truth},decoded[3],count(n)).items() if m in ['F1','AP','MCC','Jaccard','FP','FN']})
                        combo.setdefault((method,k),[]).append((row,decoded))
                        rows.append(row)
            for (method,k),values in combo.items():
                rs=[r for r,_ in values];statuses=Counter(r['status'] for r in rs)
                rec={key:rs[0][key] for key in ['job','kind','N','seed_index','topology_seed','sample_seed','sampling_mode','delta','trajectories','duration','noise','method','K']}
                rec.update(targets=n,BIC_OK=statuses['OK'],BIC_NA=statuses['NA'],BIC_success_rate=statuses['OK']/n,
                    statuses=dict(statuses),status='OK' if statuses['OK']==n else 'NA',
                    mean_Recall_at_K=float(np.mean([r['Recall_at_K'] for r in rs if r['Recall_at_K'] is not None])),
                    mean_screen_AP=float(np.mean([r['full_pipeline_screen_AP'] for r in rs if r['full_pipeline_screen_AP'] is not None])),
                    rank_min=min(r['rank'] for r in rs),rank_mean=float(np.mean([r['rank'] for r in rs])),
                    columns=rs[0]['columns'],condition_max=max(r['condition'] for r in rs),condition_median=float(np.median([r['condition'] for r in rs])),
                    smin_min=min(r['smin'] for r in rs),smin_median=float(np.median([r['smin'] for r in rs])),
                    pipeline_seconds=sum(sum(r.get(v,0) for v in ['screening_seconds','feature_seconds','regression_seconds','common_preprocess_seconds']) for r in rs))
                if rec['status']=='OK':
                    scores={}
                    for _,decoded in values:
                        for edge,value in decoded[3].items():
                            edge=tuple(sorted(edge));scores[edge]=max(scores.get(edge,0.),value)
                    rec.update(sparse_metrics(set(map(tuple,data['net']['triple_edges'])),scores,math.comb(n,3)))
                else:rec.update({m:None for m in ['F1','AP','MCC','Jaccard','FP','FN','TP','TN']})
                networks.append(rec)
        np.savez_compressed(folder/'arrays.npz',**arrays)
        save(folder/'selection_locks.json',model_locks);save(folder/'target_rows.json',rows);save(folder/'network_rows.json',networks)
        files=[p for p in folder.rglob('*') if p.is_file() and p.name!='completion.json']
        save(folder/'completion.json',dict(status='COMPLETE',seconds=time.perf_counter()-started,
            target_fits=len(rows),statuses=dict(Counter(r['status'] for r in rows)),
            hashes={str(p.relative_to(folder)):digest(p) for p in files}))
        return dict(name=job['name'],resumed=False,seconds=round(time.perf_counter()-started,2),statuses=dict(Counter(r['status'] for r in rows)))


def dispatch(jobs,out,cfg,workers):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(execute,(j,str(out),cfg)) for j in jobs]
        for future in as_completed(futures):print(json.dumps(future.result()),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['pilot','formal','b1','b2','b3','extension','matrix','report','verify'])
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT);parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args();state=protocol(args.out);cfg=state['config'];jobs=[]
    if args.stage=='pilot':jobs=[make_job('A',n,s) for n in [15,30] for s in [0,1]]
    elif args.stage=='formal':
        gate=[]
        for n in [15,30]:
            records=[]
            for s in [0,1]:
                folder=args.out/'jobs'/make_job('A',n,s)['name']
                if not verified_job(folder):raise ValueError('Finish both pilot seeds first')
                records.extend(json.loads((folder/'network_rows.json').read_text()))
            for noise in [0.,.05]:
                for k in [16,32]:
                    rs=[r for r in records if r['noise']==noise and r['K']==k]
                    rate=sum(r['BIC_OK'] for r in rs)/sum(r['targets'] for r in rs)
                    gate.append(dict(N=n,noise=noise,K=k,BIC_success_rate=rate))
        save(args.out/'formal_gate.json',dict(cells=gate,passed=all(r['BIC_success_rate']>=.95 for r in gate)))
        if not all(r['BIC_success_rate']>=.95 for r in gate):raise ValueError('Pilot numerical gate failed; do not expand or relax BIC')
        jobs=[make_job('A',n,s) for n in [15,30] for s in range(30)]
    elif args.stage in ('b1','b2','b3','extension'):
        shared=dict(methods=['pearson','scc_ols','fwl'],noises=[.05],budgets=[16])
        if args.stage=='b1':jobs=[make_job('B1',30,s,'trajectory',trajectories=g,**shared) for s in [0,1] for g in [4,8,16,32,64]]
        elif args.stage=='b2':jobs=[make_job('B2',30,s,'trajectory',duration=t,**shared) for s in [0,1] for t in [1.,2.,5.]]
        elif args.stage=='b3':jobs=[make_job('B3',30,s,delta=d,**shared) for s in [0,1] for d in [.2,.4,1.,1.5]]
        else:jobs=[make_job('B1_extension',30,s,'trajectory',trajectories=g,**shared) for s in [0,1] for g in [128,256,512]]
    elif args.stage=='matrix':
        from .matrix import run
        run(args.out,cfg)
    elif args.stage=='report':
        from .report import report
        report(args.out)
    elif args.stage=='verify':
        count_jobs=sum(verified_job(folder) for folder in (args.out/'jobs').iterdir())
        assert sources()==state['science_sources']
        base=ROOT/'results/large_scale_screening_v1/p0_p1_20261103'
        prior=json.loads((base/'delivery_artifact_sha256.json').read_text())
        assert all(digest(base/name)==sha for name,sha in prior.items())
        save(args.out/'verification.json',dict(complete_jobs=count_jobs,science_sources_unchanged=True,
            previous_delivery_artifacts_unchanged=len(prior),backend_config_unchanged=cfg['backend']==load_config()['backend']))
        print('VERIFIED',count_jobs,'jobs and',len(prior),'historical artifacts',flush=True)
    if jobs:dispatch(jobs,args.out,cfg,args.workers)


if __name__=='__main__':main()
