"""Versioned scheduler. Default execution is P1 only; larger stages are explicit."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import resource
import shutil
import sys
import time
import traceback

import numpy as np
from threadpoolctl import threadpool_limits
from .common import ROOT,HERE,save,table,digest,load_config,source_hashes,runtime_info,Monitor,timeout
from .data import generate,save_data,load_data,samples,noisy_derivative,coordinates
from .candidates import blocks,count,shared_pool
from .screeners import Context,Screener,STAT_METHODS
from .library import terms_and_groups,native_indices,decode,evaluate
from .backend import fit
from .metrics import screening_metrics,sparse_metrics


def dimensions(args,cfg):
    sizes=args.sizes or ([30] if args.stage=='p1' else [30,50,100,200] if args.stage=='p2' else [500,1000])
    if args.stage=='p1' and sizes!=[30]:raise ValueError('P1 only supports N=30')
    budgets=args.budgets or ([16] if args.stage=='p1' else cfg['budgets'])
    noises=args.noises if args.noises is not None else ([.05] if args.stage=='p1' else cfg['noise_ratios'])
    if any(n<3 for n in sizes) or any(k<1 for k in budgets):raise ValueError('Invalid N,K')
    methods=list(STAT_METHODS)+(['triadic_ndformer'] if args.checkpoint or args.stage!='p1' else [])
    if args.stage=='p1' and (args.repeats!=1 or args.trajectories_per_topology!=1):raise ValueError('First P1 is one topology/data seed')
    return sizes,budgets,noises,methods


def dry_run(args,cfg):
    sizes,budgets,noises,methods=dimensions(args,cfg);rows=[]
    for n in sizes:
        mode=args.mode if args.mode!='auto' else ('full' if n<=200 else 'pool')
        counts=samples(n,args.fixed_samples);pool=count(n) if mode=='full' else min(count(n),cfg['screening']['pool_size'])
        for k in budgets:
            actual=min(k,pool);p=6*n-2+4*actual
            rows.append(dict(N=n,mode=mode,samples=counts,candidates_per_target=count(n),candidates_network=n*count(n),
                scored_per_target=pool,K=actual,design_columns_per_target=p,
                raw_train_design_MiB=counts[0]*p*8/2**20,full_cubic_columns_forbidden=math.comb(n+3,3),
                expected_BIC_guard='NA: K_samples<=P_features' if counts[0]<=p else 'requires rank/condition/variance checks',
                fits=n*len(methods)*len(noises)*args.repeats*args.trajectories_per_topology))
    return dict(stage=args.stage,methods=methods,rows=rows,total_fits=sum(r['fits'] for r in rows),limits=cfg['limits'],
        output=str(args.out or ROOT/'results/large_scale_screening_v1/<unique date label>'),
        run_policy='explicit stage only; no automatic P2/P3 launch',noise_ratios=noises,
        time_estimate='Use P1 measured times in REPORT.md; exact SVD/BIC cost depends on rank and cannot be inferred from candidate counts alone')


def pilot(args,cfg):
    args.out.mkdir(parents=True,exist_ok=False);rows=[]
    for k2,k3 in [(0.3,0.6),(0.1,0.3),(0.6,0.6)]:
        c=dict(cfg['data'],K2=k2,K3=k3)
        d=generate(8,cfg['seeds']['topology']+900000,cfg['seeds']['trajectory']+900000,c,[256,256,256])
        diag=d['diagnostics'];acceptable=diag['phase_order_max']<.85 and diag['phase_excursion_max']<.8
        rows.append(dict(K2=k2,K3=k3,acceptable=acceptable,**diag))
    allowed=[r for r in rows if r['acceptable']]
    if not allowed:raise RuntimeError('Coupling pilot failed synchronization/excursion rule; no benchmark launched')
    chosen=allowed[0];save(args.out/'coupling_pilot.json',dict(rule='first registered combination with order max<0.85 and phase excursion<0.8; no structural F1 used',
        selected=dict(K2=chosen['K2'],K3=chosen['K3']),trials=rows))
    print(json.dumps(dict(output=str(args.out),selected=[chosen['K2'],chosen['K3']]),indent=2))


def run(args,cfg):
    if args.calibration:
        frozen=json.loads(args.calibration.read_text());cfg['data'].update(frozen['selected'])
    elif args.stage!='p1' and not args.dry_run:raise ValueError('P2/P3 require --calibration from the completed pilot; do not retune K by N')
    plan=dry_run(args,cfg)
    if args.dry_run:print(json.dumps(plan,indent=2));return
    if args.stage!='p1' and not args.checkpoint:raise ValueError('P2/P3 require an explicitly trained neural --checkpoint')
    if args.out is None:
        args.out=ROOT/'results/large_scale_screening_v1'/time.strftime('%Y%m%d_%H%M%S')
        plan=dry_run(args,cfg)
    out=args.out
    if out is None:
        out=ROOT/'results/large_scale_screening_v1'/time.strftime('%Y%m%d_%H%M%S')
    sources=source_hashes();state=dict(version=cfg['version'],config=cfg,plan=plan,sources=sources,runtime=runtime_info(),
        checkpoint_sha256=digest(args.checkpoint) if args.checkpoint else None,
        calibration_sha256=digest(args.calibration) if args.calibration else None,
        paired_seed_protocol='topology_base+rep; trajectory_base+100*rep+trajectory_repeat; separate noise/pool RNG namespaces',
        repeats=args.repeats,trajectories_per_topology=args.trajectories_per_topology,fixed_samples=args.fixed_samples)
    if out.exists():
        if not args.resume:raise FileExistsError('Output exists; choose unique --out or explicit --resume')
        prior=json.loads((out/'protocol.json').read_text())
        if prior!=state:raise ValueError('Frozen config/source/checkpoint differs; use a new output directory')
    else:
        out.mkdir(parents=True,exist_ok=False);save(out/'protocol.json',state)
        for path in sources:
            destination=out/'source_snapshot'/path;destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/path,destination)
    lock=(out/'.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # Soft and hard address-space bound; only this new experiment process.
    bound=int(cfg['limits']['memory_gib']*2**30)
    resource.setrlimit(resource.RLIMIT_AS,(bound,bound))
    model=None
    if args.checkpoint:
        from .neural import load
        import torch
        torch.set_num_threads(1);model=load(args.checkpoint)
        if args.stage!='p1' and model.smoke_only:raise ValueError('Smoke-only checkpoint cannot be used as formal P2/P3 pretraining')
    started=time.perf_counter();sizes,budgets,noises,methods=dimensions(args,cfg)
    attempts=0
    try:
        with threadpool_limits(limits=cfg['limits']['blas_threads']):
            for n in sizes:
                mode=args.mode if args.mode!='auto' else ('full' if n<=200 else 'pool')
                for rep in range(args.repeats):
                    for trajectory_rep in range(args.trajectories_per_topology):
                        topo=cfg['seeds']['topology']+rep;traj=cfg['seeds']['trajectory']+100*rep+trajectory_rep
                        folder=out/f'n{n}'/f'topology_{topo}_trajectory_{traj}';datafolder=folder/'data'
                        if datafolder.exists():
                            data=load_data(datafolder)
                            cached=json.loads((datafolder/'hashes.json').read_text())
                            assert all(digest(datafolder/k)==v for k,v in cached.items()),'Cached data changed'
                        else:
                            with Monitor() as datam:
                                data=generate(n,topo,traj,cfg['data'],samples(n,args.fixed_samples))
                                save_data(datafolder,data)
                            save(datafolder/'resources.json',datam.fields())
                            save(datafolder/'hashes.json',{p.name:digest(p) for p in datafolder.iterdir() if p.is_file()})
                        if model is not None:
                            training_graphs={r['topology_hash'] for records in model.topology_provenance.values() for r in records}
                            if data['diagnostics']['topology_sha256'] in training_graphs:
                                raise ValueError('Benchmark topology overlaps neural train/validation/test topology')
                        train=data['splits']=='train';test=data['splits']=='test'
                        x,center,cd=coordinates(data['phase'],train);save(datafolder/'coordinates.json',dict(center=center,**cd))
                        for ratio in noises:
                            y=noisy_derivative(data,ratio,cfg['seeds']['noise'],traj)
                            context=Context(x[train],y[train]);noiseout=folder/f'noise_{ratio:g}';noiseout.mkdir(exist_ok=True)
                            save(noiseout/'common_preprocessing.json',dict(seconds=context.seconds,rank=context.rank,
                                samples=int(train.sum()),response_orthogonality=float(np.max(abs(context.q.T@context.residual)))))
                            for target in range(n):
                                poolstart=time.perf_counter()
                                pairs=np.concatenate(list(blocks(n,target,cfg['screening']['block_size']))) if mode=='full' else shared_pool(n,target,cfg['screening']['pool_size'],cfg['seeds']['pool']+topo)
                                poolseconds=time.perf_counter()-poolstart
                                truth={tuple(t[1:]) for t in data['net']['triples'] if t[0]==target}
                                for method in methods:
                                    methoddir=noiseout/method;methoddir.mkdir(exist_ok=True)
                                    scoredir=methoddir/f'target_{target:04d}_screen'
                                    if scoredir.with_suffix('.json').exists():
                                        info=json.loads(scoredir.with_suffix('.json').read_text())
                                        assert info['array_sha256']==digest(scoredir.with_suffix('.npz'))
                                        with np.load(scoredir.with_suffix('.npz')) as f: scored={k:f[k] for k in f.files}
                                        scored.update(info)
                                    else:
                                        screen=Screener(method,cfg['screening'],context,model,data['phase'][train],data['groups'][train])
                                        try:
                                            with timeout(cfg['limits']['task_seconds']):scored=screen.score(x[train],y[train],target,pairs)
                                            np.savez_compressed(scoredir.with_suffix('.npz'),**{k:scored[k] for k in ['pairs','scores','ranking']})
                                            save(scoredir.with_suffix('.json'),dict(status='OK',**{k:v for k,v in scored.items() if k not in ['pairs','scores','ranking']},array_sha256=digest(scoredir.with_suffix('.npz'))))
                                        except (TimeoutError,MemoryError,np.linalg.LinAlgError) as e:
                                            status='TIMEOUT' if isinstance(e,TimeoutError) else 'OOM' if isinstance(e,MemoryError) else 'NUMERICAL_FAILURE'
                                            for k in budgets:
                                                path=methoddir/f'k{k}'/f'target_{target:04d}.json'
                                                save(path,dict(N=n,topology_seed=topo,trajectory_seed=traj,target=target,method=method,K=k,mode=mode,
                                                    noise=ratio,status=status,reason=str(e),stage='screening'))
                                            continue
                                    for k in budgets:
                                        cell=methoddir/f'k{k}';cell.mkdir(exist_ok=True);dest=cell/f'target_{target:04d}.json'
                                        if dest.exists():continue
                                        selected=scored['pairs'][scored['ranking'][:k]];terms,groups=terms_and_groups(n,target,selected)
                                        row=dict(N=n,topology_seed=topo,trajectory_seed=traj,target=target,method=method,K=k,mode=mode,
                                            noise=ratio,sample_budget='fixed' if args.fixed_samples else 'scaled',
                                            **screening_metrics(truth,scored['pairs'],scored['scores'],scored['ranking'],k,count(n)),
                                            screening_seconds=scored['seconds'],cpu_peak_rss_bytes=scored['cpu_peak_rss_bytes'],
                                            gpu_peak_bytes=None,gpu_reason='CPU run; CUDA not used',
                                            common_preprocess_seconds_per_target=context.seconds/n,
                                            data_generation_seconds_per_target=data['diagnostics']['generation_seconds']/n,
                                            pool_generation_seconds=poolseconds,features=len(terms),experimental=method=='gc_cmi')
                                        row['candidate_group_columns']={f'{j},{k}':ids for (j,k),ids in groups.items()}
                                        np.savez_compressed(dest.with_name(dest.stem+'_library.npz'),d=native_indices(terms),selected_pairs=selected)
                                        try:
                                            with Monitor() as fitm:
                                                with timeout(cfg['limits']['task_seconds']):result=fit(x[train],y[train,target],terms,cfg['backend'])
                                            row.update({key:value for key,value in result.items() if key not in ['coefficient','diagnostics','means','scales']})
                                            row['cpu_peak_rss_bytes']=max(row['cpu_peak_rss_bytes'],fitm.peak)
                                            if result['status']=='OK':
                                                coef=result['coefficient'];decoded=decode(coef,terms,target)
                                                # Lock fit + configuration before evaluating held-out data or truth.
                                                np.savez_compressed(dest.with_suffix('.npz'),coefficient=coef,d=native_indices(terms),
                                                    means=result['means'],scales=result['scales'],selected_pairs=selected)
                                                save(dest.with_name(dest.stem+'_selection_locked.json'),dict(diagnostics=result['diagnostics'],
                                                    coefficient_sha256=digest(dest.with_suffix('.npz')),test_evaluated=False))
                                                row['directed_scores']={str(order):[dict(key=list(key),score=v) for key,v in values.items() if v>0] for order,values in decoded.items()}
                                                row['nonzero_nonbias']=int(np.count_nonzero(abs(coef[1:])>1e-8))
                                                truthkeys={(target,*p) for p in truth}
                                                row['target_structure']=sparse_metrics(truthkeys,decoded[3],count(n))
                                                row['test_MSE_clean']=float(np.mean((evaluate(x[test],terms)@coef-data['derivative'][test,target])**2))
                                                row['test_MSE_observed']=float(np.mean((evaluate(x[test],terms)@coef-y[test,target])**2))
                                                row['alpha']=result['diagnostics']['alpha']
                                        except (TimeoutError,MemoryError,np.linalg.LinAlgError) as e:
                                            row.update(status='TIMEOUT' if isinstance(e,TimeoutError) else 'OOM' if isinstance(e,MemoryError) else 'NUMERICAL_FAILURE',reason=str(e))
                                        row['pipeline_seconds']=sum(row.get(key,0) for key in ['screening_seconds','pool_generation_seconds','feature_seconds','regression_seconds','common_preprocess_seconds_per_target','data_generation_seconds_per_target'])
                                        save(dest,row);attempts+=1
                                if (target+1)%5==0:print(f'N={n} seed={topo}/{traj} noise={ratio} target={target+1}/{n} completed',flush=True)
        assert all(digest(ROOT/p)==h for p,h in sources.items()),'Protected source/lock changed'
        from .report import report
        report([out],out)
        save(out/'completion.json',dict(status='COMPLETE_P1' if args.stage=='p1' else 'COMPLETE_REQUESTED_STAGE',
            elapsed_seconds=time.perf_counter()-started,newly_computed_trials=attempts,
            sources_unchanged=True,larger_stages_launched=False,backend='existing M2 guarded Adaptive LASSO BIC'))
        print('REPORT',out/'REPORT.md',flush=True)
    except BaseException as e:
        save(out/'interrupted.json',dict(reason=str(e),elapsed_seconds=time.perf_counter()-started,traceback=traceback.format_exc(),resume='rerun identical command with --resume'))
        raise
    finally:lock.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    def common(p):p.add_argument('--config',type=Path);p.add_argument('--out',type=Path,required=True)
    p=sub.add_parser('pilot');common(p)
    p=sub.add_parser('train-neural');common(p);p.add_argument('--steps',type=int);p.add_argument('--sizes',type=int,nargs='+');p.add_argument('--topologies',type=int)
    p=sub.add_parser('neural-smoke');common(p);p.add_argument('--models',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p=sub.add_parser('run');p.add_argument('--config',type=Path);p.add_argument('--out',type=Path)
    p.add_argument('--stage',choices=['p1','p2','p3'],default='p1');p.add_argument('--dry-run',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--sizes',type=int,nargs='+');p.add_argument('--budgets',type=int,nargs='+');p.add_argument('--noises',type=float,nargs='+')
    p.add_argument('--repeats',type=int,default=1);p.add_argument('--trajectories-per-topology',type=int,default=1)
    p.add_argument('--mode',choices=['auto','full','pool'],default='auto');p.add_argument('--fixed-samples',action='store_true')
    p.add_argument('--checkpoint',type=Path);p.add_argument('--calibration',type=Path)
    p=sub.add_parser('report');p.add_argument('--inputs',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();cfg=load_config(getattr(args,'config',None))
    if args.command=='pilot':pilot(args,cfg)
    elif args.command=='run':run(args,cfg)
    elif args.command=='train-neural':
        from .neural import train
        with Monitor() as monitor:
            results=train(args.out,cfg,args.steps or cfg['neural']['steps'],tuple(args.sizes or cfg['neural']['train_sizes']),args.topologies or cfg['neural']['train_topologies'])
        save(args.out/'resources.json',monitor.fields());print(json.dumps(results,indent=2))
    elif args.command=='neural-smoke':
        from .neural import load
        import torch
        torch.set_num_threads(1);args.out.mkdir(parents=True,exist_ok=False)
        data=load_data(args.data);mask=data['splits']=='train';x,_,_=coordinates(data['phase'],mask)
        y=noisy_derivative(data,.05,cfg['seeds']['noise'],data['diagnostics']['trajectory_seed'])
        pairs=np.concatenate(list(blocks(x.shape[1],0,128)));context=Context(x[mask],y[mask]);records=[]
        for architecture in ['light','transformer']:
            model=load(args.models/(architecture+'.pt'))
            hashes={r['topology_hash'] for values in model.topology_provenance.values() for r in values}
            assert data['diagnostics']['topology_sha256'] not in hashes
            screen=Screener('triadic_ndformer',cfg['screening'],context,model,data['phase'][mask],data['groups'][mask])
            with threadpool_limits(limits=1):
                scored=screen.score(x[mask],y[mask],0,pairs)
                swapped=screen.score(x[mask],y[mask],0,pairs[:,::-1])
            np.testing.assert_allclose(scored['scores'],swapped['scores'],atol=1e-7)
            np.savez_compressed(args.out/(architecture+'_scores.npz'),pairs=pairs,scores=scored['scores'],ranking=scored['ranking'])
            records.append(dict(architecture=architecture,N=x.shape[1],target_0=0,candidates=len(pairs),
                topology_disjoint=True,smoke_only=model.smoke_only,swap_max_abs=float(np.max(abs(scored['scores']-swapped['scores']))),
                parameters=sum(p.numel() for p in model.parameters()),**{k:scored[k] for k in ['seconds','cpu_peak_rss_bytes','gpu_peak_bytes','gpu_reason']}))
        save(args.out/'summary.json',records);print(json.dumps(records,indent=2))
    else:
        from .report import report
        report(args.inputs,args.out)


if __name__=='__main__':main()
