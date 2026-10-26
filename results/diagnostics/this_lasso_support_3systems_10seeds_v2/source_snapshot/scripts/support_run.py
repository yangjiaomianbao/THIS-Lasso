"""Three-system support/false-positive diagnostics, fixed and multi-network stages."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import csv
import importlib.util
import itertools
import json
import shutil
import subprocess
import sys
import time
import tomllib
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits
from src.bridge.numeric_io import file_hash, write_toml, stamp
from src.experiments.fair_v2 import network
from src.experiments.support_diagnostics import METHODS, SYSTEMS, validation_path, bic_path, align_bic, kkt
from src.solvers.large_library import initial_ridge, stlsq_dual
from src.solvers.adaptive_lasso import BICNotApplicable
from src.preprocessing.taylor import restore_coefficients

spec = importlib.util.spec_from_file_location('support_d30', ROOT/'scripts/dynamics30.py')
d30 = importlib.util.module_from_spec(spec); spec.loader.exec_module(d30)
pilot = d30.pilot
LEGACY = ROOT/'results/dynamics15/five_systems_certified_20261022'
DEFAULT = ROOT/'results/diagnostics/this_lasso_support_3systems_10seeds_v2'


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    pilot.save_json(temporary, value); temporary.replace(path)


def table(path, rows):
    rows = list(rows)
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ['status']
    with Path(path).open('w', newline='') as f:
        writer=csv.DictWriter(f, fields);writer.writeheader();writer.writerows(rows)


def init_plan(out, workers):
    out.mkdir(parents=True, exist_ok=True)
    if (out/'plan.json').exists():
        return json.loads((out/'plan.json').read_text())
    c = tomllib.loads((LEGACY/'config.toml').read_text())
    c.update(generation_systems=SYSTEMS, noise_ratios=[0., .05, .1], workers=workers,
             support_relative_tolerance=1e-6, holdout_1se_rule='independent validation-loss SE; no CV',
             bootstrap_seed=2026102604, bootstrap_resamples=5000)
    instances=[]; rng_ids=set()
    for network_id in range(6):
        if network_id == 0:
            topology_seed=weight_seed=2026102200; pairs=triples=None
        else:
            topology_seed=int(np.random.SeedSequence([2026102601,1,network_id]).generate_state(1)[0])
            weight_seed=int(np.random.SeedSequence([2026102601,2,network_id]).generate_state(1)[0])
            pairs,triples=network(topology_seed,weight_seed)
        for sample_id in range(1,11):
            seed = 2026102200+sample_id if network_id==0 else int(np.random.SeedSequence([2026102601,3,network_id,sample_id]).generate_state(1)[0])
            assert seed not in rng_ids; rng_ids.add(seed)
            folder=f'A_fixed/d{sample_id:02d}' if network_id==0 else f'B_networks/n{network_id:02d}_d{sample_id:02d}'
            nc=deepcopy(c);nc['seeds']=[seed]
            if network_id:
                nc['noise_master_seed']=int(np.random.SeedSequence([2026102601,4,network_id]).generate_state(1)[0])
                for system in SYSTEMS:
                    sign=-1 if system=='lotka_volterra' else 1
                    nc[system]['pair_edges']=[[*r[:-1],sign*r[-1]] for r in pairs]
                    nc[system]['triple_edges']=[[*r[:-1],sign*r[-1]] for r in triples]
            (out/folder).mkdir(parents=True,exist_ok=True)
            write_toml(out/folder/'config.toml',nc)
            instances.append(dict(stage='A' if network_id==0 else 'B', network_id=network_id,
                                  sample_id=sample_id,data_seed=seed,topology_seed=topology_seed,weight_seed=weight_seed,
                                  folder=folder,config=folder+'/config.toml',noise_master_seed=nc['noise_master_seed']))
    write_toml(out/'prepare_request.toml',dict(instances=instances))
    source_names=['scripts/support_run.py','scripts/support_prepare.jl','scripts/support_analyse.py',
                  'src/experiments/support_diagnostics.py','src/data/multisystem_dynamics.jl',
                  'src/data/paper_dynamics.jl','src/solvers/lars_validation.py','src/solvers/large_library.py',
                  'src/solvers/plain_lasso.py','src/solvers/adaptive_lasso.py','scripts/dynamics30.py',
                  'scripts/dynamics_pilot.py']
    source_hashes={n:file_hash(ROOT/n) for n in source_names}
    for n in source_names:
        dest=out/'source_snapshot'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dest)
    legacy_files={}
    for system in SYSTEMS:
        for seed in [2026102201,2026102202]:
            sid=seed-2026102200;target=out/f'A_fixed/d{sid:02d}'/system/f'seed_{seed}'/'data'
            shutil.copytree(LEGACY/system/f'seed_{seed}'/'data',target,dirs_exist_ok=True)
            for f in (LEGACY/system/f'seed_{seed}').rglob('*'):
                if f.is_file():legacy_files[str(f.relative_to(LEGACY))]=file_hash(f)
    plan=dict(created=stamp(), systems=SYSTEMS, noise_ratios=c['noise_ratios'], methods=METHODS,
              instances=instances,workers=workers, blas_threads=1, models_A=540,models_B=2700,
              stage_order=['A','A_analysis','conditional_A_Q4','B','B_analysis'],
              legacy_model_records=72,legacy_hashes=legacy_files,source_sha256=source_hashes,
              q4=dict(trigger='selected validation models have triple FP>=15 and aligned-path fixed-threshold F1 never reaches .9',
                      sample_id=1,max_groups_per_model=10,max_competitor_columns=128,
                      selection='top predicted scores; truth used only after group selection',
                      refit='fixed original alpha/weights on local competing columns; not full-library re-selection'),
              exclusions=['lorenz','stuart_landau'],
              canonical_system_code_order=c['systems'])
    save(out/'plan.json',plan);table(out/'seed_manifest.csv',instances)
    save(out/'progress.json',dict(status='PREPARING',created=stamp(),planned_models=3240))
    return plan


def old_trial(instance, system, ratio, method):
    if instance['stage']=='A' and instance['sample_id']<=2 and ratio in (0.,.05):
        return LEGACY/system/f'seed_{instance["data_seed"]}'/pilot.slot(ratio)/method
    return None


def verify_old(path):
    status=json.loads((path/'status.json').read_text())
    return status


def dataset(job):
    out,instance,system=job;out=Path(out);threadpool_limits(limits=1)
    cfg=tomllib.loads((out/instance['config']).read_text());base=out/instance['folder'];seed=instance['data_seed']
    data=d30.load_data(base,system,seed);folder=data['folder']
    if not (folder/'feature_audit.json').exists():d30.features(data)
    if not (folder.parent/'unit_response_noise.csv').exists():d30.prepare_noise(data,cfg,base,system,seed)
    Z=np.load(folder/'Z_train.npy');Zv=np.load(folder/'Z_validation.npy')
    train=data['splits']=='train';val=data['splits']=='validation'
    means=np.load(folder/'feature_means.npy');scales=np.load(folder/'feature_scales.npy')
    fractions=cfg['lasso']['alpha_fractions'];completed=[]
    for ratio in cfg['noise_ratios']:
        cell=folder.parent/pilot.slot(ratio);Y=np.loadtxt(cell/'Y_noisy.csv',delimiter=',',ndmin=2)
        init=initial_ridge(Z[:,1:],Y[:,train].T,cfg['adaptive']['ridge_rho'])
        np.save(cell/'Ridge_initial_standardized.npy',init)
        for method in METHODS:
            trial=cell/method;trial.mkdir(exist_ok=True)
            if (trial/'status.json').exists():
                completed.append(json.loads((trial/'status.json').read_text())['status']);continue
            started=time.perf_counter();previous=old_trial(instance,system,ratio,method)
            old=verify_old(previous) if previous else None
            status=dict(stage=instance['stage'],network_id=instance['network_id'],sample_id=instance['sample_id'],
                        system=system,seed=seed,noise_ratio=ratio,method=method,
                        legacy_source=str(previous) if previous else '',legacy_final_reused=False)
            if old and old['status']=='NA':
                status.update(status='NA',reason='Reused original BIC NA',legacy_final_reused=True)
            elif method in ('M0D','M0sD'):
                if old and old['status']=='OK':
                    C=sparse.load_npz(previous/'coefficients.npz').toarray();status['legacy_final_reused']=True
                else:
                    design=np.asfortranarray(np.load(folder/'theta.npy',mmap_mode='r')[train]) if method=='M0D' else Z
                    fitted,diag=stlsq_dual(design,Y[:,train].T,**dict(threshold=.1,rho=1.,niter=10))
                    C=fitted if method=='M0D' else restore_coefficients(fitted,means,scales)
                sparse.save_npz(trial/'coefficients.npz',sparse.csr_matrix(C));status['status']='OK'
            else:
                Cs=np.zeros((15,816));grid=np.zeros((len(fractions),15,816));one_se=np.zeros_like(Cs)
                node_status=[]
                for node in range(15):
                    diagnostic_path=trial/f'node_{node+1:03d}.json'
                    solution_path=trial/f'node_{node+1:03d}_path.npz'
                    if diagnostic_path.exists():
                        diag=json.loads(diagnostic_path.read_text())
                        if diag['status']=='OK':
                            cached=np.load(solution_path);Cs[node]=cached['selected'];grid[:,node]=cached['aligned']
                            one_se[node]=cached['one_se']
                        node_status.append(diag['status']);continue
                    node_started=time.perf_counter()
                    try:
                        if method in ('M1vC','M3C'):
                            a=np.ones(815) if method=='M1vC' else (np.abs(init[node])+cfg['adaptive']['eps'])**cfg['adaptive']['gamma']
                            coef,diag,candidates=validation_path(Z,Y[node,train],Zv,Y[node,val],a,cfg['lasso'])
                            aligned=candidates;alphas=np.array([r['alpha'] for r in diag['validation_path']])
                            alphas_aligned=alphas;one=candidates[diag['holdout_1se_index']]
                        else:
                            coef,diag,candidates=bic_path(method,Z,Y[node,train],cfg)
                            alphas=np.asarray(diag['alpha_path']);aligned,reference=align_bic(candidates,alphas,fractions)
                            alphas_aligned=reference*np.asarray(fractions);one=coef.copy()
                            a=np.ones(815) if method=='M1B' else np.asarray(diag['adaptive_scales'])[1:]
                        # Save original native BIC knots or all certified validation grid solutions.
                        residual=Z@candidates.T-Y[node,train,None]
                        validation_residual=Zv@candidates.T-Y[node,val,None]
                        grad=Z[:,1:].T@residual/len(Z);penalty=alphas[None,:]/a[:,None]
                        slopes=candidates[:,1:].T;active=slopes!=0
                        violation=np.maximum(np.abs(grad)-penalty,0.)
                        violation[active]=np.abs((grad+penalty*np.sign(slopes))[active])
                        denominator=max(1.,float(np.max(np.abs(Z[:,1:].T@(Y[node,train]-Y[node,train].mean())/len(Z)))))
                        cert=violation.max(axis=0)/denominator
                        np.savez_compressed(solution_path, selected=coef, aligned=aligned, one_se=one,
                                            candidates=candidates,alphas=alphas,aligned_alphas=alphas_aligned,
                                            train_MSE=(residual**2).mean(axis=0),validation_MSE=(validation_residual**2).mean(axis=0),
                                            validation_sample_losses=validation_residual**2,full_library_KKT=cert)
                        Cs[node]=coef;grid[:,node]=aligned;one_se[node]=one
                        diag.update(status='OK',node=node+1,wall_seconds=time.perf_counter()-node_started)
                    except BICNotApplicable as error:
                        diag=dict(status='NA',node=node+1,reason=str(error),wall_seconds=time.perf_counter()-node_started)
                    except Exception as error:
                        diag=dict(status='FAILED',node=node+1,reason=str(error),traceback=traceback.format_exc(),
                                  wall_seconds=time.perf_counter()-node_started)
                    save(diagnostic_path,diag);node_status.append(diag['status'])
                state='FAILED' if 'FAILED' in node_status else 'NA' if 'NA' in node_status else 'OK'
                status.update(status=state,node_counts={s:node_status.count(s) for s in ['OK','NA','FAILED']})
                if state=='OK':
                    C=restore_coefficients(Cs,means,scales)
                    if old and old['status']=='OK':
                        original=sparse.load_npz(previous/'coefficients.npz').toarray()
                        delta=float(np.max(np.abs(C-original)))
                        status['legacy_replay_max_abs_difference']=delta
                        if not np.allclose(C,original,atol=1e-10,rtol=1e-8):
                            raise RuntimeError(f'Legacy solver replay changed coefficients: {trial}, {delta}')
                        C=original;status['legacy_final_reused']=True
                    sparse.save_npz(trial/'coefficients.npz',sparse.csr_matrix(C))
                    np.save(trial/'aligned_path_standardized.npy',grid)
                    if method in ('M1vC','M3C'):
                        sparse.save_npz(trial/'holdout_1se_coefficients.npz',sparse.csr_matrix(restore_coefficients(one_se,means,scales)))
            status.update(fit_seconds=time.perf_counter()-started,finished=stamp())
            save(trial/'status.json',status);completed.append(status['status'])
            print('MODEL',instance['folder'],system,ratio,method,status['status'],flush=True)
    return dict(instance=instance['folder'],system=system,counts={s:completed.count(s) for s in set(completed)})


def lock_stage(out, plan, stage):
    locks={}
    for inst in plan['instances']:
        if inst['stage']!=stage:continue
        for system,ratio,method in itertools.product(SYSTEMS,plan['noise_ratios'],METHODS):
            trial=out/inst['folder']/system/f'seed_{inst["data_seed"]}'/pilot.slot(ratio)/method
            status=json.loads((trial/'status.json').read_text())
            locks[str(trial.relative_to(out))]=dict(status=status['status'],status_sha256=file_hash(trial/'status.json'),
                coefficients_sha256=file_hash(trial/'coefficients.npz') if status['status']=='OK' else '')
    save(out/f'selection_locked_{stage}.json',dict(stage=stage,models=locks,locked=stamp(),test_evaluated_before_lock=False))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=DEFAULT)
    parser.add_argument('--workers',type=int,default=8);parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();out=args.out.resolve();plan=init_plan(out,args.workers)
    for name,digest in plan['source_sha256'].items():assert file_hash(ROOT/name)==digest,name
    env=dict(os.environ,JULIA_NUM_THREADS='1',JULIA_DEPOT_PATH=str(out/'julia_depot')+':'+str(ROOT/'.julia-depot-r0'))
    (out/'logs').mkdir(exist_ok=True);(out/'julia_depot').mkdir(exist_ok=True)
    runtime=['/home/yg/.local/opt/julia-1.9.4/bin/julia','--startup-file=no','--project='+str(ROOT/'environment/r0/julia')]
    if not (out/'data_prepared.json').exists():
        with (out/'logs/prepare.log').open('w') as log:
            subprocess.run(runtime+[str(ROOT/'scripts/support_prepare.jl'),str(out)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        save(out/'data_prepared.json',dict(status='PASS',datasets=180,created=stamp()))
    if args.prepare_only:return
    # Cheap Q1/Q2 delivery from old final coefficients precedes new fitting.
    from support_analyse import legacy_support, analyse_stage, finish
    legacy_support(out,plan)
    started=time.perf_counter()
    for stage in ['A','B']:
        if (out/f'analysed_{stage}.json').exists():continue
        if (out/f'analysis_tables_{stage}.json').exists():
            save(out/'progress.json',dict(status='ANALYSING',stage=stage,updated=stamp(),
                                          resumed_from='committed analysis tables'))
            analyse_stage(out,plan,stage)
            continue
        jobs=[(str(out),i,s) for i in plan['instances'] if i['stage']==stage for s in SYSTEMS]
        save(out/'progress.json',dict(status='RUNNING',stage=stage,planned_datasets=len(jobs),updated=stamp()))
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(dataset,j) for j in jobs]
            for n,future in enumerate(as_completed(futures),1):
                result=future.result();print('DATASET_DONE',n,len(jobs),result,flush=True)
                save(out/'progress.json',dict(status='RUNNING',stage=stage,completed_datasets=n,planned_datasets=len(jobs),updated=stamp()))
        lock_stage(out,plan,stage)
        save(out/'progress.json',dict(status='ANALYSING',stage=stage,completed_datasets=len(jobs),
                                      planned_datasets=len(jobs),updated=stamp()))
        analyse_stage(out,plan,stage)
    for name,digest in plan['legacy_hashes'].items():assert file_hash(LEGACY/name)==digest,name
    finish(out,plan,time.perf_counter()-started)


if __name__=='__main__':
    try:main()
    except Exception:
        error=traceback.format_exc();print(error,flush=True)
        target=Path(sys.argv[sys.argv.index('--out')+1]) if '--out' in sys.argv else DEFAULT
        if (target/'progress.json').exists():
            state=json.loads((target/'progress.json').read_text())
            save(target/'progress.json',dict(state,status='ERROR',error=error,updated=stamp()))
        raise
