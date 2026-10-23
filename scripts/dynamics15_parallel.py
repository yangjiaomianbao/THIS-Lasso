"""Resume identical fits and schedule independent system/seed datasets concurrently."""
import argparse
import concurrent.futures
import importlib.util
import itertools
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import time
import tomllib
import traceback

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits
from src.bridge.numeric_io import file_hash
spec=importlib.util.spec_from_file_location('runner',ROOT/'scripts/dynamics15.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
pilot=r.pilot


def _node(node):return r.one(node)


def _trial(data,c,out,system,seed,ratio,method):
    folder=data['folder'];root=folder.parent/pilot.slot(ratio);trial=root/method
    if (trial/'status.json').exists():
        json.loads((trial/'status.json').read_text())
        print('REUSE_MODEL',system,seed,ratio,method,flush=True);return
    trial.mkdir(exist_ok=True);started=time.perf_counter()
    means=np.load(folder/'feature_means.npy');scales=np.load(folder/'feature_scales.npy')
    if method in ('M0D','M0sD'):
        if method=='M0D':design=np.asfortranarray(np.load(folder/'theta.npy',mmap_mode='r')[data['splits']=='train'])
        else:design=np.load(folder/'Z_train.npy',mmap_mode='r')
        Y=np.loadtxt(root/'Y_noisy.csv',delimiter=',',ndmin=2)
        C,diag=r.stlsq_dual(design,Y[:,data['splits']=='train'].T,c['stlsq']['lambda'],c['stlsq']['rho'],c['stlsq']['niter'])
        if method=='M0sD':C=r.restore_coefficients(C,means,scales)
        assert np.isfinite(C).all();sparse.save_npz(trial/'coefficients.npz',sparse.csr_matrix(C))
        pilot.save_json(trial/'status.json',dict(status='OK',method=method,fit_seconds=time.perf_counter()-started,
            numerical_variant='Original STLSQ algebra via primal/dual linear solves',**diag))
        print('FIT',system,seed,ratio,method,'OK',flush=True);return
    if method in ('M1B','M2B') and c['train_samples']<=data['meta']['P']:
        pilot.save_json(trial/'status.json',dict(status='NA',method=method,reason='Guarded BIC requires K>P',K=c['train_samples'],P=data['meta']['P']))
        print('FIT',system,seed,ratio,method,'NA',flush=True);return
    C=np.zeros((data['meta']['N'],data['meta']['P']));states=[];pending=[]
    for node in range(data['meta']['N']):
        diag_path=trial/f'node_{node+1:03d}.json';solution=trial/f'node_{node+1:03d}_solution.npz'
        if diag_path.exists():
            diag=json.loads(diag_path.read_text());states.append(diag['status'])
            if diag['status']=='OK':
                coef=np.load(solution)['coefficient_standardized'];assert coef.shape==(data['meta']['P'],) and np.isfinite(coef).all()
                C[node]=r.restore_coefficients(coef[None,:],means,scales)[0]
        else:
            if solution.exists():
                stash=out/'logs/interrupted_uncommitted'/trial.relative_to(out)
                stash.mkdir(parents=True,exist_ok=True);shutil.copy2(solution,stash/solution.name)
            pending.append(node)
    if pending:
        with concurrent.futures.ProcessPoolExecutor(max_workers=c['workers'],mp_context=multiprocessing.get_context('fork'),
                initializer=r.initialize,initargs=(data,c,root,trial,method)) as pool:
            jobs=[pool.submit(_node,node) for node in pending]
            for finished,future in enumerate(concurrent.futures.as_completed(jobs),1):
                node,coef,state=future.result();states.append(state)
                if coef is not None:C[node]=r.restore_coefficients(coef[None,:],means,scales)[0]
                if finished%15==0 or finished==len(pending):print('NODE',system,seed,ratio,method,f'{finished}/{len(pending)} pending',flush=True)
    state='FAILED' if 'FAILED' in states else 'NA' if 'NA' in states else 'OK'
    sparse.save_npz(trial/('coefficients.npz' if state=='OK' else 'available_nodes_only.npz'),sparse.csr_matrix(C))
    pilot.save_json(trial/'status.json',dict(status=state,method=method,fit_seconds=time.perf_counter()-started,
        node_counts={s:states.count(s) for s in ('OK','NA','FAILED')},full_library=True,process_isolated=True,
        resumed_nodes=data['meta']['N']-len(pending)))
    print('FIT',system,seed,ratio,method,state,flush=True)


def _dataset(c,out,system,seed):
    threadpool_limits(limits=1)
    data=r.d30.load_data(out,system,seed);folder=data['folder']
    if not (folder/'feature_audit.json').exists():r.d30.features(data)
    else:
        assert all((folder/name).exists() for name in ['theta.npy','feature_means.npy','feature_scales.npy','Z_train.npy','Z_validation.npy','Z_test.npy'])
    if not (folder.parent/'unit_response_noise.csv').exists():r.d30.prepare_noise(data,c,out,system,seed)
    for ratio in c['noise_ratios']:
        root=folder.parent/pilot.slot(ratio);init_path=root/'Ridge_initial_standardized.npy'
        if not init_path.exists():
            Z=np.load(folder/'Z_train.npy',mmap_mode='r');Y=np.loadtxt(root/'Y_noisy.csv',delimiter=',',ndmin=2)
            np.save(init_path,r.initial_ridge(Z[:,1:],Y[:,data['splits']=='train'].T,c['adaptive']['ridge_rho']))
        for method in c['method_order']:_trial(data,c,out,system,seed,ratio,method)
    return system,seed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label',default='five_systems_full_20261022')
    parser.add_argument('--datasets-at-once',type=int,default=3)
    args=parser.parse_args();assert Path(args.label).name==args.label and args.label not in ('.','..')
    out=ROOT/'results/dynamics15'/args.label;c=tomllib.loads((out/'config.toml').read_text())
    assert not (out/'selection_locked.json').exists() and not (out/'completion.json').exists()
    assert not (out/'execution_resume.json').exists()
    protocol=json.loads((out/'protocol.json').read_text())
    for name,expected in protocol['source_sha256'].items():assert file_hash(ROOT/name)==expected
    preserved={}
    for trial in out.glob('*/seed_*/noise_*/M*'):
        if not trial.is_dir():continue
        status=trial/'status.json'
        if status.exists():
            json.loads(status.read_text())
            for f in trial.glob('*'):
                if f.is_file():preserved[str(f.relative_to(out))]=file_hash(f)
        else:
            for f in trial.glob('node_*.json'):
                a=json.loads(f.read_text());preserved[str(f.relative_to(out))]=file_hash(f)
                sol=trial/f"node_{a['node_julia_index']:03d}_solution.npz"
                if a['status']=='OK':preserved[str(sol.relative_to(out))]=file_hash(sol)
    receipt=dict(reason='Schedule independent system/seed datasets concurrently; unchanged data, objective, alpha grid, tolerances, worker isolation and existing node results',
        datasets_at_once=args.datasets_at_once,workers_per_dataset=c['workers'],
        preserved_completed_files=preserved,source_sha256=file_hash(Path(__file__)),test_evaluated_before_resume=False)
    pilot.save_json(out/'execution_resume.json',receipt);started=time.perf_counter()
    try:
        with threadpool_limits(limits=1):
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.datasets_at_once,mp_context=multiprocessing.get_context('fork')) as pool:
                jobs=[pool.submit(_dataset,c,out,system,seed) for system,seed in itertools.product(c['systems'],c['seeds'])]
                for job in concurrent.futures.as_completed(jobs):print('DATASET_COMPLETE',*job.result(),flush=True)
            locks={}
            for system,seed,ratio,method in itertools.product(c['systems'],c['seeds'],c['noise_ratios'],c['method_order']):
                trial=out/system/f'seed_{seed}'/pilot.slot(ratio)/method
                locks[str(trial.relative_to(out))]=dict(status_sha256=file_hash(trial/'status.json'),
                    coefficients_sha256=file_hash(trial/'coefficients.npz') if (trial/'coefficients.npz').exists() else 'UNAVAILABLE')
            for name,expected in preserved.items():assert file_hash(out/name)==expected,name
            pilot.save_json(out/'selection_locked.json',dict(trials=locks,test_evaluated_before_lock=False,execution_resumed=True))
            records,summary=r.d30.evaluate(c,out)
        for name,expected in locks.items():
            assert file_hash(out/name/'status.json')==expected['status_sha256']
            if expected['coefficients_sha256']!='UNAVAILABLE':assert file_hash(out/name/'coefficients.npz')==expected['coefficients_sha256']
        pilot.save_json(out/'integrity_after.json',r.integrity())
        states=[a['status'] for a in records if a['order']=='triple']
        result=dict(status='COMPLETE_WITH_FAILURES' if 'FAILED' in states else 'COMPLETE',models=len(states),
            counts={s:states.count(s) for s in ('OK','NA','FAILED')},systems=c['systems'],physical_nodes=15,
            pair_edges=45,triple_edges=15,resumed_wall_seconds=time.perf_counter()-started,
            execution_resumed=True,preserved_files_unchanged=len(preserved))
        pilot.save_json(out/'completion.json',result);print(json.dumps(result),flush=True)
    except Exception:
        pilot.save_json(out/'resume_failure.json',dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
