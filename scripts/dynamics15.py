"""15 physical nodes, five dynamics, full native cubic libraries, isolated processes."""
import argparse
import concurrent.futures
import importlib.util
import itertools
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time
import tomllib
import traceback

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits
from src.bridge.numeric_io import file_hash,write_rows,write_toml
from src.preprocessing.taylor import restore_coefficients
from src.solvers.methods import fit_node
from src.solvers.adaptive_lasso import BICNotApplicable
from src.solvers.large_library import initial_ridge,stlsq_dual,validation_lasso_working_set
spec=importlib.util.spec_from_file_location('d30',ROOT/'scripts/dynamics30.py')
d30=importlib.util.module_from_spec(spec);spec.loader.exec_module(d30)
pilot=d30.pilot
METHODS=['M0D','M0sD','M1B','M2B','M1vW','M3W']
CTX=None


def make_config(path):
    rng=np.random.default_rng(2026102200);n=15;pairs=[];triples=[]
    for target in range(1,n+1):
        first=target%n+1
        other=rng.choice([j for j in range(1,n+1) if j not in (target,first)],2,replace=False)
        for source in (first,*other):pairs.append([target,int(source),float(rng.uniform(.2,.6))])
        j,k=sorted((first,int(other[0])));triples.append([target,j,k,float(rng.uniform(.1,.3))])
    c=dict(experiment='dynamics15_five_systems_full_cubic_v1',physical_nodes=n,
        systems=['kuramoto','lorenz','lotka_volterra','stuart_landau','sis'],
        seeds=[2026102201,2026102202],topology_seed=2026102200,noise_master_seed=2026102203,
        method_order=METHODS,dmax=3,train_samples=1024,validation_samples=256,test_samples=256,
        noise_ratios=[0.,.05],decode_threshold=1e-8,workers=12,blas_threads=1,
        sampling='Independent initial-condition snapshots with analytic derivatives, one sample per independent preparation; not a long-trajectory benchmark',
        candidate_policy='Full native THIS degree<=3 library; no Pearson, no known-channel or true-topology screening',
        scope='Two-seed five-system engineering benchmark; changes node count, sampling and screening jointly, not a controlled node-count-only ablation',
        stlsq=dict(lambda_=0.1,rho=1.,niter=10),
        lasso=dict(max_iter=100000,tol=1e-10,kkt_tolerance=1e-6,validation_tolerance=1e-12,
            alpha_fractions=[1.,.3,.1,.03,.01,.003,.001,.0003,.0001,1e-5,1e-6,1e-8],working_set_rounds=60,working_set_add=64),
        adaptive=dict(initialization='ridge',standardize=False,fit_intercept=True,ridge_rho=1.,gamma=1.,eps=1e-6,
            bic_noise_floor=1e-12,bic_condition_limit=1e8))
    c['stlsq']['lambda']=c['stlsq'].pop('lambda_')
    for system,group in [('kuramoto',1),('lorenz',3),('lotka_volterra',1),('stuart_landau',2),('sis',1)]:
        dims=n*group
        center=(rng.choice([-1.,1.],dims)*rng.uniform(.3,.8,dims)).tolist()
        if system=='lotka_volterra':center=rng.uniform(.8,1.2,dims).tolist()
        if system=='sis':center=rng.uniform(.4,.6,dims).tolist()
        s=dict(physical_nodes=n,center=center,box_width=.6,coordinates_per_node=group,
            pair_edges=[r.copy() for r in pairs],triple_edges=[r.copy() for r in triples])
        if system=='kuramoto':s.update(omega=rng.uniform(-.2,.2,n).tolist(),source='https://arxiv.org/html/2402.00078v2')
        elif system=='lorenz':s.update(sigma=10.,rho=28.,beta=8/3,source='https://arxiv.org/html/2402.00078v2')
        elif system=='lotka_volterra':
            s.update(growth=rng.uniform(.8,1.2,n).tolist(),self_damping=1.,source='https://arxiv.org/html/2409.10990v1')
            for r in s['pair_edges']:r[-1]*=-1
            for r in s['triple_edges']:r[-1]*=-1
        elif system=='stuart_landau':s.update(growth=1.,omega=1.,source='https://arxiv.org/html/2306.04614v1')
        elif system=='sis':s.update(recovery=1.,source='https://www.nature.com/articles/s41467-019-10431-6')
        c[system]=s
    if path.exists():raise FileExistsError(path)
    write_toml(path,c)


def initialize(data,c,root,trial,method):
    global CTX
    threadpool_limits(limits=1)
    folder=data['folder']
    Z=np.load(folder/'Z_train.npy',mmap_mode='r');Zv=np.load(folder/'Z_validation.npy',mmap_mode='r')
    Y=np.loadtxt(root/'Y_noisy.csv',delimiter=',',ndmin=2)
    init=np.load(root/'Ridge_initial_standardized.npy',mmap_mode='r') if method in ('M1vW','M3W') else None
    CTX=(data,c,trial,method,Z,Zv,Y,init)


def one(node):
    data,c,trial,method,Z,Zv,Y,init=CTX
    train=data['splits']=='train';val=data['splits']=='validation'
    coef=None;started=time.perf_counter()
    try:
        if method in ('M1B','M2B'):
            config=dict(c,lasso={k:v for k,v in c['lasso'].items() if not k.startswith('working_set_')})
            coef,diag=fit_node('M1' if method=='M1B' else 'M2',Z,Y[node,train],config)
        else:
            a=np.ones(Z.shape[1]-1) if method=='M1vW' else (np.abs(init[node])+c['adaptive']['eps'])**c['adaptive']['gamma']
            coef,diag=validation_lasso_working_set(Z[:,1:],Y[node,train],Zv[:,1:],Y[node,val],a,c['lasso'])
        np.savez_compressed(trial/f'node_{node+1:03d}_solution.npz',coefficient_standardized=coef)
        status=dict(status='OK',node_julia_index=node+1,**diag)
    except BICNotApplicable as e:status=dict(status='NA',node_julia_index=node+1,reason=str(e))
    except Exception as e:
        coef=None;status=dict(status='FAILED',node_julia_index=node+1,reason=str(e),traceback=traceback.format_exc(),diagnostics=getattr(e,'diagnostics',{}))
        if hasattr(e,'coefficient'):np.savez_compressed(trial/f'node_{node+1:03d}_unaccepted.npz',coefficient_standardized=e.coefficient)
    status.update(worker_pid=os.getpid(),process_isolated=True,wall_seconds=time.perf_counter()-started)
    pilot.save_json(trial/f'node_{node+1:03d}.json',status)
    return node,coef,status['status']


def fit_cell(data,c,out,system,seed,ratio):
    folder=data['folder'];root=out/system/f'seed_{seed}'/pilot.slot(ratio)
    train=data['splits']=='train';Y=np.loadtxt(root/'Y_noisy.csv',delimiter=',',ndmin=2)
    phi=np.load(folder/'theta.npy',mmap_mode='r');Z=np.load(folder/'Z_train.npy',mmap_mode='r')
    means=np.load(folder/'feature_means.npy');scales=np.load(folder/'feature_scales.npy')
    for method,design in [('M0D',np.asfortranarray(phi[train])),('M0sD',Z)]:
        trial=root/method;trial.mkdir();started=time.perf_counter()
        C,diag=stlsq_dual(design,Y[:,train].T,c['stlsq']['lambda'],c['stlsq']['rho'],c['stlsq']['niter'])
        if method=='M0sD':C=restore_coefficients(C,means,scales)
        assert np.isfinite(C).all();sparse.save_npz(trial/'coefficients.npz',sparse.csr_matrix(C))
        pilot.save_json(trial/'status.json',dict(status='OK',method=method,fit_seconds=time.perf_counter()-started,
            numerical_variant='Original STLSQ algebra via primal/dual linear solves',**diag))
        print('FIT',system,seed,ratio,method,'OK',flush=True)
    init=initial_ridge(Z[:,1:],Y[:,train].T,c['adaptive']['ridge_rho'])
    np.save(root/'Ridge_initial_standardized.npy',init)
    for method in ('M1B','M2B','M1vW','M3W'):
        trial=root/method;trial.mkdir();started=time.perf_counter()
        if method in ('M1B','M2B') and c['train_samples']<=data['meta']['P']:
            pilot.save_json(trial/'status.json',dict(status='NA',method=method,reason='Guarded BIC requires K>P',K=c['train_samples'],P=data['meta']['P']))
            print('FIT',system,seed,ratio,method,'NA',flush=True);continue
        C=np.zeros((data['meta']['N'],data['meta']['P']));states=[]
        with concurrent.futures.ProcessPoolExecutor(max_workers=c['workers'],mp_context=multiprocessing.get_context('fork'),
                initializer=initialize,initargs=(data,c,root,trial,method)) as pool:
            jobs=[pool.submit(one,node) for node in range(data['meta']['N'])]
            for finished,future in enumerate(concurrent.futures.as_completed(jobs),1):
                node,coef,state=future.result();states.append(state)
                if coef is not None:C[node]=restore_coefficients(coef[None,:],means,scales)[0]
                if finished%15==0 or finished==len(jobs):print('NODE',system,seed,ratio,method,f'{finished}/{len(jobs)}',flush=True)
        state='FAILED' if 'FAILED' in states else 'NA' if 'NA' in states else 'OK'
        sparse.save_npz(trial/('coefficients.npz' if state=='OK' else 'available_nodes_only.npz'),sparse.csr_matrix(C))
        pilot.save_json(trial/'status.json',dict(status=state,method=method,fit_seconds=time.perf_counter()-started,
            node_counts={s:states.count(s) for s in ('OK','NA','FAILED')},full_library=True,process_isolated=True))
        print('FIT',system,seed,ratio,method,state,flush=True)


def integrity():
    result=d30.integrity()
    old=ROOT/'results/dynamics30/pearson30_process_20261021'
    if not old.exists():
        # Explicit retirement, never silently treat missing historical files as intact.
        receipt_path=ROOT/'docs/maintenance/cleanup_dynamics30_20261023/cleanup_receipt.json'
        receipt=json.loads(receipt_path.read_text())
        assert receipt['status']=='DELETED_AND_VERIFIED',receipt_path
        assert 'results/dynamics30' in receipt['targets']
        assert not (ROOT/'results/dynamics30').exists()
        assert file_hash(receipt_path.with_name('deleted_files.csv'))==receipt['inventory_sha256']
        return dict(**result,prior_30_node_artifacts_status='DELETED_BY_USER_REQUEST',
                    prior_30_node_deletion_receipt=str(receipt_path.relative_to(ROOT)))
    records=pilot.rows(old/'artifact_manifest.csv')
    for r in records:assert file_hash(old/r['path'])==r['sha256'],r['path']
    return dict(**result,prior_30_node_artifacts_unchanged=len(records))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/dynamics15_five_systems_v1.toml')
    parser.add_argument('--make-config',action='store_true')
    parser.add_argument('--label',default='five_systems_full_20261022')
    args=parser.parse_args()
    if args.make_config:make_config(args.config);return
    assert Path(args.label).name==args.label and args.label not in ('.','..')
    c=tomllib.loads(args.config.read_text());out=ROOT/'results/dynamics15'/args.label
    out.mkdir(parents=True,exist_ok=False);(out/'logs').mkdir();(out/'julia_depot').mkdir()
    os.environ.update(PYTHONDONTWRITEBYTECODE='1',JULIA_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        JULIA_DEPOT_PATH=str(out/'julia_depot')+':'+str(ROOT/'.julia-depot-r0'))
    started=time.perf_counter()
    try:
        pilot.save_json(out/'integrity_before.json',integrity());(out/'config.toml').write_bytes(args.config.read_bytes())
        sources=['scripts/dynamics15.py','scripts/dynamics15_prepare.jl','src/data/multisystem_dynamics.jl',
            'src/data/paper_dynamics.jl','src/solvers/large_library.py','src/solvers/adaptive_lasso.py',
            'src/solvers/plain_lasso.py','scripts/dynamics30.py','scripts/dynamics_pilot.py','src/metrics/structure.py']
        pilot.save_json(out/'protocol.json',dict(config=c,source_sha256={name:file_hash(ROOT/name) for name in sources},
            candidate_screening=False,truth_used_for_training=False,selection_locked_before_test=True,
            sources={system:c[system]['source'] for system in c['systems']}))
        runtime=['/home/yg/.local/opt/julia-1.9.4/bin/julia','--startup-file=no','--project='+str(ROOT/'environment/r0/julia')]
        for name,cmd in [('dynamics_tests',runtime+[str(ROOT/'tests/test_multisystem_dynamics.jl')]),
            ('prepare',runtime+[str(ROOT/'scripts/dynamics15_prepare.jl'),str(args.config),str(out)])]:
            with (out/'logs'/f'julia_{name}.log').open('x') as log:result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError(f'Julia {name} failed')
        with threadpool_limits(limits=1):
            for system,seed in itertools.product(c['systems'],c['seeds']):
                data=d30.load_data(out,system,seed);d30.features(data);d30.prepare_noise(data,c,out,system,seed)
                for ratio in c['noise_ratios']:fit_cell(data,c,out,system,seed,ratio)
            locks={}
            for system,seed,ratio,method in itertools.product(c['systems'],c['seeds'],c['noise_ratios'],METHODS):
                trial=out/system/f'seed_{seed}'/pilot.slot(ratio)/method
                locks[str(trial.relative_to(out))]=dict(status_sha256=file_hash(trial/'status.json'),
                    coefficients_sha256=file_hash(trial/'coefficients.npz') if (trial/'coefficients.npz').exists() else 'UNAVAILABLE')
            pilot.save_json(out/'selection_locked.json',dict(trials=locks,test_evaluated_before_lock=False))
            records,summary=d30.evaluate(c,out)
        for name,expected in locks.items():
            assert file_hash(out/name/'status.json')==expected['status_sha256']
            if expected['coefficients_sha256']!='UNAVAILABLE':assert file_hash(out/name/'coefficients.npz')==expected['coefficients_sha256']
        pilot.save_json(out/'integrity_after.json',integrity())
        states=[r['status'] for r in records if r['order']=='triple']
        result=dict(status='COMPLETE_WITH_FAILURES' if 'FAILED' in states else 'COMPLETE',models=len(states),
            counts={s:states.count(s) for s in ('OK','NA','FAILED')},systems=c['systems'],physical_nodes=15,
            pair_edges=45,triple_edges=15,wall_seconds=time.perf_counter()-started)
        pilot.save_json(out/'completion.json',result);print(json.dumps(result),flush=True)
    except Exception:
        pilot.save_json(out/'failure.json',dict(traceback=traceback.format_exc(),wall_seconds=time.perf_counter()-started));raise


if __name__=='__main__':main()
