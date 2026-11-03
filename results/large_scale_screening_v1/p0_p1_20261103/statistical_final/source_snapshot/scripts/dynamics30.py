"""Thirty physical-node pilots with a complete cubic THIS library and explicit scalable variants."""
import argparse
import concurrent.futures
import csv
import importlib.util
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import tomllib
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits
from src.bridge.numeric_io import file_hash,write_toml,write_rows
from src.data.contracts import validate_splits
from src.metrics.structure import binary_metrics
from src.preprocessing.taylor import restore_coefficients
from src.solvers.large_library import initial_ridge,stlsq_dual,validation_lasso_working_set

spec=importlib.util.spec_from_file_location('pilot',ROOT/'scripts/dynamics_pilot.py')
pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)
METHODS=['M0D','M0sD','M1B','M2B','M1vW','M3W']


def make_config(path):
    c=tomllib.loads((ROOT/'configs/dynamics_pilot.toml').read_text())
    c.update(experiment='dynamics30_full_cubic_v1',seeds=[2026102105,2026102106],
             method_order=METHODS,workers=4,blas_threads=2,
             scope='Two-seed 30-physical-node synthetic-topology pilot; full library; numerical variants explicitly identified',
             topology_seed=2026102104)
    rng=np.random.default_rng(c['topology_seed']);n=30
    pairs=[];triples=[]
    for target in range(1,n+1):
        first=target%n+1
        second=int(rng.choice([j for j in range(1,n+1) if j not in (target,first)]))
        third=int(rng.choice([j for j in range(1,n+1) if j not in (target,first,second)]))
        for source in (first,second,third):pairs.append([target,source,float(rng.uniform(.2,.6))])
        j,k=sorted([first,second]);triples.append([target,j,k,float(rng.uniform(.1,.3))])
    for system in c['systems']:
        s=c[system];s['physical_nodes']=n;s['pair_edges']=pairs;s['triple_edges']=triples
        dimensions=n if system=='kuramoto' else 3*n
        s['center']=(rng.choice([-1.,1.],dimensions)*rng.uniform(.2,.8,dimensions)).tolist()
        if system=='kuramoto':s['omega']=rng.uniform(-.2,.2,n).tolist()
    c['lasso'].update(working_set_rounds=40,working_set_add=128)
    c['numerical_variants']=dict(M0D='Original raw STLSQ algebra using dual/primal solves instead of explicit full inverse',
        M0sD='Same algebra on common standardized features',M1B='Original guarded BIC; NA at K<=P',
        M2B='Original guarded Adaptive BIC; NA at K<=P',M1vW='Full-library ordinary validation LASSO with global KKT working sets',
        M3W='Full-library Ridge Adaptive validation LASSO with global KKT working sets')
    if path.exists():raise FileExistsError(path)
    write_toml(path,c)


def load_data(out,system,seed):
    folder=out/system/f'seed_{seed}'/'data'
    meta=tomllib.loads((folder/'metadata.toml').read_text())
    for name,expected in meta['file_sha256'].items():assert file_hash(folder/name)==expected,name
    manifest=pilot.rows(folder/'sample_manifest.csv')
    splits=np.array([r['split'] for r in manifest]);groups=np.array([int(r['trajectory_id']) for r in manifest])
    validate_splits(groups,splits)
    return dict(folder=folder,meta=meta,splits=splits,groups=groups,
                Y=np.loadtxt(folder/'Y.csv',delimiter=',',ndmin=2),
                d=np.loadtxt(folder/'monomial_indices.csv',delimiter=',',dtype=int,ndmin=2))


def features(data):
    folder=data['folder'];U=np.loadtxt(folder/'centered_states.csv',delimiter=',',ndmin=2)
    d=data['d'];k=U.shape[1];p=len(d)
    phi=np.lib.format.open_memmap(folder/'theta.npy',mode='w+',dtype='float64',shape=(k,p),fortran_order=True)
    for start in range(0,p,1024):
        ids=d[start:start+1024];value=np.ones((k,len(ids)))
        for col in range(ids.shape[1]):
            v=ids[:,col];value*=np.where(v[None,:]>0,U[np.maximum(v-1,0),:].T,1.)
        phi[:,start:start+len(ids)]=value
    phi.flush()
    columns=np.array(data['meta']['probe_columns_julia'])-1
    probe=np.loadtxt(folder/'theta_probe.csv',delimiter=',',ndmin=2)
    difference=float(np.max(np.abs(phi[columns].T-probe)))
    assert np.allclose(phi[columns].T,probe,atol=1e-13,rtol=1e-13)
    train=data['splits']=='train';raw=np.asfortranarray(phi[train])
    means=raw.mean(axis=0);means[0]=0.
    scales=np.sqrt(np.mean((raw-means)**2,axis=0));scales[0]=1.
    assert np.all(scales>0)
    np.save(folder/'feature_means.npy',means);np.save(folder/'feature_scales.npy',scales)
    for split in ('train','validation','test'):
        mask=data['splits']==split
        Z=np.asfortranarray((phi[mask]-means)/scales)
        np.save(folder/f'Z_{split}.npy',Z)
    pilot.save_json(folder/'feature_audit.json',dict(full_library_features=p,
        Julia_probe_max_abs_difference=difference,training_samples=int(train.sum()),
        feature_screening=False,degree=3,exchange='Exact upstream d order, Float64 NPY arrays; Julia CSV probes checked at 1e-13'))
    print(f'FEATURES_READY {data["meta"]["system"]} seed={data["meta"]["seed"]} P={p}',flush=True)


def prepare_noise(data,c,out,system,seed):
    # Reuse the already tested split-specific response noise construction unchanged.
    original=dict(Y=data['Y'],splits=data['splits'],directory=data['folder'])
    pilot.generate_noise(original,c,out,system,seed)


def fit_cell(data,c,out,system,seed,ratio):
    folder=data['folder'];root=out/system/f'seed_{seed}'/pilot.slot(ratio)
    train=data['splits']=='train';val=data['splits']=='validation'
    phi=np.load(folder/'theta.npy',mmap_mode='r')
    Z=np.load(folder/'Z_train.npy',mmap_mode='r');Zv=np.load(folder/'Z_validation.npy',mmap_mode='r')
    Y=np.loadtxt(root/'Y_noisy.csv',delimiter=',',ndmin=2)
    means=np.load(folder/'feature_means.npy');scales=np.load(folder/'feature_scales.npy')
    for method,design in [('M0D',np.asfortranarray(phi[train])),('M0sD',Z)]:
        trial=root/method;trial.mkdir();started=time.perf_counter()
        C,diag=stlsq_dual(design,Y[:,train].T,threshold=c['stlsq']['lambda'],rho=c['stlsq']['rho'],niter=c['stlsq']['niter'])
        if method=='M0sD':C=restore_coefficients(C,means,scales)
        sparse.save_npz(trial/'coefficients.npz',sparse.csr_matrix(C))
        pilot.save_json(trial/'status.json',dict(status='OK',method=method,fit_seconds=time.perf_counter()-started,**diag))
        print(f'FIT {system} seed={seed} noise={ratio} {method} OK',flush=True)
    for method in ('M1B','M2B'):
        trial=root/method;trial.mkdir()
        assert int(train.sum())<=data['meta']['P']
        pilot.save_json(trial/'status.json',dict(status='NA',method=method,fit_seconds=0.,
            reason=f'Original BIC prerequisite K>P failed: K={train.sum()}, P={data["meta"]["P"]}; no substitute variance or screened BIC'))
    X=Z[:,1:];Xv=Zv[:,1:]
    init=initial_ridge(X,Y[:,train].T,c['adaptive']['ridge_rho'])
    np.save(root/'Ridge_initial_standardized.npy',init)
    for method in ('M1vW','M3W'):
        trial=root/method;trial.mkdir();started=time.perf_counter()
        def fit_node(node):
            try:
                a=np.ones(X.shape[1]) if method=='M1vW' else (np.abs(init[node])+c['adaptive']['eps'])**c['adaptive']['gamma']
                Cz,diag=validation_lasso_working_set(X,Y[node,train],Xv,Y[node,val],a,c['lasso'])
                np.savez_compressed(trial/f'node_{node+1:03d}_solution.npz',coefficient_standardized=Cz)
                pilot.save_json(trial/f'node_{node+1:03d}.json',dict(status='OK',node_julia_index=node+1,**diag))
                return node,Cz
            except Exception as e:
                pilot.save_json(trial/f'node_{node+1:03d}.json',dict(status='FAILED',node_julia_index=node+1,
                    reason=str(e),diagnostics=getattr(e,'diagnostics',{}),traceback=traceback.format_exc()))
                if hasattr(e,'coefficient'):
                    np.savez_compressed(trial/f'node_{node+1:03d}_unaccepted.npz',coefficient_standardized=e.coefficient)
                return node,None
        C=np.zeros((Y.shape[0],Z.shape[1]));bad=[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=c['workers']) as pool:
            futures=[pool.submit(fit_node,node) for node in range(len(Y))]
            for finished,future in enumerate(concurrent.futures.as_completed(futures),1):
                node,Cz=future.result()
                if Cz is None:bad.append(node+1)
                else:C[node]=Cz/scales;C[node,0]=Cz[0]-C[node,1:]@means[1:]
                if finished%10==0 or finished==len(Y):
                    print(f'NODE_PROGRESS {system} seed={seed} noise={ratio} {method} {finished}/{len(Y)}',flush=True)
        label='FAILED' if bad else 'OK'
        if not bad:sparse.save_npz(trial/'coefficients.npz',sparse.csr_matrix(C))
        else:sparse.save_npz(trial/'available_nodes_only.npz',sparse.csr_matrix(C))
        pilot.save_json(trial/'status.json',dict(status=label,method=method,failed_nodes=bad,
            fit_seconds=time.perf_counter()-started,full_library_features=Z.shape[1],truth_screening=False))
        print(f'FIT {system} seed={seed} noise={ratio} {method} {label}',flush=True)


def evaluate(c,out):
    records=[];edge_rows=[]
    for system,seed in itertools.product(c['systems'],c['seeds']):
        data=load_data(out,system,seed);test=data['splits']=='test'
        phi=np.load(data['folder']/'theta.npy',mmap_mode='r')
        truth={2:{(int(i),int(j)) for i,j,_ in c[system]['pair_edges']},
               3:{(int(i),*sorted([int(j),int(k)])) for i,j,k,_ in c[system]['triple_edges']}}
        for ratio,method in itertools.product(c['noise_ratios'],c['method_order']):
            trial=out/system/f'seed_{seed}'/pilot.slot(ratio)/method
            status=json.loads((trial/'status.json').read_text())
            base=dict(system=system,seed=seed,noise_ratio=ratio,method=method,status=status['status'])
            if status['status']=='OK':
                C=sparse.load_npz(trial/'coefficients.npz')
                prediction=C@phi[test].T;Y=data['Y'][:,test]
                nmse=float(np.sum((prediction-Y)**2)/np.sum(Y**2))
                dense=C.toarray()
            for order,label in ((2,'pairwise'),(3,'triple')):
                if status['status']!='OK':records.append(dict(**base,order=label));continue
                keys,scores=pilot.physical_scores(dense,data['d'],data['meta']['physical_node_of_variable'],order)
                metrics=binary_metrics(np.array([key in truth[order] for key in keys]),scores,c['decode_threshold'])
                records.append(dict(**base,order=label,clean_test_NMSE=nmse,**metrics))
                for key,score in zip(keys,scores):
                    edge_rows.append(dict(**base,order=label,target=key[0],source_1=key[1],source_2=key[2] if order==3 else 0,
                                          truth=int(key in truth[order]),score=float(score)))
    fields=list(dict.fromkeys(k for r in records for k in r));write_rows(out/'trials.csv',records,fields)
    write_rows(out/'edge_scores.csv',edge_rows)
    summaries=[]
    for system,ratio,method,order in itertools.product(c['systems'],c['noise_ratios'],c['method_order'],('pairwise','triple')):
        chosen=[r for r in records if (r['system'],r['noise_ratio'],r['method'],r['order'])==(system,ratio,method,order)]
        good=[r for r in chosen if r['status']=='OK']
        record=dict(system=system,noise_ratio=ratio,method=method,order=order,attempted=len(chosen),
                    OK=len(good),NA=sum(r['status']=='NA' for r in chosen),FAILED=sum(r['status']=='FAILED' for r in chosen))
        for key in ('F1','precision','recall','AUPRC_AP','clean_test_NMSE'):
            record[key+'_mean']=float(np.mean([r[key] for r in good])) if good else ''
        summaries.append(record)
    write_rows(out/'summary.csv',summaries)
    return records,summaries


def integrity():
    baseline=pilot.protect_legacy()
    old=ROOT/'results/dynamics_pilot/kuramoto_lorenz_20261021'
    for r in pilot.rows(old/'artifact_manifest.csv'):
        assert file_hash(old/r['path'])==r['sha256'],r['path']
    return dict(**baseline,small_dynamics_pilot_unchanged=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--make-config',action='store_true')
    parser.add_argument('--config',type=Path,default=ROOT/'configs/dynamics30_v1.toml')
    parser.add_argument('--label',default='synthetic30_20261021')
    args=parser.parse_args()
    if args.make_config:make_config(args.config);return
    assert Path(args.label).name==args.label and args.label not in ('.','..')
    c=tomllib.loads(args.config.read_text());out=ROOT/'results/dynamics30'/args.label
    out.mkdir(parents=True,exist_ok=False);(out/'logs').mkdir();(out/'julia_depot').mkdir()
    os.environ.update(PYTHONDONTWRITEBYTECODE='1',JULIA_NUM_THREADS='1',
        JULIA_DEPOT_PATH=str(out/'julia_depot')+':'+str(ROOT/'.julia-depot-r0'))
    started=time.perf_counter()
    try:
        pilot.save_json(out/'integrity_before.json',integrity())
        (out/'config.toml').write_bytes(args.config.read_bytes())
        sources=[ROOT/'scripts/dynamics30.py',ROOT/'scripts/dynamics30_prepare.jl',ROOT/'src/solvers/large_library.py',
                 ROOT/'src/data/paper_dynamics.jl',ROOT/'scripts/dynamics_pilot.py']
        pilot.save_json(out/'protocol.json',dict(created_date='2026-10-21 Asia/Shanghai',scope=c['scope'],config=c,
            original_pilot_preserved=True,source_sha256={str(p.relative_to(ROOT)):file_hash(p) for p in sources},
            native_dictionary_order=True,full_cubic_library=True,truth_screening=False))
        command=['/home/yg/.local/opt/julia-1.9.4/bin/julia','--startup-file=no','--project='+str(ROOT/'environment/r0/julia'),
                 str(ROOT/'scripts/dynamics30_prepare.jl'),str(args.config),str(out)]
        with (out/'logs/julia_prepare.log').open('x') as log:
            result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        assert result.returncode==0,'Julia prepare failed; see log'
        with threadpool_limits(limits=c['blas_threads']):
            for system,seed in itertools.product(c['systems'],c['seeds']):
                data=load_data(out,system,seed);features(data);prepare_noise(data,c,out,system,seed)
                for ratio in c['noise_ratios']:fit_cell(data,c,out,system,seed,ratio)
            locks={}
            for system,seed,ratio,method in itertools.product(c['systems'],c['seeds'],c['noise_ratios'],METHODS):
                trial=out/system/f'seed_{seed}'/pilot.slot(ratio)/method
                locks[str(trial.relative_to(out))]=dict(status_sha256=file_hash(trial/'status.json'),
                    coefficients_sha256=file_hash(trial/'coefficients.npz') if (trial/'coefficients.npz').exists() else 'UNAVAILABLE')
            pilot.save_json(out/'selection_locked.json',dict(test_evaluated_before_lock=False,trials=locks))
            records,summaries=evaluate(c,out)
        pilot.save_json(out/'integrity_after.json',integrity())
        for path,expected in locks.items():
            assert file_hash(out/path/'status.json')==expected['status_sha256']
            if expected['coefficients_sha256']!='UNAVAILABLE':assert file_hash(out/path/'coefficients.npz')==expected['coefficients_sha256']
        statuses=[r['status'] for r in records if r['order']=='triple']
        completion=dict(status='COMPLETE_WITH_FAILURES' if 'FAILED' in statuses else 'COMPLETE',trials=len(statuses),
            counts={s:statuses.count(s) for s in ('OK','NA','FAILED')},wall_seconds=time.perf_counter()-started,
            physical_nodes_per_system=30,old_results_preserved=True,complete_cubic_library=True)
        pilot.save_json(out/'completion.json',completion)
        print(json.dumps(completion),flush=True)
    except Exception:
        pilot.save_json(out/'failure.json',dict(traceback=traceback.format_exc(),wall_seconds=time.perf_counter()-started))
        raise


if __name__=='__main__':main()
