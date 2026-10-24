"""Locked V2 design, shared preprocessing, seed namespaces and solver adapter."""
from pathlib import Path
from copy import deepcopy
import csv
import hashlib
import json
import tomllib
import numpy as np
from src.bridge.numeric_io import write_toml, write_rows
from src.preprocessing.taylor import restore_coefficients
from src.solvers.large_library import initial_ridge
from src.solvers.lars_validation import validation_lasso_accelerated
from src.solvers.stlsq_validation import stlsq_validation

ROOT = Path(__file__).resolve().parents[2]
SYSTEMS = ['kuramoto','lorenz','lotka_volterra','stuart_landau','sis']
METHODS = ['STLSQ_val','LASSO_val','Adaptive_val']
NAMES = dict(zip(SYSTEMS,['Kuramoto','Lorenz','Lotka–Volterra','Stuart–Landau','SIS']))
FRACTIONS = [1.,.3,.1,.03,.01,.003,.001,.0003,.0001,1e-5,1e-6,1e-8]


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def save_json(path, data):
    def convert(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(data,default=convert,ensure_ascii=False,indent=2)+'\n')


def load_csv(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))


def seed(master, domain, *ids):
    return int(np.random.SeedSequence([master,domain,*ids]).generate_state(1)[0])


def seed_design(master=2026102301):
    rows=[]
    for network_id in range(1,11):
        ns=seed(master,1,network_id)
        ts=seed(master,2,ns,0); ws=seed(master,2,ns,1)
        for sample_id in range(1,4):
            ds=seed(master,3,network_id,sample_id)
            rows.append(dict(network_id=network_id, sample_id=sample_id, network_seed=ns,
                topology_seed=ts, weight_seed=ws, data_seed=ds,
                noise_seed=seed(master,4,ns,ds), instance_id=f'n{network_id:02d}_d{sample_id:02d}'))
    assert len({r['data_seed'] for r in rows})==30
    assert len({r['noise_seed'] for r in rows})==30
    assert len({r['topology_seed'] for r in rows})==10
    return rows


def network(topology_seed, weight_seed):
    tr=np.random.default_rng(topology_seed);wr=np.random.default_rng(weight_seed)
    pairs=[];triples=[]
    for target in range(1,16):
        inputs=sorted(int(x) for x in tr.choice([x for x in range(1,16) if x!=target],3,replace=False))
        sources=sorted(int(x) for x in tr.choice(inputs,2,replace=False))
        pairs += [[target,j,float(wr.uniform(.2,.6))] for j in inputs]
        triples.append([target,*sources,float(wr.uniform(.1,.3))])
    validate_network(pairs,triples)
    return pairs,triples


def validate_network(pairs, triples):
    p={(int(i),int(j)) for i,j,_ in pairs};t={(int(i),int(j),int(k)) for i,j,k,_ in triples}
    assert len(pairs)==len(p)==45 and len(triples)==len(t)==15
    for i in range(1,16):
        assert sum(a==i for a,b in p)==3
        assert sum(a==i for a,b,c in t)==1
    assert all(i!=j and 1<=i<=15 and 1<=j<=15 for i,j in p)
    assert all(len({i,j,k})==3 and j<k and (i,j) in p and (i,k) in p for i,j,k in t)


def make_plan(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    old=tomllib.loads((ROOT/'configs/dynamics15_five_systems_certified_v1_1.toml').read_text())
    c=deepcopy(old)
    for field in ['seeds','topology_seed','noise_master_seed','numerical_variants']:
        c.pop(field,None)
    for s in SYSTEMS:
        c[s].pop('pair_edges');c[s].pop('triple_edges')
    c.update(experiment='15_node_fair_comparison_v2', systems=SYSTEMS, method_order=METHODS,
        scope='10 independent network realizations (topology and weights), 3 independent data preparations each; paired five-system benchmark',
        numerical_backend='STLSQ Ridge and hard threshold; unchanged certified LARS/working-set LASSO backends',
        repetitions=seed_design(), workers=8, blas_threads=1,
        bootstrap_seed=2026102302, bootstrap_resamples=2000,
        sampling='Independent t=0 state/analytic derivative preparations, not an observed time series',
        tie_rule='first descending relative penalty within tolerance * validation response energy',
        selection_scope='validation prediction MSE; no guarantee of structure F1 optimum',
        standardization='training feature means and centered RMS, exclude constant from penalty',
        stlsq=dict(rho=1.,niter=10,alpha_fractions=FRACTIONS,validation_tolerance=1e-12),
        planned_model_count=900, formal_run_requires_explicit_confirmation=True)
    c['lasso']['alpha_fractions']=FRACTIONS
    write_toml(out/'config_locked.toml',c)
    write_rows(out/'seed_manifest.csv',c['repetitions'])
    records=[];topologies=set()
    for r in c['repetitions'][::3]:
        pairs,triples=network(r['topology_seed'],r['weight_seed'])
        topology=tuple(tuple(x[:-1]) for x in pairs+triples)
        assert topology not in topologies;topologies.add(topology)
        for order,edges in [('pairwise',pairs),('triple',triples)]:
            for e in edges:
                records.append(dict(network_id=r['network_id'],network_seed=r['network_seed'],
                    topology_seed=r['topology_seed'],weight_seed=r['weight_seed'],order=order,
                    target=e[0],source_1=e[1],source_2=e[2] if order=='triple' else 0,absolute_weight=e[-1]))
    write_rows(out/'network_manifest.csv',records)
    save_json(out/'plan_lock.json',dict(files={f.name:sha(f) for f in out.iterdir() if f.is_file()},
        formal_experiment_run=False,independent_network_realizations=10,data_replicates_per_network=3,
        network_effect_includes_topology_and_weights=True))
    return c


def read_plan(path):
    path=Path(path)
    lock=json.loads((path/'plan_lock.json').read_text())
    for name,digest in lock['files'].items():assert sha(path/name)==digest,name
    return tomllib.loads((path/'config_locked.toml').read_text())


def instance_config(config, repetition, systems):
    c=deepcopy(config);pairs,triples=network(repetition['topology_seed'],repetition['weight_seed'])
    c['seeds']=[repetition['data_seed']]
    c['generation_systems']=list(systems)  # Do not change SYSTEMS: Julia uses its stable system code.
    for s in SYSTEMS:
        sign=-1 if s=='lotka_volterra' else 1
        c[s]['pair_edges']=[[*r[:-1],sign*r[-1]] for r in pairs]
        c[s]['triple_edges']=[[*r[:-1],sign*r[-1]] for r in triples]
    return c


def fit_scaling(phi_train):
    raw=np.asfortranarray(phi_train,dtype=float)
    assert np.array_equal(raw[:,0],np.ones(len(raw)))
    means=raw.mean(axis=0);means[0]=0.
    scales=np.sqrt(np.mean((raw-means)**2,axis=0));scales[0]=1.
    if not np.isfinite(scales).all() or np.any(scales<=0):raise ValueError('Degenerate full library')
    return means,scales


def standardize(phi, means, scales):
    return np.asfortranarray((phi-means)/scales)


def noise(Y, train, repetition, system_code, splits):
    sigma=np.std(Y[:,train],axis=1,ddof=0);unit=np.empty_like(Y);streams=[]
    for split_code,split in enumerate(['train','validation','test'],1):
        mask=splits==split
        for node in range(len(Y)):
            entropy=[repetition['noise_seed'],repetition['network_seed'],repetition['data_seed'],system_code,split_code,node+1]
            rng=np.random.default_rng(np.random.SeedSequence(entropy))
            unit[node,mask]=rng.standard_normal(mask.sum());streams.append(entropy)
    return unit,sigma,streams


def solve(method, Z, y, Zv, yv, ridge_initial, config):
    """Only shared training/validation arrays and solver options cross this boundary."""
    if Z.shape[1]!=Zv.shape[1] or not np.all(Z[:,0]==1) or not np.all(Zv[:,0]==1):
        raise ValueError('Require one common bias column')
    X,Xv=Z[:,1:],Zv[:,1:]
    if np.max(np.abs(X.mean(axis=0)),initial=0.)>1e-11:
        raise ValueError('Features were not externally standardized on training data')
    if method=='STLSQ_val':
        coef,diag=stlsq_validation(X,y,Xv,yv,config['stlsq'],initial=ridge_initial)
    elif method in ('LASSO_val','Adaptive_val'):
        a=np.ones(X.shape[1]) if method=='LASSO_val' else (np.abs(ridge_initial)+1e-6)
        coef,diag=validation_lasso_accelerated(X,y,Xv,yv,a,config['lasso'])
        assert diag['kkt_scaled_violation']<=config['lasso']['kkt_tolerance']
        diag['parameter']=diag['alpha'];diag['parameter_reference']=diag['alpha_max_train']
        for row in diag['validation_path']:
            row.update(parameter=row['alpha'],relative_parameter=row['alpha']/diag['alpha_max_train'],status='OK')
    else:raise ValueError(method)
    diag.update(method=method,free_intercept=float(coef[0]),intercept_penalized=False,
        internally_standardized=False,parameter_boundary=diag['selected_path_index'] in (0,11),
        tie_rule=config['tie_rule'])
    assert len(diag['validation_path'])==12
    return coef,diag
