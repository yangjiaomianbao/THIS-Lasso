"""Sparse undirected Kuramoto network and trajectory-disjoint analytic data."""
import hashlib
import json
import time
import numpy as np
from scipy.integrate import solve_ivp
from src.data.contracts import validate_splits


def samples(n,fixed=False):
    return [1024,256,256] if fixed or n<=50 else [2048,512,512] if n<=200 else [4096,1024,1024]


def unique_edges(rng,n,order,count):
    import math
    count=min(int(count),math.comb(n,order));edges=set()
    while len(edges)<count: edges.add(tuple(sorted(rng.choice(n,order,replace=False).tolist())))
    return np.asarray(sorted(edges),dtype=int).reshape(-1,order)


def network(n,topology_seed,cfg):
    if n<3: raise ValueError('N>=3 required')
    rng=np.random.default_rng(np.random.SeedSequence([topology_seed,n,11]))
    pair=unique_edges(rng,n,2,round(n*cfg['mean_pair_degree']/2))
    tri=unique_edges(rng,n,3,round(n*cfg['mean_triad_degree']/3))
    directed2=np.array([(i,j) for a,b in pair for i,j in [(a,b),(b,a)]],int).reshape(-1,2)
    directed3=np.array([(i,*sorted(set(t)-{i})) for t in tri for i in t],int).reshape(-1,3)
    return dict(n=n,pair_edges=pair,triple_edges=tri,pairs=directed2,triples=directed3,
        degree2=np.bincount(directed2[:,0],minlength=n),degree3=np.bincount(directed3[:,0],minlength=n),
        omega=rng.uniform(-.25,.25,n),center=rng.uniform(-np.pi,np.pi,n),topology_seed=int(topology_seed))


def rhs(theta,net,cfg):
    theta=np.asarray(theta);out=np.broadcast_to(net['omega'],theta.shape).copy()
    p,t=net['pairs'],net['triples']
    if theta.ndim==1:
        if len(p): np.add.at(out,p[:,0],cfg['K2']*np.sin(theta[p[:,1]]-theta[p[:,0]])/np.maximum(net['degree2'][p[:,0]],1))
        if len(t): np.add.at(out,t[:,0],cfg['K3']*np.sin(theta[t[:,1]]+theta[t[:,2]]-2*theta[t[:,0]])/np.maximum(net['degree3'][t[:,0]],1))
    else:
        # Rows are samples; vectorize edges, never construct a dense N^3 tensor.
        for row in range(len(theta)): out[row]=rhs(theta[row],net,cfg)
    return out


def generate(n,topology_seed,trajectory_seed,cfg,counts=None):
    started=time.perf_counter();counts=counts or samples(n);points=cfg['points_per_trajectory']
    if any(c<=0 or c%points for c in counts): raise ValueError('Every split must contain whole 256-point trajectories')
    net=network(n,topology_seed,cfg);values=[];groups=[];splits=[];calls=[]
    timegrid=np.linspace(0,cfg['duration'],points)
    ident=0
    for split,count in zip(['train','validation','test'],counts):
        for _ in range(count//points):
            rng=np.random.default_rng(np.random.SeedSequence([trajectory_seed,n,22,ident]))
            initial=net['center']+rng.uniform(-cfg['initial_width']/2,cfg['initial_width']/2,n)
            sol=solve_ivp(lambda _,theta:rhs(theta,net,cfg),(0,cfg['duration']),initial,t_eval=timegrid,
                method='RK45',rtol=cfg['rtol'],atol=cfg['atol'])
            if not sol.success or sol.y.shape!=(n,points): raise RuntimeError('RK45 failed: '+sol.message)
            values.append(sol.y.T);groups.extend([ident]*points);splits.extend([split]*points);calls.append(sol.nfev);ident+=1
    phase=np.concatenate(values);dy=rhs(phase,net,cfg);groups=np.array(groups);splits=np.array(splits)
    validate_splits(groups,splits)
    order=np.abs(np.exp(1j*phase).mean(axis=1))
    velocity_spread=dy.std(axis=1)
    lag=[]
    for i in range(ident):
        v=dy[groups==i];a=v[:-1]-v[:-1].mean(axis=0);b=v[1:]-v[1:].mean(axis=0)
        denom=np.linalg.norm(a,axis=0)*np.linalg.norm(b,axis=0)
        lag.extend(np.divide(np.sum(a*b,axis=0),denom,out=np.zeros(n),where=denom>1e-14))
    topo_hash=hashlib.sha256(net['pair_edges'].tobytes()+net['triple_edges'].tobytes()+net['omega'].tobytes()).hexdigest()
    diagnostics=dict(generation_seconds=time.perf_counter()-started,nfev=calls,
        phase_order_mean=float(order.mean()),phase_order_max=float(order.max()),
        frequency_spread_mean=float(velocity_spread.mean()),derivative_lag1_median=float(np.median(lag)),
        phase_excursion_max=float(max(np.max(abs(v-v[0])) for v in values)),
        topology_sha256=topo_hash,mean_pair_degree=float(net['degree2'].mean()),mean_triad_degree=float(net['degree3'].mean()),
        trajectory_count=ident,counts=counts,topology_seed=topology_seed,trajectory_seed=trajectory_seed,
        response_noise_definition='ratio * clean training derivative std(ddof=0); independent split/node streams; no state/process noise',
        integration='RK45 sparse directed edge scatter',index_base=0,split_contract='whole trajectories, no overlap')
    return dict(phase=phase,sin_phase=np.sin(phase),cos_phase=np.cos(phase),derivative=dy,
        groups=groups,splits=splits,time=np.tile(timegrid,ident),net=net,diagnostics=diagnostics)


def save_data(path,data):
    path.mkdir(parents=True,exist_ok=True)
    from .common import save
    arrays={k:v for k,v in data.items() if isinstance(v,np.ndarray)}
    arrays.update({'net_'+k:v for k,v in data['net'].items() if isinstance(v,np.ndarray)})
    np.savez_compressed(path/'data.npz',**arrays)
    save(path/'metadata.json',dict(**data['diagnostics'],n=data['net']['n'],dtype='float64',shape=list(data['phase'].shape)))


def load_data(path):
    meta=json.loads((path/'metadata.json').read_text())
    with np.load(path/'data.npz',allow_pickle=False) as d:
        data={k:d[k] for k in d.files if not k.startswith('net_')}
        net={k[4:]:d[k] for k in d.files if k.startswith('net_')}
    net.update(n=meta['n'],topology_seed=meta['topology_seed']);data.update(net=net,diagnostics=meta)
    validate_splits(data['groups'],data['splits']);return data


def noisy_derivative(data,ratio,master_seed,trajectory_seed):
    y=data['derivative'].copy();n=y.shape[1];train=data['splits']=='train';scale=y[train].std(axis=0)
    for s,split in enumerate(['train','validation','test']):
        ids=np.flatnonzero(data['splits']==split)
        for node in range(n):
            rng=np.random.default_rng(np.random.SeedSequence([master_seed,trajectory_seed,n,33,s,node]))
            y[ids,node]+=ratio*scale[node]*rng.normal(size=len(ids))
    return y


def coordinates(phase,train):
    center=np.angle(np.exp(1j*phase[train]).mean(axis=0))
    x=np.angle(np.exp(1j*(phase-center)))
    return x,center,dict(branch_risk_fraction=float(np.mean(abs(x)>np.pi-.2)),
        minimum_training_resultant=float(abs(np.exp(1j*phase[train]).mean(axis=0)).min()))
