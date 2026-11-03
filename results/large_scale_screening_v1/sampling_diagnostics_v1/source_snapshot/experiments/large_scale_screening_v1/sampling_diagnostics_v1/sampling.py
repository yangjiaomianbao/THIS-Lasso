"""IID preparations and the unmodified legacy RK45 generator, same dynamics."""
import hashlib
import time
import numpy as np
from ..data import generate as trajectory_generate, network, rhs, noisy_derivative, save_data, load_data
from src.data.contracts import validate_splits


def generate(n, topology_seed, sample_seed, cfg, counts=(1024,256,256), *,
             sampling_mode='local_iid', delta=.6, train_trajectories=4, duration=.5):
    counts=list(counts)
    if len(counts)!=3 or any(int(c)!=c or c<1 for c in counts):
        raise ValueError('Three positive integer split counts required')
    if sampling_mode=='trajectory':
        if train_trajectories<1 or counts[0]%train_trajectories:
            raise ValueError('Training count must be divisible by trajectory count')
        c=dict(cfg,points_per_trajectory=counts[0]//train_trajectories,duration=duration)
        if c['points_per_trajectory']<2 or duration<=0:raise ValueError('At least two points and positive duration required')
        data=trajectory_generate(n,topology_seed,sample_seed,c,counts)
        data['diagnostics'].update(sampling_mode='trajectory',train_trajectories=train_trajectories,
            duration=duration,initial_width=cfg['initial_width'],delta=None)
        return data
    if sampling_mode!='local_iid':raise ValueError('Unknown sampling_mode')
    if not np.isfinite(delta) or delta<=0:raise ValueError('Positive finite delta required')
    started=time.perf_counter();net=network(n,topology_seed,cfg);values=[];splits=[]
    for split_id,(split,count) in enumerate(zip(('train','validation','test'),counts)):
        rng=np.random.default_rng(np.random.SeedSequence([sample_seed,n,4401,split_id]))
        values.append(net['center']+delta*rng.uniform(-.5,.5,(count,n)))
        splits.extend([split]*count)
    phase=np.concatenate(values);dy=rhs(phase,net,cfg);groups=np.arange(sum(counts));splits=np.array(splits)
    validate_splits(groups,splits)
    topo_hash=hashlib.sha256(str(n).encode()+net['pair_edges'].tobytes()+net['triple_edges'].tobytes()).hexdigest()
    return dict(phase=phase,sin_phase=np.sin(phase),cos_phase=np.cos(phase),derivative=dy,
        groups=groups,splits=splits,time=np.full(len(phase),np.nan),net=net,
        diagnostics=dict(sampling_mode='local_iid',delta=float(delta),train_trajectories=0,
            independent_train_preparations=counts[0],trajectory_count=0,counts=counts,
            topology_seed=topology_seed,trajectory_seed=sample_seed,topology_sha256=topo_hash,
            dynamics_sha256=hashlib.sha256(net['omega'].tobytes()+net['center'].tobytes()).hexdigest(),
            generation_seconds=time.perf_counter()-started,nfev=[],duration=None,
            response_noise_definition='ratio * clean training derivative std(ddof=0); independent split/node streams; no state/process noise',
            integration='none; analytic rhs at independent preparations',index_base=0,
            split_contract='independent split RNG streams; groups are sample IDs, never time windows'))


def coordinates(data, center_mode='fixed'):
    phase=data['phase'];train=data['splits']=='train'
    if center_mode=='fixed':center=data['net']['center'].copy()
    elif center_mode=='circular':center=np.angle(np.exp(1j*phase[train]).mean(axis=0))
    elif center_mode=='arithmetic':center=phase[train].mean(axis=0)
    else:raise ValueError('Unknown center mode')
    # Keep the legacy local principal-value coordinate convention, record branches.
    return np.angle(np.exp(1j*(phase-center))),center


def triadic_coefficients(net,cfg,center=None):
    center=net['center'] if center is None else np.asarray(center)
    rows=[]
    for i,j,k in net['triples']:
        amplitude=cfg['K3']/max(int(net['degree3'][i]),1)
        angle=center[j]+center[k]-2*center[i]
        rows.append(dict(target=int(i),j=int(j),k=int(k),angle=float(angle),
            amplitude=float(amplitude),quadratic_cross=float(-amplitude*np.sin(angle)),
            relative_quadratic_visibility=float(abs(np.sin(angle)))))
    return rows


def taylor_rhs(phase,net,cfg,center=None,degree=3):
    """Full analytic expansion, distinct from a screened sparse fitted library."""
    center=net['center'] if center is None else np.asarray(center)
    dx=np.angle(np.exp(1j*(np.asarray(phase)-center)))
    result=np.broadcast_to(net['omega'],dx.shape).copy()
    for order,edges in [(2,net['pairs']),(3,net['triples'])]:
        for edge in edges:
            i=int(edge[0]);sources=edge[1:];weight=-(order-1)
            angle=float(center[sources].sum()+weight*center[i])
            z=dx[:,sources].sum(axis=1)+weight*dx[:,i]
            value=np.full(len(dx),np.sin(angle))
            if degree>=1:value+=np.cos(angle)*z
            if degree>=2:value-=np.sin(angle)*z*z/2
            if degree>=3:value-=np.cos(angle)*z*z*z/6
            result[:,i]+=cfg['K'+str(order)]/max(int(net['degree'+str(order)][i]),1)*value
    return result
