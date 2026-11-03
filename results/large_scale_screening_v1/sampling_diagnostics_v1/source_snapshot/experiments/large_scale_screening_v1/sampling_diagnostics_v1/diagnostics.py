"""Numerical excitation, state coverage and analytic truncation diagnostics."""
import numpy as np
from ..backend import standardized
from ..library import evaluate
from .sampling import coordinates,taylor_rhs


def lag1(values,groups=None):
    values=np.asarray(values);parts=[]
    for ident in np.unique(groups) if groups is not None else [None]:
        v=values if ident is None else values[groups==ident]
        if len(v)<3:continue
        a=v[:-1]-v[:-1].mean(axis=0);b=v[1:]-v[1:].mean(axis=0)
        denom=np.linalg.norm(a,axis=0)*np.linalg.norm(b,axis=0)
        parts.extend((np.sum(a*b,axis=0)/np.maximum(denom,1e-300))[denom>1e-14])
    return float(np.median(parts)) if parts else None


def spectrum(matrix):
    matrix=np.asarray(matrix);s=np.linalg.svd(matrix,compute_uv=False)
    tol=max(matrix.shape)*np.finfo(float).eps*s[0] if len(s) else 0.
    p=s/s.sum() if s.sum() else np.zeros_like(s)
    rank=int(np.sum(s>tol))
    return dict(rows=matrix.shape[0],columns=matrix.shape[1],rank=rank,rank_tolerance=tol,
        condition=float(s[0]/s[-1]) if len(s) and s[-1]>0 else float('inf'),
        smax=float(s[0]) if len(s) else 0.,smin=float(s[-1]) if len(s) else 0.,
        entropy_effective_rank=float(np.exp(-np.sum(p[p>0]*np.log(p[p>0])))),
        stable_rank=float(np.sum(s*s)/(s[0]*s[0])) if len(s) and s[0]>0 else 0.,
        **{f'rank_rtol_{tol:g}':int(np.sum(s>tol*s[0])) for tol in [1e-4,1e-6,1e-8,1e-10,1e-12,1e-14]}),s


def design_diagnostics(x,terms,*,scale_mode='standard'):
    theta=evaluate(x,terms)
    if scale_mode=='standard':matrix,_,_=standardized(theta);matrix=matrix[:,1:]
    elif scale_mode=='raw':matrix=theta[:,1:]-theta[:,1:].mean(axis=0)
    else:raise ValueError('Unknown scale mode')
    info,s=spectrum(matrix)
    if scale_mode=='standard':
        corr=matrix.T@matrix/len(matrix);np.fill_diagonal(corr,0)
        info['maximum_abs_term_correlation']=float(np.max(abs(corr)))
        info['fraction_term_pairs_abs_corr_gt_0_99']=float(np.sum(abs(corr)>.99)/(matrix.shape[1]*(matrix.shape[1]-1))) if matrix.shape[1]>1 else 0.
    return info,s,matrix


def state_diagnostics(data,cfg):
    mask=data['splits']=='train';phase=data['phase'][mask];dy=data['derivative'][mask];net=data['net']
    dx,center=coordinates(data);x=dx[mask];centered=x-x.mean(axis=0);cov=centered.T@centered/len(x)
    eigen=np.linalg.eigvalsh(cov);order=abs(np.exp(1j*phase).mean(axis=1))
    # Real-edge temporal PLV within each trajectory, plus instantaneous phase/frequency agreement.
    plv=[];phase_agreement=[];frequency_gap=[]
    for edge in net['pair_edges']:
        j,k=edge;diff=phase[:,j]-phase[:,k];phase_agreement.extend(np.cos(diff))
        frequency_gap.extend(abs(dy[:,j]-dy[:,k]))
        if data['diagnostics']['sampling_mode']=='trajectory':
            for group in np.unique(data['groups'][mask]):plv.append(float(abs(np.exp(1j*diff[data['groups'][mask]==group]).mean())))
    poly=taylor_rhs(phase,net,cfg);err=np.mean((poly-dy)**2,axis=0);variance=dy.var(axis=0)
    b=np.column_stack([np.ones(len(x)),x]);coef=np.linalg.lstsq(b,dy,rcond=None)[0]
    r2=1-np.sum((dy-b@coef)**2,axis=0)/np.maximum(np.sum((dy-dy.mean(axis=0))**2,axis=0),1e-300)
    return dict(train_samples=len(x),independent_trajectories=data['diagnostics']['train_trajectories'],
        independent_preparations=len(x) if data['diagnostics']['sampling_mode']=='local_iid' else data['diagnostics']['train_trajectories'],
        state_std_mean=float(x.std(axis=0).mean()),state_std_min=float(x.std(axis=0).min()),
        coordinate_range_mean=float(np.ptp(x,axis=0).mean()),max_abs_fixed_center_offset=float(abs(x).max()),
        raw_max_abs_fixed_center_offset=float(abs(phase-center).max()),branch_fraction=float(np.mean(abs(phase-center)>np.pi)),
        covariance_min_eigenvalue=float(eigen[0]),covariance_max_eigenvalue=float(eigen[-1]),
        covariance_condition=float(eigen[-1]/eigen[0]) if eigen[0]>0 else float('inf'),
        state_stable_rank=float(np.sum(eigen)/eigen[-1]),global_order_mean=float(order.mean()),global_order_max=float(order.max()),
        edge_temporal_PLV_mean=float(np.mean(plv)) if plv else None,
        edge_phase_cos_mean=float(np.mean(phase_agreement)),edge_frequency_gap_mean=float(np.mean(frequency_gap)),
        derivative_lag1_within_trajectory=lag1(dy,data['groups'][mask]) if data['diagnostics']['sampling_mode']=='trajectory' else None,
        derivative_lag1_row_order=lag1(dy),low_order_response_R2_mean=float(r2.mean()),
        low_order_response_R2_min=float(r2.min()),taylor3_MSE=float(err.mean()),
        taylor3_relative_RMSE_mean=float(np.sqrt(err/np.maximum(variance,1e-300)).mean()),
        within_trajectory_excursion_max=float(max(np.max(abs(phase[data['groups'][mask]==g]-phase[data['groups'][mask]==g][0])) for g in np.unique(data['groups'][mask]))) if plv else None,
        state_quantiles=np.quantile(x,[0,.01,.25,.5,.75,.99,1],axis=0).tolist())
