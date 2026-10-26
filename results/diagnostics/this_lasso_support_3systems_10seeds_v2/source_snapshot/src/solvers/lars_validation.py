"""Full weighted-LASSO validation path with certified LARS acceleration.

Same objective, alpha grid and tolerance as the full working-set implementation.
An early numerical LARS endpoint can cover smaller alphas only when each such
candidate independently satisfies the configured full-library KKT certificate.
Otherwise fall back to the original working-set solver without changing settings.
"""
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import lars_path
from .large_library import validation_lasso_working_set


def _certified_path(X,y,Xv,yv,a,config):
    n,p=X.shape;a=np.asarray(a,float)
    if a.shape!=(p,) or np.any(a<=0):raise ValueError('Invalid weight transform')
    ym=float(y.mean());yc=y-ym;correlation=X.T@yc/n
    scale=max(1.,float(np.max(np.abs(correlation))))
    base=max(float(np.max(np.abs(correlation)*a)),np.finfo(float).eps)
    grid=sorted(set(base*np.asarray(config['alpha_fractions'])),reverse=True)
    # LARS has an absolute small-alpha stopping scale. Uniformly rescaling
    # columns and alpha leaves beta and the original objective unchanged.
    # Keep the smallest requested knot above that absolute numerical floor.
    path_scale=1e4/base;effective_a=a*path_scale
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        alphas,active,theta=lars_path(np.asfortranarray(X*effective_a),yc,method='lasso',alpha_min=float(min(grid)*path_scale),
            max_iter=5000,copy_X=False,return_path=True)
    alphas=alphas/path_scale
    if any(issubclass(w.category,ConvergenceWarning) for w in caught):
        raise RuntimeError('LARS emitted a convergence warning; use original working-set solver')
    if not np.isfinite(theta).all() or not np.isfinite(alphas).all():raise RuntimeError('Nonfinite LARS path')
    rows=[];solutions=[]
    for alpha in grid:
        j=int(np.searchsorted(-alphas,-alpha,side='right'))
        if j==0:beta=effective_a*theta[:,0]
        elif j==len(alphas):beta=effective_a*theta[:,-1]
        else:
            fraction=(alphas[j-1]-alpha)/(alphas[j-1]-alphas[j])
            beta=effective_a*((1-fraction)*theta[:,j-1]+fraction*theta[:,j])
        gradient=X.T@(ym+X@beta-y)/n;threshold=alpha/a;nz=beta!=0
        violation=np.maximum(np.abs(gradient)-threshold,0.)
        violation[nz]=np.abs(gradient[nz]+threshold[nz]*np.sign(beta[nz]))
        kkt=float(np.max(violation,initial=0.))/scale
        if kkt>config['kkt_tolerance']:raise RuntimeError(f'LARS full-library certificate failed: {kkt:.6g}')
        mse=float(np.mean((ym+Xv@beta-yv)**2))
        if not np.isfinite(mse):raise RuntimeError('Nonfinite validation score')
        rows.append(dict(alpha=float(alpha),MSE=mse,nonzero=int(np.count_nonzero(beta)),
            full_library_kkt=kkt,full_library_scans=1,convergence_warning=False,
            warnings=[str(w.message) for w in caught],
            endpoint_reused_with_independent_KKT=bool(alpha<alphas[-1])))
        solutions.append(beta)
    floor=config['validation_tolerance']*max(float(np.mean((yv-ym)**2)),np.finfo(float).tiny)
    best=min(row['MSE'] for row in rows)
    selected=next(i for i,row in enumerate(rows) if row['MSE']<=best+floor)
    coefficient=np.concatenate(([ym],solutions[selected]))
    diag=dict(version='full-library-lars-certificate-1.0',selection='validation',alpha=rows[selected]['alpha'],
        alpha_max_train=base,selected_path_index=selected,validation_path=rows,
        kkt_scaled_violation=rows[selected]['full_library_kkt'],intercept=0.,full_library_features=p+1,
        truth_or_topology_screening=False,numerical_backend='Full weighted LARS with independent all-grid KKT',
        lars_knots=len(alphas),lars_max_knots=5000,lars_endpoint_alpha=float(alphas[-1]))
    diag['uniform_path_scale']=path_scale
    return coefficient,diag


def validation_lasso_accelerated(X,y,Xv,yv,a,config):
    try:return _certified_path(X,y,Xv,yv,a,config)
    except Exception as error:
        coefficient,diag=validation_lasso_working_set(X,y,Xv,yv,a,config)
        diag=dict(diag,accelerator_fallback_reason=str(error),numerical_backend='Original working-set fallback')
        return coefficient,diag
