"""Full-library dual Ridge/STLSQ and global-KKT working-set LASSO.

No topology/truth screening. Working sets are temporary numerical subproblems;
every accepted path point must satisfy stationarity across the entire library.
These are explicitly versioned large-library numerical variants, not M0/M1 IDs.
"""
import warnings
import numpy as np
from scipy.linalg import solve
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso


class FullLibraryConvergenceError(RuntimeError):
    def __init__(self,message,diagnostics,coefficient):
        super().__init__(message)
        self.diagnostics=diagnostics
        self.coefficient=coefficient


def ridge_coefficients(X, Y, rho=1.):
    """Coefficients [P,N] for ||Y-XB||²+rho||B||², penalizing every supplied column."""
    X=np.asarray(X);Y=np.asarray(Y)
    n,p=X.shape
    if rho<=0:raise ValueError('Positive Ridge penalty required')
    if p>n:
        gram=X@X.T
        gram.flat[::n+1]+=rho
        return X.T@solve(gram,Y,assume_a='pos',check_finite=False)
    gram=X.T@X
    gram.flat[::p+1]+=rho
    return solve(gram,X.T@Y,assume_a='pos',check_finite=False)


def stlsq_dual(X,Y,threshold=.1,rho=1.,niter=10):
    """Same algebra, global stopping and threshold convention as pinned mySINDy."""
    C=ridge_coefficients(X,Y,rho).T
    previous=1e6;iteration=1
    while iteration<niter and np.count_nonzero(np.abs(C)>1e-6)!=previous:
        iteration+=1;previous=np.count_nonzero(np.abs(C)>1e-6)
        small=np.abs(C)<threshold
        C[small]=0.
        for node in range(C.shape[0]):
            active=np.flatnonzero(~small[node])
            if len(active):C[node,active]=ridge_coefficients(X[:,active],Y[:,[node]],rho).ravel()
    return C,dict(iterations=iteration,nonzero=int(np.count_nonzero(C)))


def initial_ridge(X,Y,rho=1.):
    """Training-only free intercept; X is the common centered standardized design without bias."""
    means=Y.mean(axis=0)
    return ridge_coefficients(X,Y-means,rho).T


def validation_lasso_working_set(X,y,Xv,yv,a,config):
    """Solve the full weighted-L1 path; beta=a*theta, objective penalty=sum(abs(theta))."""
    n,p=X.shape;a=np.asarray(a,float)
    if a.shape!=(p,) or np.any(a<=0):raise ValueError('Invalid weight transform')
    ym=float(np.mean(y));yc=y-ym
    correlation=X.T@yc/n
    gradient_scale=max(1.,float(np.max(np.abs(correlation))))
    base=max(float(np.max(np.abs(correlation)*a)),np.finfo(float).eps)
    grid=sorted(set(base*np.asarray(config['alpha_fractions'])),reverse=True)
    beta=np.zeros(p);gradient=-correlation
    records=[];solutions=[]
    for alpha in grid:
        active=np.flatnonzero(beta!=0)
        candidate_warnings=[];convergence_warning=False;scans=0
        for step in range(config.get('working_set_rounds',40)+1):
            penalty=alpha/a
            violation=np.maximum(np.abs(gradient)-penalty,0.)
            nz=beta!=0
            violation[nz]=np.abs(gradient[nz]+penalty[nz]*np.sign(beta[nz]))
            kkt=float(np.max(violation,initial=0.))/gradient_scale
            if kkt<=config['kkt_tolerance']:
                break
            if step==config.get('working_set_rounds',40):
                raise RuntimeError('Full-library working-set round budget exhausted')
            inactive_violation=violation.copy();inactive_violation[active]=0.
            candidates=np.flatnonzero(inactive_violation>config['kkt_tolerance']*gradient_scale)
            if len(candidates):
                limit=config.get('working_set_add',128)
                if len(candidates)>limit:
                    candidates=candidates[np.argsort(inactive_violation[candidates],kind='stable')[-limit:]]
                active=np.union1d(active,candidates)
            elif len(active)==0:
                raise RuntimeError('Nonstationary path with no active/violating coordinates')
            elif scans==0:
                pass # Re-optimize the warm support when alpha changes.
            else:
                # No omitted violating feature: numerical active subproblem failed its full KKT check.
                raise RuntimeError(f'Active-coordinate stationarity failure {kkt:.6g}')
            U=np.asfortranarray(X[:,active]*a[active])
            solver=Lasso(alpha=float(alpha),fit_intercept=False,max_iter=config['max_iter'],tol=config['tol'],
                         selection='cyclic',precompute=len(active)<=n//2,warm_start=True)
            solver.coef_=beta[active]/a[active]
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always');solver.fit(U,yc)
            candidate_warnings.extend(f'{w.category.__name__}: {w.message}' for w in caught)
            convergence_warning=any(issubclass(w.category,ConvergenceWarning) for w in caught)
            beta.fill(0.);beta[active]=a[active]*solver.coef_
            residual=X[:,active]@beta[active]-yc
            gradient=X.T@residual/n;scans+=1
            active=active[beta[active]!=0]
        else:
            raise RuntimeError('Full-library working-set round budget exhausted')
        chosen=np.flatnonzero(beta!=0)
        prediction=ym+(Xv[:,chosen]@beta[chosen] if len(chosen) else np.zeros(len(yv)))
        mse=float(np.mean((prediction-yv)**2))
        if not np.isfinite(mse):raise RuntimeError('Nonfinite validation score')
        records.append(dict(alpha=float(alpha),MSE=mse,nonzero=len(chosen),full_library_kkt=kkt,
                            full_library_scans=scans,convergence_warning=convergence_warning,warnings=candidate_warnings))
        solutions.append((chosen.copy(),beta[chosen].copy()))
    floor=config['validation_tolerance']*max(float(np.mean((yv-ym)**2)),np.finfo(float).tiny)
    best=min(r['MSE'] for r in records)
    selected=next(i for i,r in enumerate(records) if r['MSE']<=best+floor)
    record=records[selected]
    indices,values=solutions[selected]
    coef=np.zeros(p+1);coef[0]=ym;coef[indices+1]=values
    diagnostics=dict(version='full-library-ws-1.1',selection='validation',alpha=record['alpha'],
                     alpha_max_train=base,selected_path_index=selected,validation_path=records,
                     kkt_scaled_violation=record['full_library_kkt'],intercept=0.,
                     full_library_features=p+1,truth_or_topology_screening=False,
                     precompute_policy='Gram only when active features <= training samples/2')
    if record['convergence_warning']:
        raise FullLibraryConvergenceError('Selected path point raised ConvergenceWarning',diagnostics,coef)
    return coef,diagnostics
