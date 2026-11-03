"""Thin M2 wrapper; no substitute fitting, variance estimate or criterion."""
import time
import numpy as np
from src.solvers.methods import fit_node
from src.solvers.adaptive_lasso import BICNotApplicable, SolverConvergenceError
from src.preprocessing.taylor import restore_coefficients
from .library import evaluate


def standardized(theta):
    means=theta.mean(axis=0);scales=theta.std(axis=0)
    means[0]=0;scales[0]=1;scales[scales<1e-14]=1
    return (theta-means)/scales,means,scales


def fit(x,y,terms,config):
    if len(x)<=len(terms):
        # Same public backend guard; avoid its expensive Ridge initializer first.
        return dict(status='NA',reason=f'Original BIC guard K<=P: {len(x)}<={len(terms)}',
            features=len(terms),feature_seconds=0.,regression_seconds=0.)
    started=time.perf_counter();theta=evaluate(x,terms);construction=time.perf_counter()-started
    z,means,scales=standardized(theta)
    started=time.perf_counter()
    try:
        c=dict(config);c.pop('selection',None)
        coef,diag=fit_node('M2',z,y,dict(lasso={},adaptive=c))
    except BICNotApplicable as e:
        return dict(status='NA',reason=str(e),features=len(terms),feature_seconds=construction,
            regression_seconds=time.perf_counter()-started)
    except SolverConvergenceError as e:
        return dict(status='NUMERICAL_FAILURE',reason=str(e),features=len(terms),feature_seconds=construction,
            regression_seconds=time.perf_counter()-started)
    raw=restore_coefficients(coef[None,:],means,scales)[0]
    return dict(status='OK',reason='',features=len(terms),feature_seconds=construction,
        regression_seconds=time.perf_counter()-started,coefficient=raw,diagnostics=diag,means=means,scales=scales)
