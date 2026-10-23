"""Ordinary LASSO paths extracted unchanged from the R3 bridge and R4 wrapper."""
from types import SimpleNamespace
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LassoLarsIC

from .adaptive_lasso import AdaptiveLassoConfig, SolverConvergenceError, _prepare, _bic_variance


def fit_plain_lasso(X, y, config, validation=None):
    """Ordinary original-unit LASSO, with the leading constant left unpenalized."""
    c = AdaptiveLassoConfig(**config)
    allowed = {'selection', 'alpha', 'validation_alphas', 'alpha_fractions',
               'validation_tolerance', 'max_iter', 'tol', 'kkt_tolerance'}
    if set(config)-allowed:
        raise ValueError('Plain LASSO accepts only L1 selection/convergence options')
    if c.selection not in ('fixed', 'validation'):
        raise ValueError('Plain LASSO requires explicit fixed or validation selection')
    active = np.flatnonzero(np.ptp(X, axis=0) != 0)
    constant = np.flatnonzero(np.ptp(X, axis=0) == 0)
    if any(X[0, j] != 0 for j in constant if j != 0):
        raise ValueError('Multiple nonzero constants are unidentifiable')
    means, ym = X.mean(axis=0), float(y.mean())
    U, yc = X[:, active]-means[active], y-ym
    if not len(active):
        coef = np.zeros(X.shape[1]);coef[0] = ym
        return coef, dict(selection='constant_only', alpha=0., intercept=0., warnings=[], converged=True)
    if c.selection == 'validation':
        if validation is None:
            raise ValueError('Independent validation required')
        Xv, yv = validation['X_validation'], validation['y_validation']
        if not np.all(Xv[:, constant] == X[0, constant]):
            raise ValueError('A training-constant column changes in validation')
        base = max(float(np.max(np.abs(U.T@yc))/len(y)), np.finfo(float).eps)
        grid = c.validation_alphas if c.validation_alphas is not None else base*np.asarray(c.alpha_fractions)
    else:
        grid = [c.alpha]
    if any(a is None or not np.isfinite(a) or a <= 0 for a in grid):
        raise ValueError('Positive finite alpha required')
    models, rows = [], []
    for alpha in sorted(set(grid), reverse=True):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            if np.all(yc == 0):
                model = SimpleNamespace(coef_=np.zeros(len(active)), n_iter_=0, dual_gap_=0.)
            else:
                model = Lasso(alpha=alpha, fit_intercept=False, max_iter=c.max_iter, tol=c.tol,
                              selection='cyclic').fit(U, yc)
        coef = np.zeros(X.shape[1]);coef[active] = model.coef_
        coef[0] = ym-means[active]@model.coef_
        score = float(np.mean((Xv@coef-yv)**2)) if c.selection == 'validation' else 0.
        if not np.isfinite(score):
            raise SolverConvergenceError('Nonfinite LASSO validation score')
        rows.append(dict(alpha=float(alpha), MSE=score, nonzero=int(np.count_nonzero(model.coef_)),
            n_iter=int(model.n_iter_), dual_gap=float(model.dual_gap_),
            warnings=[f'{w.category.__name__}: {w.message}' for w in caught],
            convergence_warning=any(issubclass(w.category, ConvergenceWarning) for w in caught)))
        models.append(coef)
    if c.selection == 'validation':
        floor = c.validation_tolerance*max(float(np.mean((yv-ym)**2)), np.finfo(float).tiny)
        best = min(row['MSE'] for row in rows)
        selected = next(i for i, row in enumerate(rows) if row['MSE'] <= best+floor)
    else:
        selected = 0
    coef, chosen = models[selected], rows[selected]
    gradient = U.T@(X@coef-y)/len(y)
    nonzero = coef[active] != 0
    violation = np.maximum(np.abs(gradient)-chosen['alpha'], 0.)
    violation[nonzero] = np.abs(gradient[nonzero]+chosen['alpha']*np.sign(coef[active][nonzero]))
    kkt = float(np.max(violation))/max(1., float(np.max(np.abs(U.T@yc/len(y)))))
    if chosen['convergence_warning'] or kkt > c.kkt_tolerance:
        raise SolverConvergenceError(f'Plain LASSO convergence/KKT failure: {chosen}, {kkt}')
    return coef, dict(selection=c.selection, alpha=chosen['alpha'], intercept=0., converged=True,
        algorithm_version='r3-plain-lasso-1.0.0',
        config={name:getattr(c, name) for name in sorted(allowed)},
        validation_path=rows, selected_path_index=selected, kkt_scaled_violation=kkt,
        objective=float(np.mean((X@coef-y)**2)/2+chosen['alpha']*np.sum(np.abs(coef[active]))),
        feature_means=means, response_mean=ym,
        warnings=[w for row in rows for w in row['warnings']])



def fit_plain_bic(X, y, config):
    c = AdaptiveLassoConfig(**config)
    p = _prepare(X, y, c)
    U = p['z']  # common external scaling, standardize=False in this wrapper
    variance, info = _bic_variance(X, p, U, c)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model = LassoLarsIC(criterion='bic', fit_intercept=False,
                            noise_variance=variance, max_iter=c.max_iter).fit(U, p['yc'])
    coef = np.zeros(X.shape[1]);coef[p['active']] = model.coef_
    coef[0] = p['response_mean']-p['means'][p['active']]@model.coef_
    gradient = U.T@(X@coef-y)/len(y)
    nonzero = model.coef_ != 0
    violations = np.maximum(np.abs(gradient)-model.alpha_, 0.)
    violations[nonzero] = np.abs(gradient[nonzero]+model.alpha_*np.sign(model.coef_[nonzero]))
    kkt = float(np.max(violations))/max(1., float(np.max(np.abs(U.T@p['yc']/len(y)))))
    if any(issubclass(w.category, ConvergenceWarning) for w in caught) or kkt > c.kkt_tolerance:
        raise SolverConvergenceError(f'Ordinary LARS/BIC convergence/KKT failure: {kkt}')
    return coef, dict(selection='bic', alpha=float(model.alpha_), noise_variance=variance,
        alpha_path=model.alphas_, bic_path=model.criterion_, kkt_scaled_violation=kkt,
        warnings=[f'{w.category.__name__}: {w.message}' for w in caught], **info)

