"""V2 only: centered, standardized STLSQ with an unpenalized intercept.

The historical stlsq_dual and all LASSO implementations remain unchanged.
No test response, graph, feature indices or truth is accepted by this interface.
"""
import warnings
import numpy as np
from .large_library import ridge_coefficients


def select_validation_path(path, y_validation, response_mean, tolerance):
    """Same tie rule as existing LASSO: first descending parameter within floor."""
    floor = tolerance * max(float(np.mean((y_validation-response_mean)**2)), np.finfo(float).tiny)
    best = min(row['MSE'] for row in path)
    return next(i for i, row in enumerate(path) if row['MSE'] <= best+floor)


def stlsq_validation(X, y, Xv, yv, config, initial=None):
    X, y, Xv, yv = map(np.asarray, (X, y, Xv, yv))
    if X.ndim != 2 or Xv.shape[1] != X.shape[1] or y.shape != (len(X),) or yv.shape != (len(Xv),):
        raise ValueError('Invalid train/validation shapes')
    if not all(np.isfinite(a).all() for a in (X, y, Xv, yv)):
        raise ValueError('Nonfinite inputs')
    if np.max(np.abs(X.mean(axis=0)), initial=0.) > 1e-11:
        raise ValueError('Use common externally centered features without a constant column')
    if np.any(np.std(X, axis=0) <= 1e-14):
        raise ValueError('Constant columns are excluded from penalized features')
    rho, limit = config.get('rho', 1.), config.get('niter', 10)
    fractions = config['alpha_fractions']
    if rho <= 0 or limit < 2 or list(fractions) != sorted(fractions, reverse=True) or min(fractions) <= 0:
        raise ValueError('Invalid STLSQ parameters')
    ym = float(y.mean()); yc = y-ym
    if initial is None:
        initial = ridge_coefficients(X, yc[:, None], rho).ravel()
    initial = np.asarray(initial, float)
    if initial.shape != (X.shape[1],) or not np.isfinite(initial).all():
        raise ValueError('Invalid training Ridge initializer')
    reference = float(np.max(np.abs(initial), initial=0.))
    solutions, path = [], []
    for fraction in fractions:
        threshold = reference * fraction
        beta = initial.copy(); previous = 1000000; iteration = 1
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            # Preserve the pinned STLSQ count-based stopping and strict < pruning.
            while iteration < limit and np.count_nonzero(np.abs(beta)>1e-6) != previous:
                iteration += 1; previous = np.count_nonzero(np.abs(beta)>1e-6)
                small = np.abs(beta) < threshold; beta[small] = 0.
                active = np.flatnonzero(~small)
                if len(active):
                    beta[active] = ridge_coefficients(X[:, active], yc[:, None], rho).ravel()
        if caught or not np.isfinite(beta).all():
            raise RuntimeError('STLSQ numerical warning/nonfinite path: '+str([str(w.message) for w in caught]))
        mse = float(np.mean((ym + Xv@beta-yv)**2))
        if not np.isfinite(mse):
            raise RuntimeError('STLSQ nonfinite validation MSE')
        path.append(dict(parameter=threshold, relative_parameter=float(fraction), MSE=mse,
                         nonzero=int(np.count_nonzero(beta)), iterations=iteration, status='OK'))
        solutions.append(beta)
    selected = select_validation_path(path, yv, ym, config['validation_tolerance'])
    coef = np.r_[ym, solutions[selected]]
    return coef, dict(version='stlsq-validation-free-intercept-v2', selection='validation',
        parameter=path[selected]['parameter'], parameter_reference=reference,
        selected_path_index=selected, validation_path=path, free_intercept=ym,
        intercept_penalized=False, internally_standardized=False,
        parameter_boundary=selected in (0,len(path)-1),
        tie_rule='first descending relative penalty within tolerance * validation response energy')
