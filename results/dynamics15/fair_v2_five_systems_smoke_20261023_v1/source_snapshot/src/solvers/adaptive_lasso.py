"""Training-only Adaptive LASSO, with an objective in the original feature units.

min (||y - intercept - X @ beta||^2 / (2*K) + alpha * sum(w*abs(beta))).
One nonzero constant column is a free bias and is never penalized. Returned
coefficients retain every original column, including constants and zero columns.
Standardization conditions the initializer; the L1 coordinate transform explicitly
preserves this original-unit objective instead of silently changing its weights.
"""
from dataclasses import asdict, dataclass
from typing import Mapping
from types import SimpleNamespace
import warnings

import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LassoLarsIC, Ridge

ALGORITHM_VERSION = 'r2-adaptive-lasso-1.0.0'


class BICNotApplicable(ValueError):
    """BIC assumptions failed; explicitly choose independent-trajectory validation."""


class SolverConvergenceError(RuntimeError):
    """The selected numerical solution failed convergence or stationarity checks."""


@dataclass(frozen=True)
class AdaptiveLassoConfig:
    initialization: str = 'ridge'
    ridge_rho: float = 1.0
    gamma: float = 1.0
    eps: float = 1e-6
    standardize: bool = True
    fit_intercept: bool = True
    selection: str = 'bic'
    alpha: float | None = None
    validation_alphas: tuple[float, ...] | None = None
    alpha_fractions: tuple[float, ...] = (1.0, .3, .1, .03, .01, .003, .001, .0003, .0001, .00001, .000001, .00000001)
    validation_tolerance: float = 1e-12
    constant_tol: float = 0.0
    bic_noise_floor: float = 1e-12
    bic_condition_limit: float = 1e8
    max_iter: int = 100000
    tol: float = 1e-10
    kkt_tolerance: float = 1e-6


def _configuration(config):
    c = AdaptiveLassoConfig() if config is None else (
        AdaptiveLassoConfig(**config) if isinstance(config, Mapping) else config)
    if not isinstance(c, AdaptiveLassoConfig):
        raise TypeError('config must be AdaptiveLassoConfig or a mapping')
    if c.initialization not in ('ridge', 'ols') or c.selection not in ('bic', 'validation', 'fixed'):
        raise ValueError('Unknown initialization or selection strategy')
    for name in ('gamma', 'eps', 'ridge_rho', 'tol', 'kkt_tolerance', 'bic_noise_floor', 'bic_condition_limit'):
        if not np.isfinite(getattr(c, name)) or getattr(c, name) <= 0:
            raise ValueError(f'{name} must be finite and positive')
    for name in ('constant_tol', 'validation_tolerance'):
        if not np.isfinite(getattr(c, name)) or getattr(c, name) < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    if not isinstance(c.max_iter, int) or c.max_iter < 1:
        raise ValueError('max_iter must be a positive integer')
    if c.selection == 'fixed' and (c.alpha is None or not np.isfinite(c.alpha) or c.alpha <= 0):
        raise ValueError('Fixed selection requires a finite positive alpha')
    fractions = np.asarray(c.alpha_fractions, dtype=float)
    if fractions.ndim != 1 or not len(fractions) or np.any(~np.isfinite(fractions)) or np.any((fractions <= 0) | (fractions > 1)):
        raise ValueError('alpha_fractions must be in (0,1]')
    if c.validation_alphas is not None:
        grid = np.asarray(c.validation_alphas, dtype=float)
        if grid.ndim != 1 or not len(grid) or np.any(~np.isfinite(grid)) or np.any(grid <= 0):
            raise ValueError('validation_alphas must be finite and positive')
    return c


def _arrays(X, y, name='training'):
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != len(y) or not X.shape[1] or len(y) < 2:
        raise ValueError(f'{name}: require X:[K,P], y:[K], K>=2, P>=1')
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError(f'{name}: nonfinite input')
    return X, y


def _prepare(X, y, c):
    constant = np.ptp(X, axis=0) <= c.constant_tol
    nonzero_constants = np.flatnonzero(constant & (np.abs(X[0]) > c.constant_tol))
    if len(nonzero_constants) > 1:
        raise ValueError('Multiple nonzero constant columns give an unidentifiable bias; retain one')
    bias_column = int(nonzero_constants[0]) if len(nonzero_constants) else None
    has_bias = c.fit_intercept or bias_column is not None
    means = X.mean(axis=0) if has_bias else np.zeros(X.shape[1])
    response_mean = float(y.mean()) if has_bias else 0.0
    active = np.flatnonzero(~constant)
    centered = X[:, active] - means[active]
    scales = np.ones(X.shape[1])
    if c.standardize:
        scales[active] = np.sqrt(np.mean(centered**2, axis=0))
    if not np.isfinite(scales).all() or np.any(scales[active] <= np.sqrt(np.finfo(float).tiny)):
        raise ValueError('Feature scaling is numerically unreliable')
    return dict(active=active, constant=constant, bias_column=bias_column,
                has_bias=has_bias, means=means, scales=scales,
                response_mean=response_mean, z=centered / scales[active], yc=y-response_mean)


def _validation(Xv, yv, X, p, train_groups, validation_groups, c):
    if Xv is None or yv is None or train_groups is None or validation_groups is None:
        raise ValueError('Validation selection requires X_validation, y_validation and disjoint trajectory group IDs')
    Xv, yv = _arrays(Xv, yv, 'validation')
    if Xv.shape[1] != X.shape[1]:
        raise ValueError('Training/validation feature counts differ')
    tg, vg = np.asarray(train_groups), np.asarray(validation_groups)
    if tg.shape != (len(X),) or vg.shape != (len(Xv),):
        raise ValueError('Trajectory group IDs must have one entry per sample')
    for groups in (tg, vg):
        if any(g is None or (isinstance(g, (float, np.floating)) and not np.isfinite(g)) for g in groups):
            raise ValueError('Missing trajectory group ID')
    if set(tg.tolist()) & set(vg.tolist()):
        raise ValueError('A trajectory cannot occur in both training and validation')
    if not np.all(np.abs(Xv[:, p['constant']] - X[0, p['constant']]) <= c.constant_tol):
        raise ValueError('A training-constant column changes in validation')
    return Xv, yv, dict(train_trajectories=len(set(tg.tolist())), validation_trajectories=len(set(vg.tolist())))


def _fit_cd(U, yc, alpha, c):
    if np.all(yc == 0):
        return SimpleNamespace(coef_=np.zeros(U.shape[1]), n_iter_=0, dual_gap_=0.0), [], False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model = Lasso(alpha=alpha, fit_intercept=False, max_iter=c.max_iter, tol=c.tol, selection='cyclic').fit(U, yc)
    return model, [f'{w.category.__name__}: {w.message}' for w in caught], any(issubclass(w.category, ConvergenceWarning) for w in caught)


def _bic_variance(X, p, U, c):
    n, original_p = X.shape
    q = len(p['active'])
    message = ' Explicitly use selection="validation" with independent trajectory groups.'
    if n <= original_p or n <= q + int(p['has_bias']):
        raise BICNotApplicable('K<=P or no residual degrees of freedom.' + message)
    singular = np.linalg.svd(p['z'], compute_uv=False)
    rank = np.linalg.matrix_rank(p['z'])
    condition = float(singular[0] / singular[-1]) if singular[-1] else float('inf')
    weighted_condition = float(np.linalg.cond(U))
    if rank < q or max(condition, weighted_condition) > c.bic_condition_limit:
        raise BICNotApplicable('Rank-deficient or numerically unstable full design.' + message)
    residual = p['yc'] - p['z'] @ np.linalg.lstsq(p['z'], p['yc'], rcond=None)[0]
    mse = float(np.mean(residual**2))
    signal = max(float(np.mean(p['yc']**2)), np.finfo(float).tiny)
    if not np.isfinite(mse) or mse <= c.bic_noise_floor * signal:
        raise BICNotApplicable('Full-model noise variance is zero or below its reliability floor.' + message)
    df = n-q-int(p['has_bias'])
    return float(np.sum(residual**2)/df), dict(full_design_rank=int(rank),
        full_design_condition=condition, weighted_design_condition=weighted_condition,
        noise_degrees_of_freedom=df, full_model_residual_MSE=mse)


def _select(X, p, U, a, c, validation):
    if c.selection == 'bic':
        variance, details = _bic_variance(X, p, U, c)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model = LassoLarsIC(criterion='bic', fit_intercept=False,
                noise_variance=variance, max_iter=c.max_iter).fit(U, p['yc'])
        return model.coef_, float(model.alpha_), dict(details, selection='bic', noise_variance=variance,
            alpha_path=model.alphas_.tolist(), bic_path=model.criterion_.tolist(),
            warnings=[f'{w.category.__name__}: {w.message}' for w in caught],
            convergence_warning=any(issubclass(w.category, ConvergenceWarning) for w in caught))
    if c.selection == 'fixed':
        model, caught, failed = _fit_cd(U, p['yc'], c.alpha, c)
        return model.coef_, float(c.alpha), dict(selection='fixed', warnings=caught,
            convergence_warning=failed, n_iter=int(model.n_iter_), dual_gap=float(model.dual_gap_))
    Xv, yv, group_info = validation
    Uv = (Xv[:, p['active']] - p['means'][p['active']]) * a
    alpha_max = float(np.max(np.abs(U.T @ p['yc'])) / len(X))
    base = max(alpha_max, np.finfo(float).eps)
    grid = sorted(set(c.validation_alphas if c.validation_alphas is not None else base*np.asarray(c.alpha_fractions)), reverse=True)
    models, rows, all_warnings = [], [], []
    for alpha in grid:
        model, caught, failed = _fit_cd(U, p['yc'], alpha, c)
        mse = float(np.mean((p['response_mean']+Uv@model.coef_-yv)**2))
        if not np.isfinite(mse):
            raise SolverConvergenceError('A validation candidate produced nonfinite predictions')
        rows.append(dict(alpha=float(alpha), MSE=mse, nonzero=int(np.count_nonzero(model.coef_)),
            n_iter=int(model.n_iter_), dual_gap=float(model.dual_gap_), warnings=caught, convergence_warning=failed))
        models.append(model)
        all_warnings.extend(f'alpha={alpha:.17g}: {w}' for w in caught)
    floor = c.validation_tolerance * max(float(np.mean((yv-p['response_mean'])**2)), np.finfo(float).tiny)
    best = min(row['MSE'] for row in rows)
    selected = next(i for i, row in enumerate(rows) if row['MSE'] <= best+floor)
    return models[selected].coef_, float(grid[selected]), dict(selection='validation',
        validation_path=rows, selected_path_index=selected, alpha_max_train=alpha_max,
        validation_tie_tolerance=floor, warnings=all_warnings,
        convergence_warning=rows[selected]['convergence_warning'], **group_info)


def fit_adaptive_lasso(X, y, config=None, *, X_validation=None, y_validation=None,
                       train_groups=None, validation_groups=None):
    """Return (coef[P], diagnostics); predict via X@coef+diagnostics['intercept'].

    Raw-unit initializer beta_init comes from training-only standardized Ridge
    (or OLS), then w=(abs(beta_init)+eps)**(-gamma). For standardized regression
    coordinates eta=s*beta, the equivalent weights are w/s, never silently w.
    Validation selection requires disjoint independent-trajectory IDs. BIC never
    falls back automatically. No test data or structural truth is accepted.
    """
    c = _configuration(config)
    X, y = _arrays(X, y)
    p = _prepare(X, y, c)
    validation = _validation(X_validation, y_validation, X, p, train_groups, validation_groups, c) if c.selection == 'validation' else None
    n, features = X.shape
    active, scales, z, yc = p['active'], p['scales'], p['z'], p['yc']
    if len(active):
        if c.initialization == 'ridge':
            initial_scaled = Ridge(alpha=c.ridge_rho, fit_intercept=False, solver='svd').fit(z, yc).coef_
        else:
            initial_scaled = np.linalg.lstsq(z, yc, rcond=None)[0]
        beta_initial = initial_scaled / scales[active]
        with np.errstate(over='raise', under='raise', invalid='raise', divide='raise'):
            try:
                a = (np.abs(beta_initial) + c.eps)**c.gamma
                w = 1.0 / a
            except FloatingPointError as exc:
                raise ValueError('Adaptive weights overflow/underflow; revise gamma/eps and units explicitly') from exc
        U = z * (scales[active] * a)  # == (X_active - train_mean) * a
        theta, alpha, selection = _select(X, p, U, a, c, validation)
        beta = a * theta
    else:
        beta_initial = a = w = theta = beta = np.empty(0)
        U = np.empty((n, 0))
        alpha = float(c.alpha) if c.selection == 'fixed' else 0.0
        selection = dict(selection='constant_only', requested_selection=c.selection,
            reason='No penalized variable; model selection not performed', warnings=[], convergence_warning=False)
    coef = np.zeros(features)
    coef[active] = beta
    bias = float(p['response_mean'] - p['means'][active] @ beta)
    intercept = bias
    if p['bias_column'] is not None:
        coef[p['bias_column']] = bias / X[0, p['bias_column']]
        intercept = 0.0  # bias is already represented by the original constant column

    prediction = X @ coef + intercept
    transformed_prediction = p['response_mean'] + U @ theta
    weights, adaptive_scales, scaled_coef, initial_coef = (np.zeros(features) for _ in range(4))
    weights[active], adaptive_scales[active], scaled_coef[active], initial_coef[active] = w, a, theta, beta_initial
    initial_bias = float(p['response_mean'] - p['means'][active] @ beta_initial)
    if p['bias_column'] is not None:
        initial_coef[p['bias_column']] = initial_bias / X[0, p['bias_column']]
    residual = z @ (scales[active] * beta) - yc
    gradient = z.T @ residual / n
    threshold = alpha * w / scales[active]
    nonzero = beta != 0
    violation = np.maximum(np.abs(gradient)-threshold, 0)
    violation[nonzero] = np.abs(gradient[nonzero] + threshold[nonzero]*np.sign(beta[nonzero]))
    max_violation = float(np.max(violation, initial=0.0))
    gradient_scale = max(1.0, float(np.max(np.abs(z.T @ yc/n), initial=0.0)))
    scaled_violation = max_violation / gradient_scale
    if selection['convergence_warning'] or scaled_violation > c.kkt_tolerance:
        raise SolverConvergenceError(f'Selected solution failed convergence/KKT: scaled violation={scaled_violation:.6g}; warnings={selection["warnings"]}')
    penalty = float(weights @ np.abs(coef))
    objective = float(np.mean((y-prediction)**2)/2 + alpha*penalty)
    transformed_objective = float(np.mean((yc-U@theta)**2)/2 + alpha*np.sum(np.abs(theta)))
    diagnostics = dict(algorithm_version=ALGORITHM_VERSION, sklearn_version=sklearn.__version__,
        numpy_version=np.__version__, config=asdict(c), shape=dict(K=n, P=features, penalized_features=len(active)),
        alpha=alpha, intercept=intercept, bias=bias, constant_index=p['bias_column'],
        constant_indices=np.flatnonzero(p['constant']), active_indices=active,
        feature_means=p['means'], feature_scales=scales, response_mean=p['response_mean'],
        initial_coef=initial_coef, initial_intercept=initial_bias if p['bias_column'] is None else 0.0,
        weights=weights, adaptive_scales=adaptive_scales, scaled_lasso_coef=scaled_coef,
        coef_standardized=coef*scales, objective=objective, transformed_objective=transformed_objective,
        prediction_restore_max_abs=float(np.max(np.abs(prediction-transformed_prediction))),
        kkt_max_abs=max_violation, kkt_scaled_violation=scaled_violation,
        bias_stationarity=float(abs(np.mean(y-prediction))) if p['has_bias'] else 0.0,
        zero_centered_response=bool(np.all(yc == 0)), converged=True, **selection)
    return coef, diagnostics
