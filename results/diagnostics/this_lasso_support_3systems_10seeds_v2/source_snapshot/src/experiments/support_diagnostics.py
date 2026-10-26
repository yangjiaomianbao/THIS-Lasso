"""Support diagnostics around unchanged production solvers; no truth enters fitting."""
import sys
import numpy as np
from sklearn.linear_model import LassoLarsIC, Lasso
from src.solvers import lars_validation, large_library
from src.solvers.methods import fit_node
from src.preprocessing.taylor import restore_coefficients
from src.metrics.structure import edge_scores, binary_metrics

METHODS = ['M0D', 'M0sD', 'M1B', 'M2B', 'M1vC', 'M3C']
SYSTEMS = ['kuramoto', 'lotka_volterra', 'sis']
NAMES = dict(zip(METHODS, ['STLSQ raw', 'STLSQ std', 'LASSO BIC', 'Adaptive BIC', 'LASSO val', 'Adaptive val']))
REL_TOL = 1e-6
THRESHOLDS = [0., 1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, .1]


def capture_fit(function, *args, **kwargs):
    """Observe return-frame arrays, preserving exact original solver operations.

    Process-local profiling is active only inside the call. Production modules
    are never patched. A missing/incompatible capture is an explicit error.
    """
    captured = {}
    previous = sys.getprofile()
    def observe(frame, event, result):
        if event != 'return' or result is None:
            return
        local = frame.f_locals
        if frame.f_code is lars_validation._certified_path.__code__ and isinstance(result, tuple):
            captured['validation'] = np.asarray(local['solutions']).copy()
        elif frame.f_code is large_library.validation_lasso_working_set.__code__ and isinstance(result, tuple):
            values = np.zeros((len(local['solutions']), local['p']))
            for i, (ids, beta) in enumerate(local['solutions']):
                values[i, ids] = beta
            captured['validation'] = values
        elif isinstance(local.get('self'), LassoLarsIC) and 'coef_path_' in local:
            captured['bic'] = np.array(local['coef_path_'], copy=True)
    sys.setprofile(observe)
    try:
        coefficient, diagnostic = function(*args, **kwargs)
    finally:
        sys.setprofile(previous)
    return coefficient, diagnostic, captured


def validation_path(Z, y, Zv, yv, weights, config):
    coef, diag, capture = capture_fit(lars_validation.validation_lasso_accelerated,
                                    Z[:, 1:], y, Zv[:, 1:], yv, weights, config)
    slopes = capture['validation']
    candidates = np.column_stack((np.full(len(slopes), y.mean()), slopes))
    assert np.array_equal(candidates[diag['selected_path_index']], coef)
    errors = (Zv @ candidates.T - yv[:, None]) ** 2
    means = errors.mean(axis=0)
    minimum = int(np.argmin(means))
    se = float(errors[:, minimum].std(ddof=1) / np.sqrt(len(yv)))
    conservative = int(np.flatnonzero(means <= means[minimum] + se)[0])
    diag.update(holdout_1se_index=conservative, holdout_best_MSE_SE=se,
                conservative_rule='largest alpha within SE of minimum fixed-holdout MSE; not cross-validation')
    return coef, diag, candidates


def bic_path(method, Z, y, config):
    clean = dict(config, lasso={k: v for k, v in config['lasso'].items() if not k.startswith('working_set_')})
    coef, diag, capture = capture_fit(fit_node, 'M1' if method == 'M1B' else 'M2', Z, y, clean)
    transformed = capture['bic'].T
    active = np.asarray(diag.get('active_indices', np.arange(1, Z.shape[1])), int)
    scales = np.asarray(diag.get('adaptive_scales', np.ones(Z.shape[1])))[active]
    candidates = np.zeros((len(transformed), Z.shape[1]))
    candidates[:, active] = transformed * scales
    candidates[:, 0] = y.mean() - candidates[:, active] @ Z[:, active].mean(axis=0)
    selected = int(np.argmin(diag['bic_path']))
    assert np.allclose(candidates[selected], coef, atol=1e-12, rtol=1e-10)
    diag['selected_path_index'] = selected
    return coef, diag, candidates


def align_bic(candidates, alphas, fractions):
    """Interpolation for diagnosis, never a replacement for native BIC selection."""
    alphas = np.asarray(alphas, float)
    base = max(float(alphas[0]), np.finfo(float).eps)
    result = []
    for alpha in base * np.asarray(fractions):
        j = int(np.searchsorted(-alphas, -alpha, side='right'))
        if j == 0:
            coef = candidates[0]
        elif j == len(alphas):
            coef = candidates[-1]
        else:
            t = (alphas[j-1] - alpha) / (alphas[j-1] - alphas[j])
            coef = candidates[j-1] * (1-t) + candidates[j] * t
        result.append(coef)
    return np.asarray(result), base


def support_rows(C, Cz, truth, d, scales, response_std):
    """Counts exclude the leading bias unless an explicit field says otherwise."""
    rows = []
    for target in range(len(C)):
        c, z, t = C[target], Cz[target], truth[target]
        mask = np.abs(c) > 1e-8
        exact = c != 0
        relative = np.abs(c) > REL_TOL * np.max(np.abs(c[1:]), initial=0.)
        normalized = np.abs(c) * scales / max(response_std[target], np.finfo(float).tiny) > REL_TOL
        for a in (mask, exact, relative, normalized):
            a[0] = False
        true = np.abs(t) > 1e-8; true[0] = False
        categories = dict(constant=0, self_only=0, pair_related=0, triple_related=0, other_higher=0)
        native = np.zeros(len(c), bool)
        for j, monomial in enumerate(d):
            variables = [int(v) for v in monomial if v]
            others = set(variables) - {target+1}
            kind = ('constant' if not variables else 'self_only' if not others else
                    'pair_related' if len(others) == 1 else 'triple_related' if len(others) == 2 else 'other_higher')
            if abs(c[j]) > 1e-8:
                categories[kind] += 1
            native[j] = (len(variables) in (1, 2) and len(set(variables)) == len(variables) and target+1 not in variables)
        rows.append(dict(node=target+1, exact_nonzero=int(exact.sum()), absolute_nonzero=int(mask.sum()),
                         standardized_nonzero=int(np.count_nonzero(np.abs(z[1:]) > 1e-8)),
                         relative_nonzero=int(relative.sum()), normalized_nonzero=int(normalized.sum()),
                         including_bias_nonzero=int(np.count_nonzero(np.abs(c) > 1e-8)),
                         support_rate=float(mask.sum() / (len(c)-1)), truth_nonzero=int(true.sum()),
                         coef_TP=int(np.sum(mask & true)), coef_FP=int(np.sum(mask & ~true)),
                         coef_FN=int(np.sum(~mask & true)),
                         native_readout_support=int(np.sum(mask & native)),
                         outside_native_readout_support=int(np.sum(mask & ~native)), **categories))
    mask = np.abs(C) > 1e-8; mask[:, 0] = False
    result = {k: sum(row[k] for row in rows) for k in rows[0] if k not in ('node', 'support_rate')}
    result.update(distinct_nonzero_basis=int(np.any(mask, axis=0).sum()),
                  support_rate=float(mask.sum() / (C.shape[0] * (C.shape[1]-1))),
                  candidates_excluding_bias=C.shape[0] * (C.shape[1]-1))
    return result, rows


def evaluate_structure(C, d, config, system, threshold=1e-8):
    result = []
    for order, name in [(2, 'pairwise'), (3, 'triple')]:
        keys, scores = edge_scores(C, d, order)
        truth = {(int(r[0]), *sorted(int(v) for v in r[1:-1]))
                 for r in config[system]['pair_edges' if order == 2 else 'triple_edges']}
        metrics = binary_metrics(np.array([k in truth for k in keys]), scores, threshold)
        result.append(dict(order=name, predicted_edges=metrics['TP']+metrics['FP'], **metrics))
    return result


def verify_score_exports(C,d,native,raw):
    """Native THIS uses a strict 1e-8 cutoff; raw ranking scores use none."""
    checked=0;subthreshold=0;valid=set()
    for order in [2,3]:
        keys,scores=edge_scores(C,d,order)
        for key,value in zip(keys,scores):
            k=(order,key[0],key[1],key[2] if order==3 else 0);v=float(value);valid.add(k)
            assert raw.get(k,0.)==v,('raw_score_mismatch',k,v,raw.get(k))
            expected=v if v>1e-8 else 0.
            assert native.get(k,0.)==expected,('native_threshold_mismatch',k,expected,native.get(k))
            subthreshold+=int(0<v<=1e-8);checked+=1
    assert set(native)<=valid and set(raw)<=valid
    return checked,subthreshold


def kkt(X, y, coefficients, alpha, a):
    beta = coefficients[1:]
    gradient = X.T @ (coefficients[0] + X @ beta - y) / len(y)
    penalty = alpha / a
    active = beta != 0
    violation = np.maximum(np.abs(gradient) - penalty, 0.)
    violation[active] = np.abs(gradient[active] + penalty[active] * np.sign(beta[active]))
    denominator = max(1., float(np.max(np.abs(X.T @ (y-y.mean()) / len(y)))))
    return float(np.max(violation, initial=0.) / denominator)
