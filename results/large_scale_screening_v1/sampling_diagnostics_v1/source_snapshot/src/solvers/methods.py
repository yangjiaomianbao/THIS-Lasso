"""Stable Python method identities; preserve ordinary/adaptive selection paths."""
from .plain_lasso import fit_plain_lasso, fit_plain_bic
from .adaptive_lasso import fit_adaptive_lasso


def fit_node(method, X, y, config, validation=None):
    common = dict(config['lasso'])
    adaptive = dict(common, **config['adaptive'])
    if method == 'M1':
        return fit_plain_bic(X, y, dict(adaptive, selection='bic'))
    if method == 'M2':
        return fit_adaptive_lasso(X, y, dict(adaptive, selection='bic'))
    if method == 'M3':
        return fit_adaptive_lasso(X, y, dict(adaptive, selection='validation'), **validation)
    if method == 'M1v':
        return fit_plain_lasso(X, y, dict(common, selection='validation'), validation)
    raise ValueError('Unknown Python method')

