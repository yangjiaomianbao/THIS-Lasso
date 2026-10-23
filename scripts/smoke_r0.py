"""R0 basic regression and unmodified Su import; no oscillator simulation."""
import importlib.metadata as metadata
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ.setdefault('MPLBACKEND', 'Agg')
sys.dont_write_bytecode = True
import numpy as np
from sklearn.linear_model import LassoLarsIC

assert sys.version_info[:2] == (3, 11), sys.version
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'upstream' / 'Su-AdaptiveLASSO'))
from GeneralInteraction import GeneralInteraction

for name in ('numpy', 'scipy', 'scikit-learn', 'statsmodels', 'skglm', 'matplotlib', 'pandas', 'tabulate'):
    print(f'{name}={metadata.version(name)}')
print(f'Python={sys.version}; executable={sys.executable}')
assert callable(GeneralInteraction.solve_ada_lasso_one_node)
print('Unmodified Su GeneralInteraction import passed (including skglm).')

# More samples than features, full-rank design, nonzero residual variance for BIC.
rng = np.random.default_rng(20261016)
X = rng.normal(size=(80, 5))
y = 0.7 + X @ np.array([1.5, 0.0, -0.8, 0.0, 0.3]) + rng.normal(0, 0.05, 80)
assert np.linalg.matrix_rank(X) == X.shape[1]
fit = LassoLarsIC(criterion='bic').fit(X, y)
mse = float(np.mean((fit.predict(X) - y) ** 2))
assert fit.coef_.shape == (5,)
assert np.isfinite(fit.coef_).all() and np.isfinite(fit.alpha_)
assert np.isfinite(fit.intercept_) and mse < 0.05
print(f'LassoLarsIC: X={X.shape}; alpha={fit.alpha_:.17g}; MSE={mse:.17g}')
print(f'coef={fit.coef_}; intercept={fit.intercept_:.17g}')

# Call the author's original single-node routine on a tiny regression fixture.
# Only the three metadata attributes used by this routine are supplied; no
# oscillator constructor or simulation is run. This is not our R2 solver.
X_su = X[:, :3]
y_su = X_su @ np.array([1.5, -0.8, 0.3]) + rng.normal(0, 0.05, 80)
initial = np.linalg.lstsq(X_su, y_su, rcond=None)[0]
prepared = {'self_diff': y_su, 'two_lib': X_su[:, :2].T, 'three_lib': X_su[:, 2:].T}
result = GeneralInteraction.solve_ada_lasso_one_node(
    SimpleNamespace(n_nodes=3, type2=3, type3=3), prepared,
    {'2': initial[:2], '3': initial[2:]},
)
su_coef = np.concatenate((result['2'], result['3']))
assert su_coef.shape == (3,) and np.isfinite(su_coef).all()
assert np.isfinite(result['alpha'])
print(f'Original Su single-node call passed: coef={su_coef}; alpha={result["alpha"]:.17g}')
print('R0_PYTHON_SMOKE_PASS')
