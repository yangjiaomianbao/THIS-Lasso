"""R2 only: read fixed R1 regression matrices; never decode a Julia structure."""
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
import sys
import tomllib

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.solvers import AdaptiveLassoConfig, BICNotApplicable, fit_adaptive_lasso


def main():
    directory = ROOT/'data/r1/revision'
    metadata = tomllib.loads((directory/'metadata.toml').read_text())
    before = {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'data/r1').rglob('*') if p.is_file()}
    for name, expected in metadata['file_sha256'].items():
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest() == expected
    manifest = list(csv.DictReader((directory/'sample_manifest.csv').open()))
    train = np.array([row['split']=='train' for row in manifest])
    val = np.array([row['split']=='validation' for row in manifest])
    groups = np.array([int(row['trajectory_id']) for row in manifest])
    theta = np.loadtxt(directory/'theta.csv', delimiter=',', ndmin=2)
    response = np.loadtxt(directory/'derivatives.csv', delimiter=',', ndmin=2)[6]
    X, y = theta[:, train].T, response[train]
    Xv, yv = theta[:, val].T, response[val]
    try:
        fit_adaptive_lasso(X, y, AdaptiveLassoConfig())
    except BICNotApplicable as exc:
        print('EXPECTED_BIC_REJECTION:', str(exc))
    else:
        raise AssertionError('Clean exact polynomial should fail the BIC noise-variance guard')
    config = AdaptiveLassoConfig(selection='validation')
    coef, diagnostics = fit_adaptive_lasso(X, y, config, X_validation=Xv, y_validation=yv,
        train_groups=groups[train], validation_groups=groups[val])
    summary = dict(algorithm_version=diagnostics['algorithm_version'], sklearn_version=diagnostics['sklearn_version'],
        config=asdict(config), node_julia_index=7, train_shape=list(X.shape), validation_shape=list(Xv.shape),
        train_trajectories=diagnostics['train_trajectories'], validation_trajectories=diagnostics['validation_trajectories'],
        alpha=diagnostics['alpha'], intercept=diagnostics['intercept'], constant_index=diagnostics['constant_index'],
        nonzero_coefficients=[dict(python_feature_index=int(i), julia_feature_index=int(i+1), coefficient=float(coef[i]))
                              for i in np.flatnonzero(np.abs(coef)>1e-9)],
        validation_MSE=float(np.mean((Xv@coef+diagnostics['intercept']-yv)**2)),
        objective=diagnostics['objective'], transformed_objective=diagnostics['transformed_objective'],
        kkt_scaled_violation=diagnostics['kkt_scaled_violation'], prediction_restore_max_abs=diagnostics['prediction_restore_max_abs'],
        weight_min=float(np.min(diagnostics['weights'][diagnostics['active_indices']])),
        weight_max=float(np.max(diagnostics['weights'][diagnostics['active_indices']])),
        warnings=diagnostics['warnings'], validation_path=diagnostics['validation_path'],
        data_sha256={str(p.relative_to(ROOT)):h for p,h in before.items()})
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==h for p,h in before.items())
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    print('R1_DATA_UNCHANGED; NO_TEST_EVALUATION; NO_JULIA_INVOCATION_OR_STRUCTURE_DECODING')
    print('R2_SOLVER_DEMO_PASS')


if __name__ == '__main__':
    main()
