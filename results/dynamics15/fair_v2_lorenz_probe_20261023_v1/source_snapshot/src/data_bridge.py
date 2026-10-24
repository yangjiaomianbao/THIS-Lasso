"""R3 file bridge. Node IDs in d remain Julia 1-based IDs, with zero padding."""
from pathlib import Path
import argparse
import tomllib

import numpy as np
import sklearn

from src.solvers import AdaptiveLassoConfig, fit_adaptive_lasso, SolverConvergenceError

from src.bridge.numeric_io import file_hash, _toml, write_toml, read_matrix
from src.solvers.plain_lasso import fit_plain_lasso

BRIDGE_VERSION = 'r3-csv-toml-1'


def load_request(directory):
    directory = Path(directory)
    m = tomllib.loads((directory/'metadata.toml').read_text())
    if m['schema_version'] != BRIDGE_VERSION or m['index_base'] != 1 or m['padding_node_id'] != 0:
        raise ValueError('Unsupported schema or node-index convention')
    n, k, p, degree = (m[key] for key in ('N', 'K', 'P', 'dmax'))
    if any(type(v) is not int or v < 1 for v in (n, k, p, degree)):
        raise ValueError('Positive integer dimensions required')
    if m['feature_entry_count'] != p*degree:
        raise ValueError('Incorrect feature entry count')
    required = {'theta.csv', 'Y.csv', 'monomial_indices.csv'}
    if m.get('has_validation', False):
        required |= {'validation_theta.csv', 'validation_Y.csv', 'train_groups.csv', 'validation_groups.csv'}
    if set(m['file_sha256']) != required:
        raise ValueError('Unexpected or missing request files')
    for name, expected in m['file_sha256'].items():
        if file_hash(directory/name) != expected:
            raise ValueError(f'Input SHA256 mismatch: {name}')
    if file_hash(directory/'monomial_indices.csv') != m['feature_order_hash']:
        raise ValueError('Feature order hash mismatch')
    theta = read_matrix(directory/'theta.csv', (p, k))
    y = read_matrix(directory/'Y.csv', (n, k))
    d = read_matrix(directory/'monomial_indices.csv', (p, degree), integer=True)
    if any(m[key] != list(shape) for key, shape in
           [('theta_shape', theta.shape), ('Y_shape', y.shape), ('d_shape', d.shape)]):
        raise ValueError('Declared matrix shapes disagree with dimensions')
    if np.any((d < 0) | (d > n)) or np.any(d[0] != 0) or not np.all(theta[0] == 1):
        raise ValueError('Invalid node IDs or missing leading constant feature')
    request = dict(metadata=m, theta=theta, Y=y, d=d, validation=None)
    if m.get('has_validation', False):
        kv = m['validation_K']
        tv = read_matrix(directory/'validation_theta.csv', (p, kv))
        yv = read_matrix(directory/'validation_Y.csv', (n, kv))
        tg = read_matrix(directory/'train_groups.csv', (k, 1), integer=True)[:, 0]
        vg = read_matrix(directory/'validation_groups.csv', (kv, 1), integer=True)[:, 0]
        if set(tg) & set(vg):
            raise ValueError('Training and validation trajectories overlap')
        request['validation'] = dict(X_validation=tv.T, Y_validation=yv, train_groups=tg, validation_groups=vg)
    return request


def process_directory(directory, *, echo=False):
    directory = Path(directory)
    if (directory/'response.toml').exists() or (directory/'coefficients.csv').exists():
        raise FileExistsError('Output exists; use a new bridge directory')
    request_hash = file_hash(directory/'metadata.toml')
    req = load_request(directory)
    m, theta, Y = req['metadata'], req['theta'], req['Y']
    outputs = {}
    if echo:
        for name, matrix, fmt in [('roundtrip_theta.csv', theta, '%.17g'), ('roundtrip_Y.csv', Y, '%.17g'),
                                  ('roundtrip_indices.csv', req['d'], '%d')]:
            np.savetxt(directory/name, matrix, delimiter=',', fmt=fmt)
            outputs[name] = file_hash(directory/name)
    else:
        if m['solver'] not in ('lasso', 'adaptive_lasso'):
            raise ValueError('Python bridge supports lasso/adaptive_lasso only')
        coefficients, nodes = [], []
        for node, y in enumerate(Y):
            validation = None
            if req['validation'] is not None:
                validation = dict(req['validation'])
                validation['y_validation'] = validation.pop('Y_validation')[node]
            if m['solver'] == 'adaptive_lasso':
                coef, diagnostics = fit_adaptive_lasso(theta.T, y, m.get('config', {}), **(validation or {}))
            else:
                coef, diagnostics = fit_plain_lasso(theta.T, y, m.get('config', {}), validation)
            if diagnostics['intercept'] != 0:
                raise ValueError('Leading constant must absorb the intercept')
            coefficients.append(coef);nodes.append(dict(node_julia_index=node+1, **diagnostics))
        C = np.asarray(coefficients)
        if C.shape != (m['N'], m['P']) or not np.isfinite(C).all():
            raise ValueError('Invalid coefficient output')
        np.savetxt(directory/'coefficients.csv', C, delimiter=',', fmt='%.17g')
        write_toml(directory/'diagnostics.toml', dict(nodes=nodes))
        outputs = {name:file_hash(directory/name) for name in ('coefficients.csv', 'diagnostics.toml')}
    # Recheck the immutable request before committing a response marker.
    load_request(directory)
    if file_hash(directory/'metadata.toml') != request_hash:
        raise ValueError('Request metadata changed during processing')
    response = dict(schema_version=BRIDGE_VERSION, mode='echo' if echo else 'solve', solver=m['solver'],
        request_metadata_sha256=request_hash, feature_order_hash=m['feature_order_hash'],
        feature_count=m['P'], feature_entry_count=m['feature_entry_count'], coefficient_shape=[m['N'], m['P']],
        sklearn_version=sklearn.__version__, numpy_version=np.__version__, output_sha256=outputs)
    write_toml(directory/'response.toml', response)
    print(f'R3_BRIDGE_PASS solver={m["solver"]} mode={response["mode"]} N={m["N"]} P={m["P"]} K={m["K"]}')
    return response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--echo', action='store_true')
    args = parser.parse_args()
    process_directory(args.directory, echo=args.echo)


if __name__ == '__main__':
    main()
