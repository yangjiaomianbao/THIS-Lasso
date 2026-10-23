import copy
from pathlib import Path
import tempfile
import tomllib
import unittest

import numpy as np

from src.data_bridge import (BRIDGE_VERSION, file_hash, load_request, process_directory,
                             read_matrix, write_toml, fit_plain_lasso)


def request(directory, *, solver='adaptive_lasso', config=None, n=2, k=80):
    rng = np.random.default_rng(73)
    theta = np.vstack([np.ones(k), rng.normal(size=k), rng.normal(size=k)])
    C = np.array([[.3, 1.2, -.4], [-.7, .1, .8]])[:n]
    Y = C@theta
    d = np.array([[0], [1], [min(2, n)]])
    for name, a, fmt in [('theta.csv', theta, '%.17g'), ('Y.csv', Y, '%.17g'), ('monomial_indices.csv', d, '%d')]:
        np.savetxt(directory/name, a, delimiter=',', fmt=fmt)
    hashes = {name:file_hash(directory/name) for name in ('theta.csv', 'Y.csv', 'monomial_indices.csv')}
    m = dict(schema_version=BRIDGE_VERSION, N=n, K=k, P=3, dmax=1, index_base=1, padding_node_id=0,
        feature_order_hash=hashes['monomial_indices.csv'], feature_entry_count=3,
        theta_shape=[3, k], Y_shape=[n, k], d_shape=[3, 1], file_sha256=hashes,
        has_validation=False, solver=solver, config=config or dict(selection='fixed', alpha=1e-8))
    write_toml(directory/'metadata.toml', m)
    return m, theta, Y, C


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)

    def test_adaptive_coefficients_keep_node_and_feature_axes(self):
        m, theta, Y, truth = request(self.directory)
        response = process_directory(self.directory)
        C = read_matrix(self.directory/'coefficients.csv', (2, 3))
        np.testing.assert_allclose(C, truth, atol=2e-7)
        np.testing.assert_allclose(C@theta, Y, atol=6e-7)
        self.assertEqual(response['feature_order_hash'], m['feature_order_hash'])
        self.assertEqual(response['coefficient_shape'], [2, 3])
        diagnostics = tomllib.loads((self.directory/'diagnostics.toml').read_text())
        self.assertEqual([n['node_julia_index'] for n in diagnostics['nodes']], [1, 2])
        self.assertEqual(diagnostics['nodes'][0]['intercept'], 0.)

    def test_plain_lasso_is_unweighted_and_has_free_constant(self):
        _, theta, Y, truth = request(self.directory, solver='lasso')
        process_directory(self.directory)
        C = read_matrix(self.directory/'coefficients.csv', truth.shape)
        np.testing.assert_allclose(C, truth, atol=1e-7)
        c, d = fit_plain_lasso(theta.T, Y[0], dict(selection='fixed', alpha=100.))
        np.testing.assert_array_equal(c[1:], [0., 0.])
        self.assertAlmostEqual(c[0], Y[0].mean())

    def test_one_target_keeps_one_row(self):
        request(self.directory, n=1)
        process_directory(self.directory)
        self.assertEqual(read_matrix(self.directory/'coefficients.csv', (1, 3)).shape, (1, 3))

    def test_one_coefficient_column_returns_a_matrix(self):
        m, _, Y, _ = request(self.directory)
        np.savetxt(self.directory/'theta.csv', np.ones((1, 80)), delimiter=',', fmt='%.17g')
        np.savetxt(self.directory/'monomial_indices.csv', [[0]], delimiter=',', fmt='%d')
        for name in ('theta.csv', 'monomial_indices.csv'):
            m['file_sha256'][name] = file_hash(self.directory/name)
        m.update(P=1, theta_shape=[1, 80], d_shape=[1, 1], feature_entry_count=1,
                 feature_order_hash=m['file_sha256']['monomial_indices.csv'])
        write_toml(self.directory/'metadata.toml', m)
        process_directory(self.directory)
        C = read_matrix(self.directory/'coefficients.csv', (2, 1))
        np.testing.assert_allclose(C[:, 0], Y.mean(axis=1))

    def test_plain_constant_response_is_analytic_without_warnings(self):
        _, theta, _, _ = request(self.directory)
        coef, diagnostics = fit_plain_lasso(theta.T, np.full(80, 3.), dict(selection='fixed', alpha=.1))
        np.testing.assert_array_equal(coef, [3., 0., 0.])
        self.assertEqual(diagnostics['warnings'], [])

    def test_one_row_one_column_reading(self):
        for a in (np.array([[.12345678901234567]]), np.array([[1., 2., 3.]]), np.array([[1.], [2.], [3.]])):
            p = self.directory/'shape.csv'
            np.savetxt(p, a, delimiter=',', fmt='%.17g')
            np.testing.assert_array_equal(read_matrix(p, a.shape), a)

    def test_echo_roundtrip_is_exact(self):
        _, theta, Y, _ = request(self.directory)
        process_directory(self.directory, echo=True)
        np.testing.assert_array_equal(read_matrix(self.directory/'roundtrip_theta.csv', theta.shape), theta)
        np.testing.assert_array_equal(read_matrix(self.directory/'roundtrip_Y.csv', Y.shape), Y)
        self.assertEqual(file_hash(self.directory/'roundtrip_indices.csv'), file_hash(self.directory/'monomial_indices.csv'))

    def test_reordered_indices_rejected_even_with_updated_file_hash(self):
        m, *_ = request(self.directory)
        d = np.array([[0], [2], [1]])
        np.savetxt(self.directory/'monomial_indices.csv', d, delimiter=',', fmt='%d')
        m['file_sha256']['monomial_indices.csv'] = file_hash(self.directory/'monomial_indices.csv')
        write_toml(self.directory/'metadata.toml', m)
        with self.assertRaisesRegex(ValueError, 'Feature order hash'):
            load_request(self.directory)

    def test_changed_numeric_input_rejected(self):
        request(self.directory)
        with (self.directory/'theta.csv').open('a') as f:
            f.write('9\n')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            load_request(self.directory)

    def test_wrong_declared_shape_rejected(self):
        m, *_ = request(self.directory)
        m['K'] += 1
        write_toml(self.directory/'metadata.toml', m)
        with self.assertRaisesRegex(ValueError, 'expected finite matrix'):
            load_request(self.directory)

    def test_wrong_index_base_and_feature_count_rejected(self):
        m, *_ = request(self.directory)
        for key, value in [('index_base', 0), ('feature_entry_count', 999)]:
            changed = copy.deepcopy(m);changed[key] = value
            write_toml(self.directory/'metadata.toml', changed)
            with self.assertRaises(ValueError):
                load_request(self.directory)

    def test_fractional_node_id_rejected(self):
        m, *_ = request(self.directory)
        np.savetxt(self.directory/'monomial_indices.csv', [[0], [1.5], [2]], delimiter=',', fmt='%.17g')
        h = file_hash(self.directory/'monomial_indices.csv')
        m['file_sha256']['monomial_indices.csv'] = m['feature_order_hash'] = h
        write_toml(self.directory/'metadata.toml', m)
        with self.assertRaisesRegex(ValueError, 'integer IDs'):
            load_request(self.directory)

    def test_stale_output_cannot_be_overwritten(self):
        request(self.directory)
        process_directory(self.directory)
        with self.assertRaises(FileExistsError):
            process_directory(self.directory)

    def test_validation_requires_nonoverlapping_trajectory_ids(self):
        m, theta, Y, _ = request(self.directory)
        for name, a in [('validation_theta.csv', theta), ('validation_Y.csv', Y),
                        ('train_groups.csv', np.arange(80).reshape(-1, 1)),
                        ('validation_groups.csv', np.arange(80).reshape(-1, 1))]:
            np.savetxt(self.directory/name, a, delimiter=',', fmt='%.17g')
            m['file_sha256'][name] = file_hash(self.directory/name)
        m.update(has_validation=True, validation_K=80, config=dict(selection='validation'))
        write_toml(self.directory/'metadata.toml', m)
        with self.assertRaisesRegex(ValueError, 'trajectories overlap'):
            load_request(self.directory)

    def test_plain_validation_uses_training_means_and_keeps_all_candidates(self):
        _, theta, Y, _ = request(self.directory)
        X, y = theta.T, Y[0]
        validation = dict(X_validation=X[40:], y_validation=y[40:])
        c, d = fit_plain_lasso(X[:40], y[:40], dict(selection='validation', validation_alphas=(.01, .001)), validation)
        np.testing.assert_array_equal(d['feature_means'], X[:40].mean(axis=0))
        self.assertEqual(len(d['validation_path']), 2)
        self.assertLess(np.mean((validation['X_validation']@c-validation['y_validation'])**2), .001)


if __name__ == '__main__':
    unittest.main()
