import copy
from pathlib import Path
import sys
import tempfile
import tomllib
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from r5_noise import response_noise_basis, inject_response_noise, true_strength_group, validate_config, noise_slot
from r5_experiment import fit_trial
from src.solvers import BICNotApplicable, SolverConvergenceError


class NoiseTests(unittest.TestCase):
    def setUp(self):
        self.config = tomllib.loads((ROOT/'configs/r5_e1_smoke.toml').read_text())
        self.splits = np.array(['train']*6+['validation']*3+['test']*3)
        self.Y = np.arange(24., dtype=float).reshape(2, 12)/7

    def basis(self, Y=None, seed=22261017):
        return response_noise_basis(self.Y if Y is None else Y, self.splits, seed, self.config['noise_master_seed'])

    def test_scope_rejects_batch_E2_E3_and_other_levels(self):
        validate_config(self.config)
        for patch_values in ({'experiment':'E2'}, {'experiment':'E3'}, {'phase':'batch'}, {'seeds':list(range(30))},
                             {'noise_ratios':[0., .1]}, {'E2_authorized':True}, {'full_30_seed_authorized':True}):
            c = copy.deepcopy(self.config);c.update(patch_values)
            with self.assertRaises(ValueError):
                validate_config(c)

    def test_scales_use_only_clean_training_responses(self):
        scales, unit, _ = self.basis()
        changed = self.Y.copy();changed[:, 6:] = 1e8
        other_scales, other_unit, _ = self.basis(changed)
        np.testing.assert_array_equal(scales, other_scales)
        np.testing.assert_array_equal(unit, other_unit)
        np.testing.assert_allclose(scales, np.std(self.Y[:, :6], axis=1, ddof=0))

    def test_deterministic_noise_streams(self):
        a, u, streams = self.basis();b, v, other = self.basis()
        np.testing.assert_array_equal(a, b);np.testing.assert_array_equal(u, v)
        self.assertEqual(streams, other)
        self.assertEqual(len({tuple(r['entropy']) for r in streams}), 6)

    def test_seed_node_and_split_have_distinct_streams(self):
        _, unit, _ = self.basis();_, other, _ = self.basis(seed=22261018)
        self.assertFalse(np.array_equal(unit, other))
        self.assertFalse(np.array_equal(unit[0], unit[1]))
        self.assertFalse(np.array_equal(unit[0, 6:9], unit[0, 9:]))

    def test_zero_noise_is_bitwise_clean(self):
        scales, unit, _ = self.basis()
        Y, eta = inject_response_noise(self.Y, scales, unit, 0.)
        np.testing.assert_array_equal(Y, self.Y);np.testing.assert_array_equal(eta, np.zeros_like(self.Y))

    def test_common_random_numbers_scale_linearly(self):
        scales, unit, _ = self.basis()
        _, low = inject_response_noise(self.Y, scales, unit, .05)
        _, high = inject_response_noise(self.Y, scales, unit, .20)
        np.testing.assert_array_equal(high, low*4)

    def test_std_ratio_is_not_variance_ratio(self):
        splits = np.array(['train']*10000+['validation']*10000+['test']*10000)
        clean = np.tile([-2., 2.], 15000).reshape(1, -1)
        scales, unit, _ = response_noise_basis(clean, splits, 1, 2)
        _, eta = inject_response_noise(clean, scales, unit, .1)
        self.assertEqual(scales[0], 2.)
        self.assertAlmostEqual(float(np.std(eta[:, :10000])), .2, delta=.01)

    def test_noise_functions_do_not_mutate_inputs(self):
        original = self.Y.copy();scales, unit, _ = self.basis()
        before_scales, before_unit = scales.copy(), unit.copy()
        inject_response_noise(self.Y, scales, unit, .1)
        np.testing.assert_array_equal(original, self.Y)
        np.testing.assert_array_equal(scales, before_scales);np.testing.assert_array_equal(unit, before_unit)

    def test_constant_training_signal_gets_zero_absolute_noise(self):
        constant = np.ones_like(self.Y)
        scales, unit, _ = self.basis(constant)
        changed, eta = inject_response_noise(constant, scales, unit, .2)
        np.testing.assert_array_equal(eta, np.zeros_like(eta));np.testing.assert_array_equal(changed, constant)

    def test_true_strength_bins_and_boundaries(self):
        self.assertEqual([true_strength_group(v) for v in (.08, -.10, .14, -.17, .20)],
                         ['weak', 'weak', 'medium', 'medium', 'strong'])

    def test_noise_directory_ids_do_not_collide(self):
        self.assertEqual([noise_slot(r) for r in self.config['noise_ratios']],
                         ['noise_000', 'noise_050', 'noise_100', 'noise_200'])

    def test_bad_injection_contract_rejected(self):
        scales, unit, _ = self.basis()
        with self.assertRaises(ValueError):
            inject_response_noise(self.Y, scales, unit[:, :-1], .1)
        with self.assertRaises(ValueError):
            inject_response_noise(self.Y, scales, unit, -.1)

    def mock_failure(self, exception, method):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary);source = root/'source';source.mkdir()
            (source/'metadata.toml').write_text('seed = 1\n')
            noise = root/'noise';noise.mkdir();(noise/'metadata.toml').write_text('ratio = 0.1\n')
            np.savetxt(noise/'Y_noisy.csv', self.Y, delimiter=',')
            data = dict(directory=source, metadata=dict(seed=1), splits=self.splits, groups=np.arange(12),
                        Z=np.vstack([np.ones(12), np.arange(12.), np.arange(12.)**2]), Y=self.Y)
            trial = root/'trial'
            with patch('r5_experiment.fit_node', side_effect=exception):
                fit_trial(method, data, dict(nodes=2), trial, .1, noise/'metadata.toml')
            self.assertFalse((trial/'coefficients.csv').exists())
            return tomllib.loads((trial/'status.toml').read_text())

    def test_BIC_NA_is_retained_without_fake_coefficients(self):
        status = self.mock_failure(BICNotApplicable('explicit guard'), 'M1')
        self.assertEqual(status['status'], 'NA');self.assertEqual(status['unavailable_nodes'], [1, 2])

    def test_selected_convergence_failure_is_retained_without_fake_coefficients(self):
        status = self.mock_failure(SolverConvergenceError('KKT failure'), 'M3')
        self.assertEqual(status['status'], 'FAILED');self.assertIn('KKT failure', status['reason'])


if __name__ == '__main__':
    unittest.main()
