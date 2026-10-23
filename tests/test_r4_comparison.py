import copy
from pathlib import Path
import sys
import tomllib
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from r4_metrics import binary_metrics, edge_scores, restore_coefficients, validate_phase, validate_splits
from r4_methods import fit_node
from src.solvers import BICNotApplicable


class R4Tests(unittest.TestCase):
    def setUp(self):
        self.config = tomllib.loads((ROOT/'configs/r4_dry_run.toml').read_text())

    def test_phase_A_is_exactly_two_seeds(self):
        validate_phase(self.config)
        for patch in ({'phase':'B'}, {'seeds':list(range(30))}, {'full_phase_B_authorized':True},
                      {'seeds':[1, 1]}, {'pearson_enabled':True}, {'train_K':1000}):
            c = copy.deepcopy(self.config);c.update(patch)
            with self.assertRaises(ValueError):
                validate_phase(c)

    def test_counts_and_average_precision(self):
        m = binary_metrics([1, 0, 1, 0], [.9, .8, .2, .1], .5)
        self.assertEqual([m[k] for k in ('TP', 'FP', 'FN', 'TN')], [1, 1, 1, 1])
        self.assertEqual(m['F1'], .5)
        self.assertEqual(m['MCC'], 0.)
        self.assertAlmostEqual(m['AUPRC_AP'], (1.+2/3)/2)

    def test_strict_threshold(self):
        m = binary_metrics([1, 1], [1e-8, np.nextafter(1e-8, 1.)])
        self.assertEqual((m['TP'], m['FN']), (1, 1))

    def test_empty_positive_truth_marks_AP_NA(self):
        m = binary_metrics([0, 0, 0], [0., .2, 0.])
        self.assertEqual(m['AUPRC_status'], 'NA')
        self.assertEqual(m['AUPRC_reason'], 'no_positive_truth')
        self.assertTrue(np.isnan(m['AUPRC_AP']))
        self.assertTrue(np.isnan(m['recall']))

    def test_all_positive_truth_marks_AP_NA(self):
        m = binary_metrics([1, 1], [.3, .2])
        self.assertEqual(m['AUPRC_reason'], 'no_negative_truth')

    def test_no_predicted_edges_keeps_FN_and_undefined_MCC(self):
        m = binary_metrics([1, 0, 0], [0., 0., 0.])
        self.assertEqual((m['TP'], m['FP'], m['FN'], m['TN']), (0, 0, 1, 2))
        self.assertEqual(m['F1'], 0.)
        self.assertEqual(m['MCC_status'], 'NA')
        self.assertAlmostEqual(m['AUPRC_AP'], 1/3)

    def test_empty_candidate_set_is_defined_without_AP(self):
        m = binary_metrics([], [])
        self.assertEqual(m['candidate_count'], 0)
        self.assertEqual(m['AUPRC_status'], 'NA')

    def test_edge_universe_sizes_and_imbalance(self):
        d = np.zeros((1, 3), int);C = np.zeros((8, 1))
        self.assertEqual(len(edge_scores(C, d, 2)[0]), 56)
        self.assertEqual(len(edge_scores(C, d, 3)[0]), 168)
        m = binary_metrics([1]+[0]*167, [1.]+[0.]*167)
        self.assertEqual(m['positive_count'], 1)
        self.assertEqual(m['negative_count'], 167)
        self.assertEqual(m['AUPRC_AP'], 1.)

    def test_repeated_sources_and_targets_do_not_become_edges(self):
        d = np.array([[0, 0], [1, 1], [1, 2]])
        C = np.zeros((3, 3));C[2, 1] = 9.;C[0, 2] = 8.;C[2, 2] = -.2
        keys, scores = edge_scores(C, d, 3)
        self.assertEqual(dict(zip(keys, scores)), {(1, 2, 3):0., (2, 1, 3):0., (3, 1, 2):.2})

    def test_max_absolute_aggregation_for_same_edge_key(self):
        d = np.array([[0, 0], [1, 2], [2, 1]])
        C = np.zeros((3, 3));C[2, 1:] = [.2, -.7]
        keys, scores = edge_scores(C, d, 3)
        self.assertEqual(dict(zip(keys, scores))[(3, 1, 2)], .7)

    def test_taylor_support_differs_from_decoded_edges(self):
        d = np.array([[0, 0], [0, 1], [1, 1]])
        C = np.zeros((3, 3));C[0, 1] = -1.;C[2, 2] = .3
        self.assertEqual(np.count_nonzero(C), 2)
        self.assertEqual(np.count_nonzero(edge_scores(C, d, 2)[1]), 0)
        self.assertEqual(np.count_nonzero(edge_scores(C, d, 3)[1]), 0)

    def test_original_unit_prediction_restoration(self):
        rng = np.random.default_rng(81)
        raw = np.column_stack([np.ones(120), rng.normal(size=(120, 4))*[.01, 100., 3., 8.]+[7., -3., 20., 1.]])
        means = raw.mean(axis=0);means[0] = 0.
        scales = np.sqrt(np.mean((raw-means)**2, axis=0));scales[0] = 1.
        Z = (raw-means)/scales
        C = rng.normal(size=(3, 5))
        restored = restore_coefficients(C, means, scales)
        np.testing.assert_allclose(raw@restored.T, Z@C.T, atol=1e-11, rtol=1e-11)

    def test_trajectory_splits_must_be_disjoint(self):
        self.assertEqual(validate_splits([1, 1, 2, 3], ['train', 'train', 'validation', 'test']),
                         dict(train=1, validation=1, test=1))
        with self.assertRaisesRegex(ValueError, 'crosses'):
            validate_splits([1, 1, 2], ['train', 'validation', 'test'])
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            validate_splits([1, 2], ['train', 'validation'])

    def fixture(self, noise):
        rng = np.random.default_rng(15)
        X = np.column_stack([np.ones(240), rng.normal(size=(240, 5))])
        y = X@np.array([.3, 1.2, -.4, 0., 0., 0.])+rng.normal(0., noise, 240)
        return X, y

    def test_both_BIC_methods_work_on_valid_noisy_fixture(self):
        X, y = self.fixture(.1)
        results = [fit_node(method, X, y, self.config) for method in ('M1', 'M2')]
        for C, diagnostics in results:
            self.assertEqual(diagnostics['selection'], 'bic')
            self.assertLess(np.mean((X@C-y)**2), .03)
        self.assertAlmostEqual(results[0][1]['noise_variance'], results[1][1]['noise_variance'])

    def test_both_BIC_methods_reject_exact_clean_data_without_fallback(self):
        X, y = self.fixture(0.)
        for method in ('M1', 'M2'):
            with self.subTest(method=method), self.assertRaisesRegex(BICNotApplicable, 'noise variance'):
                fit_node(method, X, y, self.config)

    def test_BIC_rejects_rank_deficiency_for_both_methods(self):
        X, y = self.fixture(.1);X[:, -1] = X[:, 1]
        for method in ('M1', 'M2'):
            with self.subTest(method=method), self.assertRaisesRegex(BICNotApplicable, 'Rank-deficient'):
                fit_node(method, X, y, self.config)

    def test_matched_validation_methods_remain_distinct_groups(self):
        X, y = self.fixture(.05)
        validation = dict(X_validation=X[120:], y_validation=y[120:], train_groups=np.arange(120), validation_groups=np.arange(120, 240))
        for method in ('M1v', 'M3'):
            C, diagnostics = fit_node(method, X[:120], y[:120], self.config, validation)
            self.assertEqual(diagnostics['selection'], 'validation')
            self.assertEqual(len(diagnostics['validation_path']), 12)
            self.assertLess(np.mean((X[120:]@C-y[120:])**2), .03)


if __name__ == '__main__':
    unittest.main()
