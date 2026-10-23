import copy
from pathlib import Path
import sys
import tomllib
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1];sys.path.insert(0, str(ROOT/'scripts'))
from r5_batch_stats import bootstrap_indices, describe, paired_values, validate_full, trajectory_seed_base


class FullStatisticsTests(unittest.TestCase):
    def test_full_scope_and_seed_lock(self):
        c = tomllib.loads((ROOT/'configs/r5_e1_full_independent.toml').read_text());validate_full(c)
        for changes in ({'full_30_seed_authorized':False}, {'experiment':'E2'}, {'E3_authorized':True},
                        {'seeds':list(range(30))}, {'noise_ratios':[0., .1]}, {'bootstrap_replicates':10},
                        {'trajectory_seed_stride':1}, {'reuse_pilot_clean_data':True}):
            changed = copy.deepcopy(c);changed.update(changes)
            with self.assertRaises(ValueError):validate_full(changed)

    def test_clean_trajectory_namespaces_have_no_cross_trial_overlap(self):
        c = tomllib.loads((ROOT/'configs/r5_e1_full_independent.toml').read_text())
        streams = [set(range(trajectory_seed_base(c, s)+1, trajectory_seed_base(c, s)+161)) for s in c['seeds']]
        self.assertEqual(len(set.union(*streams)), 30*160)
        self.assertFalse(set.union(*streams) & set(range(22261018, 22261207)))

    def test_bootstrap_seed_reproducible(self):
        np.testing.assert_array_equal(bootstrap_indices(30, 100, 17), bootstrap_indices(30, 100, 17))
        self.assertEqual(bootstrap_indices(30, 100, 17).shape, (100, 30))

    def test_constant_metric_has_point_interval(self):
        result = describe(np.ones(30), bootstrap_indices(30, 1000, 17))
        self.assertEqual((result['mean'], result['std'], result['ci_low'], result['ci_high']), (1., 0., 1., 1.))

    def test_pairing_keeps_common_seed_effect(self):
        common = np.arange(30.)*100.
        delta = paired_values(common+1., common)
        result = describe(delta, bootstrap_indices(30, 1000, 17))
        self.assertEqual((result['mean'], result['ci_low'], result['ci_high']), (1., 1., 1.))

    def test_all_NA_not_imputed_as_zero(self):
        result = describe(np.full(30, np.nan), bootstrap_indices(30, 100, 17))
        self.assertEqual(result['n_available'], 0);self.assertTrue(np.isnan(result['mean']))
        self.assertEqual(result['bootstrap_available'], 0)

    def test_partial_NA_has_explicit_conditional_denominator(self):
        result = describe([1., np.nan, 3.], bootstrap_indices(3, 1000, 17))
        self.assertEqual((result['n_total'], result['n_available'], result['mean']), (3, 2, 2.))
        self.assertAlmostEqual(result['std'], np.sqrt(2.))

    def test_pair_requires_both_methods_available(self):
        np.testing.assert_allclose(paired_values([1., np.nan, 3.], [np.nan, 2., 1.]), [np.nan, np.nan, 2.], equal_nan=True)

    def test_nontrivial_interval_and_sample_SD(self):
        result = describe(np.arange(30.), bootstrap_indices(30, 10000, 17))
        self.assertEqual(result['mean'], 14.5);self.assertAlmostEqual(result['std'], np.arange(30.).std(ddof=1))
        self.assertLess(result['ci_low'], result['mean']);self.assertGreater(result['ci_high'], result['mean'])


if __name__ == '__main__':unittest.main()
