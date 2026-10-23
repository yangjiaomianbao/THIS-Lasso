import unittest

import numpy as np

from src.solvers import AdaptiveLassoConfig, BICNotApplicable, SolverConvergenceError, fit_adaptive_lasso


def fixture(seed=7, n=240, p=8, noise=.05):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.zeros(p)
    beta[[0, 2, 4]] = [1.5, -.9, .35]
    y = 1.2 + X@beta + rng.normal(0, noise, n)
    return X, y, beta


def validation_args(X, y):
    cut = len(y)//2
    return X[:cut], y[:cut], dict(X_validation=X[cut:], y_validation=y[cut:],
        train_groups=np.arange(cut), validation_groups=np.arange(cut, len(y)))


class AdaptiveLassoTests(unittest.TestCase):
    def test_known_sparse_noisy_coefficients(self):
        X, y, truth = fixture()
        coef, d = fit_adaptive_lasso(X, y, dict(selection='fixed', alpha=.005))
        np.testing.assert_allclose(coef, truth, atol=.06)
        self.assertLess(np.mean((X@coef+d['intercept']-y)**2), .02)
        self.assertEqual(set(np.flatnonzero(coef)), {0, 2, 4})

    def test_bic_on_valid_noisy_full_rank_data(self):
        X, y, _ = fixture()
        coef, d = fit_adaptive_lasso(X, y)
        self.assertEqual(d['selection'], 'bic')
        self.assertGreater(d['noise_variance'], 0)
        self.assertEqual(d['noise_degrees_of_freedom'], len(y)-X.shape[1]-1)
        self.assertLess(np.mean((X@coef+d['intercept']-y)**2), .02)

    def test_weighted_and_transformed_objectives_are_equal(self):
        X, y, _ = fixture()
        coef, d = fit_adaptive_lasso(X, y, dict(selection='fixed', alpha=.02, gamma=1.3))
        U = (X-d['feature_means'])*d['adaptive_scales']
        theta, alpha = d['scaled_lasso_coef'], d['alpha']
        raw = np.mean((y-X@coef-d['intercept'])**2)/2 + alpha*np.sum(d['weights']*np.abs(coef))
        transformed = np.mean((y-d['response_mean']-U@theta)**2)/2 + alpha*np.sum(np.abs(theta))
        self.assertAlmostEqual(raw, transformed, places=12)
        self.assertAlmostEqual(raw, d['objective'], places=12)
        np.testing.assert_allclose(coef, d['adaptive_scales']*theta, atol=1e-12)

    def test_explicit_weight_formula_in_original_units(self):
        X, y, _ = fixture()
        X *= np.array([.01, 100, 2, 5, .2, 10, 3, .1])
        coef, d = fit_adaptive_lasso(X, y, dict(selection='fixed', alpha=.01, gamma=1.2))
        expected = (np.abs(d['initial_coef'])+d['config']['eps'])**(-d['config']['gamma'])
        np.testing.assert_allclose(d['weights'], expected)
        self.assertLess(abs(d['objective']-d['transformed_objective']), 1e-10)

    def test_independent_kkt_check(self):
        X, y, _ = fixture()
        coef, d = fit_adaptive_lasso(X, y, dict(selection='fixed', alpha=.02))
        gradient = X.T@(X@coef+d['intercept']-y)/len(y)
        active = coef != 0
        self.assertLess(np.max(np.abs(gradient[active]+d['alpha']*d['weights'][active]*np.sign(coef[active]))), 1e-6)
        self.assertTrue(np.all(np.abs(gradient[~active]) <= d['alpha']*d['weights'][~active]+1e-6))
        self.assertLess(abs(np.mean(X@coef+d['intercept']-y)), 1e-12)

    def test_scale_and_offset_prediction_restoration(self):
        X, y, _ = fixture()
        X = X*np.array([.001, 1000, 2, 4, .1, 20, 3, .3])+np.arange(8)*10
        coef, d = fit_adaptive_lasso(X, y, dict(selection='fixed', alpha=.005))
        U = (X-d['feature_means'])*d['adaptive_scales']
        np.testing.assert_allclose(X@coef+d['intercept'], d['response_mean']+U@d['scaled_lasso_coef'], atol=1e-9)
        self.assertLess(d['prediction_restore_max_abs'], 1e-9)

    def test_free_constant_column_even_without_additional_intercept(self):
        rng = np.random.default_rng(2)
        variables = rng.normal(size=(120, 4))
        variables -= variables.mean(axis=0)
        X = np.column_stack([np.full(120, 2.0), variables])
        y = 3.0+variables@np.array([1., -.5, 0., 0.])
        coef, d = fit_adaptive_lasso(X, y, dict(selection='fixed', alpha=100., fit_intercept=False))
        self.assertAlmostEqual(coef[0], 1.5, places=12)
        self.assertEqual(d['intercept'], 0)
        self.assertEqual(d['weights'][0], 0)
        self.assertTrue(np.all(coef[1:]==0))

    def test_constant_column_prediction_and_no_duplicate_intercept(self):
        X, y, _ = fixture()
        X = np.column_stack([np.ones(len(y)), X])
        coef, d = fit_adaptive_lasso(X, y, dict(selection='fixed', alpha=.005))
        self.assertEqual(d['constant_index'], 0)
        self.assertEqual(d['intercept'], 0)
        self.assertLess(d['prediction_restore_max_abs'], 1e-12)
        self.assertLess(abs(np.mean(y-X@coef)), 1e-12)

    def test_zero_column_retained_as_zero(self):
        X, y, _ = fixture()
        X = np.column_stack([X, np.zeros(len(y))])
        coef, d = fit_adaptive_lasso(X, y, dict(selection='fixed', alpha=.01))
        self.assertEqual(coef[-1], 0)
        self.assertEqual(d['weights'][-1], 0)

    def test_multiple_nonzero_constant_columns_rejected(self):
        X, y, _ = fixture()
        with self.assertRaisesRegex(ValueError, 'Multiple nonzero constant'):
            fit_adaptive_lasso(np.column_stack([np.ones(len(y)), np.full(len(y), 2.), X]), y)

    def test_constant_only_problem_has_no_model_selection(self):
        coef, d = fit_adaptive_lasso(np.full((20, 1), 2.), np.arange(20.), dict(selection='bic'))
        self.assertEqual(d['selection'], 'constant_only')
        self.assertEqual(d['intercept'], 0)
        self.assertEqual(coef[0], 4.75)

    def test_no_bias_when_no_constant_and_intercept_disabled(self):
        X, _, beta = fixture()
        coef, d = fit_adaptive_lasso(X, X@beta, dict(selection='fixed', alpha=.001, fit_intercept=False))
        self.assertEqual(d['intercept'], 0)
        np.testing.assert_array_equal(d['feature_means'], np.zeros(X.shape[1]))
        np.testing.assert_allclose(coef, beta, atol=.01)

    def test_extremely_small_initial_coefficients_remain_finite(self):
        X, _, _ = fixture()
        coef, d = fit_adaptive_lasso(X, np.full(len(X), 4.), dict(selection='fixed', alpha=.01, eps=1e-8, gamma=2))
        self.assertTrue(np.isfinite(d['weights']).all())
        np.testing.assert_array_equal(coef, np.zeros(X.shape[1]))
        self.assertTrue(d['zero_centered_response'])

    def test_unrepresentable_weights_are_rejected(self):
        X, _, _ = fixture()
        with self.assertRaisesRegex(ValueError, 'overflow/underflow'):
            fit_adaptive_lasso(X, np.zeros(len(X)), dict(selection='fixed', alpha=.01, eps=1e-300, gamma=2))

    def test_ols_initialization_switch(self):
        X, y, _ = fixture()
        coef, d = fit_adaptive_lasso(X, y, dict(selection='fixed', alpha=.01, initialization='ols', standardize=False))
        expected = np.linalg.lstsq(X-X.mean(axis=0), y-y.mean(), rcond=None)[0]
        np.testing.assert_allclose(d['initial_coef'], expected, atol=1e-12)

    def test_fixed_seed_and_cyclic_solver_are_repeatable(self):
        X, y, _ = fixture()
        a, da = fit_adaptive_lasso(X, y)
        b, db = fit_adaptive_lasso(X, y)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(da['alpha'], db['alpha'])

    def test_bic_rejects_K_le_P(self):
        rng = np.random.default_rng(1)
        for n, p in [(30, 60), (30, 30)]:
            with self.subTest(n=n, p=p), self.assertRaisesRegex(BICNotApplicable, 'K<=P'):
                fit_adaptive_lasso(rng.normal(size=(n, p)), rng.normal(size=n))

    def test_bic_rejects_duplicate_features(self):
        X, y, _ = fixture()
        with self.assertRaisesRegex(BICNotApplicable, 'Rank-deficient'):
            fit_adaptive_lasso(np.column_stack([X, X[:, 0]]), y)

    def test_bic_rejects_unreliable_noise_variance(self):
        X, _, beta = fixture()
        with self.assertRaisesRegex(BICNotApplicable, 'noise variance'):
            fit_adaptive_lasso(X, 1.2+X@beta)

    def test_bic_rejects_ill_conditioned_full_model(self):
        X, y, _ = fixture()
        X[:, 1] = X[:, 0]+np.random.default_rng(3).normal(0, 1e-10, len(X))
        with self.assertRaisesRegex(BICNotApplicable, 'unstable'):
            fit_adaptive_lasso(X, y)

    def test_high_dimensional_validation_is_supported(self):
        rng = np.random.default_rng(9)
        X = rng.normal(size=(100, 60))
        y = X[:, 0]-.8*X[:, 1]+rng.normal(0, .02, 100)
        coef, d = fit_adaptive_lasso(X[:30], y[:30], dict(selection='validation', validation_alphas=(.05, .01, .002)),
            X_validation=X[30:], y_validation=y[30:], train_groups=np.arange(30), validation_groups=np.arange(30, 100))
        self.assertEqual(d['selection'], 'validation')
        self.assertLess(np.mean((X[30:]@coef+d['intercept']-y[30:])**2), .2)
        self.assertLess(d['kkt_scaled_violation'], 1e-6)

    def test_duplicate_features_validation_is_explicit_and_valid(self):
        X, y, _ = fixture(n=320)
        X = np.column_stack([X, X[:, 0]])
        train_X, train_y, val = validation_args(X, y)
        coef, d = fit_adaptive_lasso(train_X, train_y, dict(selection='validation'), **val)
        self.assertLess(np.mean((val['X_validation']@coef+d['intercept']-val['y_validation'])**2), .03)

    def test_highly_correlated_features_fixed_objective(self):
        X, y, _ = fixture(n=160)
        X[:, 1] = X[:, 0]+np.random.default_rng(6).normal(0, .02, len(X))
        coef, d = fit_adaptive_lasso(X, y, dict(selection='fixed', alpha=.02))
        self.assertLess(d['kkt_scaled_violation'], 1e-6)
        self.assertLess(d['objective'], .2)

    def test_validation_requires_group_ids(self):
        X, y, _ = fixture()
        with self.assertRaisesRegex(ValueError, 'group IDs'):
            fit_adaptive_lasso(X[:120], y[:120], dict(selection='validation'), X_validation=X[120:], y_validation=y[120:])

    def test_trajectory_leakage_is_rejected(self):
        X, y, _ = fixture()
        with self.assertRaisesRegex(ValueError, 'both training and validation'):
            fit_adaptive_lasso(X[:120], y[:120], dict(selection='validation'), X_validation=X[120:], y_validation=y[120:],
                train_groups=np.arange(120), validation_groups=np.arange(120))

    def test_validation_cannot_change_training_preprocessing_or_weights(self):
        X, y, _ = fixture()
        Xt, yt, val = validation_args(X, y)
        _, a = fit_adaptive_lasso(Xt, yt, dict(selection='validation'), **val)
        changed = dict(val, X_validation=val['X_validation']+20, y_validation=val['y_validation']+50)
        _, b = fit_adaptive_lasso(Xt, yt, dict(selection='validation'), **changed)
        for name in ['feature_means','feature_scales','initial_coef','weights','adaptive_scales']:
            np.testing.assert_array_equal(a[name], b[name])

    def test_validation_rejects_changed_constant_column(self):
        X, y, _ = fixture()
        X = np.column_stack([np.ones(len(y)), X])
        Xt, yt, val = validation_args(X, y)
        val['X_validation'] = val['X_validation'].copy()
        val['X_validation'][:, 0] = 2
        with self.assertRaisesRegex(ValueError, 'constant column changes'):
            fit_adaptive_lasso(Xt, yt, dict(selection='validation'), **val)

    def test_bic_does_not_silently_fall_back(self):
        X, y, _ = fixture(n=40)
        Xt, yt, val = validation_args(np.column_stack([X]*4), y)
        with self.assertRaises(BICNotApplicable):
            fit_adaptive_lasso(Xt, yt, dict(selection='bic'), **val)

    def test_inputs_are_not_mutated(self):
        X, y, _ = fixture()
        before_X, before_y = X.copy(), y.copy()
        fit_adaptive_lasso(X, y, dict(selection='fixed', alpha=.01))
        np.testing.assert_array_equal(X, before_X)
        np.testing.assert_array_equal(y, before_y)

    def test_invalid_inputs_and_configuration(self):
        X, y, _ = fixture()
        for c in [dict(eps=0),dict(gamma=-1),dict(ridge_rho=-1),dict(selection='fixed',alpha=0),dict(selection='unknown')]:
            with self.subTest(config=c), self.assertRaises(ValueError):
                fit_adaptive_lasso(X, y, c)
        for bad_X, bad_y in [(X, y[:, None]), (X[:-1],y), (X*np.nan,y)]:
            with self.assertRaises(ValueError):
                fit_adaptive_lasso(bad_X, bad_y)


if __name__ == '__main__':
    unittest.main(verbosity=2)
