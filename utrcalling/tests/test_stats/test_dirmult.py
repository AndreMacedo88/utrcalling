"""Tests for utrcalling.core.stats.dirmult: the Dirichlet-multinomial regression."""

import unittest
from unittest import mock

import numpy as np
import scipy.optimize
import scipy.stats

from utrcalling.core.stats import dirmult


def _simulate(n=1500, precision=6.0, depth=4.0, seed=0, beta=None):
    rng = np.random.default_rng(seed)
    x = rng.random(n)
    design = np.column_stack([np.ones(n), x, (rng.random(n) < 0.5).astype(float)])
    if beta is None:
        beta = np.array([[0.4, -0.3], [1.5, -1.0], [-0.5, 0.2]])
    shares = dirmult.proportions(beta, design)
    totals = rng.poisson(depth, n) + 1
    counts = np.array(
        [
            rng.multinomial(t, rng.dirichlet(precision * p))
            for t, p in zip(totals, shares, strict=True)
        ]
    )
    return design, counts, beta


class TestCounts(unittest.TestCase):
    def test_entries_and_totals(self):
        counts = dirmult.Counts.from_matrix([[2, 0, 1], [0, 3, 0]])
        self.assertEqual(counts.n_units, 2)
        self.assertEqual(counts.n_sites, 3)
        np.testing.assert_array_equal(counts.totals, [3, 3])
        np.testing.assert_array_equal(counts.values, [2, 1, 3])
        np.testing.assert_array_equal(counts.dense(), [[2, 0, 1], [0, 3, 0]])


class TestLikelihood(unittest.TestCase):
    def test_matches_scipys_dirichlet_multinomial(self):
        design, counts, beta = _simulate(n=200)
        precision = 3.5
        shares = dirmult.proportions(beta, design)
        expected = sum(
            scipy.stats.dirichlet_multinomial.logpmf(y, precision * p, y.sum())
            for y, p in zip(counts, shares, strict=True)
        )
        data = dirmult.Counts.from_matrix(counts)
        found = dirmult.log_likelihood(beta, precision, design, data) + data.constant()
        self.assertAlmostEqual(found, expected, places=6)

    def test_proportions_are_a_softmax_with_the_last_site_as_reference(self):
        shares = dirmult.proportions([[np.log(2.0), 0.0]], [[1.0]])
        np.testing.assert_allclose(shares, [[0.5, 0.25, 0.25]])

    def test_shares_too_small_for_floating_point_keep_a_finite_likelihood(self):
        data = dirmult.Counts.from_matrix([[1, 1]])
        likelihood = dirmult.log_likelihood([[-800.0]], 5.0, [[1.0]], data)
        self.assertTrue(np.isfinite(likelihood))
        self.assertLess(likelihood, -790)

    def test_gradient_matches_finite_differences(self):
        design, counts, beta = _simulate(n=300)
        data = dirmult.Counts.from_matrix(counts)
        joint = np.append(beta.ravel(order="F"), np.log(4.0))
        for parameters, precision in ((joint, None), (beta.ravel(order="F"), 4.0)):
            _, gradient = dirmult._objective(parameters, design, data, precision, 3.0, 0.4)
            numeric = scipy.optimize.approx_fprime(
                parameters,
                lambda p, s=precision: dirmult._objective(p, design, data, s, 3.0, 0.4)[0],
                1e-6,
            )
            np.testing.assert_allclose(gradient, numeric, rtol=1e-4, atol=1e-3)


class TestFit(unittest.TestCase):
    def test_recovers_the_coefficients_and_the_precision(self):
        design, counts, beta = _simulate(n=4000, precision=6.0)
        fitted = dirmult.fit(design, dirmult.Counts.from_matrix(counts))
        self.assertTrue(fitted.converged)
        np.testing.assert_allclose(fitted.beta, beta, atol=0.25)
        self.assertAlmostEqual(np.log(fitted.precision), np.log(6.0), delta=0.25)

    def test_at_a_fixed_precision_only_the_coefficients_move(self):
        design, counts, _ = _simulate(n=800)
        data = dirmult.Counts.from_matrix(counts)
        fitted = dirmult.fit(design, data, precision=2.0)
        self.assertEqual(fitted.precision, 2.0)
        self.assertAlmostEqual(
            fitted.log_likelihood, dirmult.log_likelihood(fitted.beta, 2.0, design, data)
        )
        moved = fitted.beta + 0.05
        self.assertLess(dirmult.log_likelihood(moved, 2.0, design, data), fitted.log_likelihood)

    def test_the_penalty_pulls_the_precision_toward_its_target(self):
        design, counts, _ = _simulate(n=60, precision=6.0)
        data = dirmult.Counts.from_matrix(counts)
        free = dirmult.fit(design, data)
        pulled = dirmult.fit(design, data, target=200.0, strength=50.0)
        self.assertGreater(pulled.precision, free.precision)

    def test_starts_from_the_better_of_two_starting_points(self):
        design, counts, beta = _simulate(n=300)
        data = dirmult.Counts.from_matrix(counts)
        pooled = dirmult.pooled_beta(design, data)
        binomial = dirmult.binomial_beta(design, data)
        # The pooled start gives every unit the pooled proportions.
        shares = dirmult.proportions(pooled, design)
        pooled_shares = (counts.sum(axis=0) + 0.5) / (counts.sum() + 1.5)
        np.testing.assert_allclose(shares, np.broadcast_to(pooled_shares, shares.shape), atol=1e-9)
        # The binomial start follows the design.
        np.testing.assert_allclose(binomial, beta, atol=0.6)
        for precision in (0.1, 5.0):
            likelihoods = [
                dirmult.log_likelihood(b, precision, design, data) for b in (pooled, binomial)
            ]
            better = (pooled, binomial)[int(np.argmax(likelihoods))]
            with mock.patch.object(
                dirmult.scipy.optimize, "minimize", wraps=scipy.optimize.minimize
            ) as minimize:
                dirmult.fit(design, data, precision=precision)
            scale = np.abs(design).max(axis=0)[:, None]
            np.testing.assert_allclose(
                minimize.call_args.args[1], (better * scale).ravel(order="F")
            )

    def test_a_given_start_and_the_precision_limits(self):
        design, counts, beta = _simulate(n=300)
        data = dirmult.Counts.from_matrix(counts)
        fitted = dirmult.fit(design, data, start=beta, start_precision=1e9)
        low, high = np.exp(dirmult.LOG_PRECISION_BOUNDS)
        self.assertTrue(low <= fitted.precision <= high)

    def test_a_site_absent_from_a_group_still_converges(self):
        design, counts, _ = _simulate(n=400)
        counts[design[:, 2] == 1, 0] = 0  # site 1 never seen in the second group
        fitted = dirmult.fit(design, dirmult.Counts.from_matrix(counts), precision=6.0)
        shares = dirmult.proportions(fitted.beta, design)
        self.assertTrue(np.isfinite(fitted.log_likelihood))
        self.assertLess(shares[design[:, 2] == 1, 0].max(), 1e-3)

    def test_a_stop_with_a_flat_gradient_counts_as_converged(self):
        design, counts, _ = _simulate(n=200)
        data = dirmult.Counts.from_matrix(counts)
        real = scipy.optimize.minimize

        def stopped(gradient):
            def minimize(*args, **kwargs):
                result = real(*args, **kwargs)
                result.success = False
                result.jac = np.full_like(result.jac, gradient)
                return result

            return minimize

        with mock.patch.object(dirmult.scipy.optimize, "minimize", side_effect=stopped(1e-9)):
            self.assertTrue(dirmult.fit(design, data, precision=4.0).converged)
        with mock.patch.object(dirmult.scipy.optimize, "minimize", side_effect=stopped(1.0)):
            self.assertFalse(dirmult.fit(design, data, precision=4.0).converged)


class TestMomentPrecision(unittest.TestCase):
    def test_estimates_the_precision_of_deep_units(self):
        rng = np.random.default_rng(1)
        shares = np.array([0.5, 0.3, 0.2])
        counts = np.array([rng.multinomial(50, rng.dirichlet(10 * shares)) for _ in range(400)])
        estimate = dirmult.moment_precision(dirmult.Counts.from_matrix(counts))
        self.assertAlmostEqual(np.log(estimate), np.log(10), delta=0.3)

    def test_units_without_overdispersion_give_the_upper_limit(self):
        rng = np.random.default_rng(2)
        counts = rng.multinomial(40, [0.5, 0.5], size=300)
        estimate = dirmult.moment_precision(dirmult.Counts.from_matrix(counts))
        self.assertEqual(estimate, np.exp(dirmult.LOG_PRECISION_BOUNDS[1]))

    def test_single_molecules_say_nothing(self):
        counts = np.eye(2)[np.arange(20) % 2]
        self.assertTrue(np.isnan(dirmult.moment_precision(dirmult.Counts.from_matrix(counts))))


class TestLoess(unittest.TestCase):
    def test_reproduces_a_quadratic(self):
        x = np.linspace(-2, 3, 300)
        y = 1 + 0.5 * x - 0.3 * x**2
        np.testing.assert_allclose(dirmult.loess(x, y), y, atol=1e-4)

    def test_smooths_noise_around_a_trend(self):
        rng = np.random.default_rng(3)
        x = rng.random(2000) * 4
        trend = np.sin(x)
        fitted = dirmult.loess(x, trend + rng.normal(0, 0.3, x.size))
        self.assertLess(np.abs(fitted - trend).mean(), 0.06)

    def test_equal_predictors(self):
        np.testing.assert_allclose(dirmult.loess(np.ones(5), np.arange(5.0)), 2.0)


class TestVarianceRatios(unittest.TestCase):
    def test_near_one_for_the_true_model(self):
        design, counts, beta = _simulate(n=3000, precision=6.0)
        data = dirmult.Counts.from_matrix(counts)
        ratios = dirmult.variance_ratios(dirmult.proportions(beta, design), data, 6.0)
        np.testing.assert_allclose(ratios, 1.0, atol=0.15)
        # Too high a precision promises less spread than there is.
        tight = dirmult.variance_ratios(dirmult.proportions(beta, design), data, 1e4)
        self.assertTrue((tight > 1.3).all())

    def test_fewer_than_three_units(self):
        data = dirmult.Counts.from_matrix([[1, 1], [2, 0]])
        self.assertTrue(np.isnan(dirmult.variance_ratios(np.full((2, 2), 0.5), data, 3.0)).all())
