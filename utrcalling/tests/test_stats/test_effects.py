"""Tests for utrcalling.core.stats.effects: effect sizes and descriptive measures."""

import unittest

import numpy as np

from utrcalling.core.stats import dirmult, effects
from utrcalling.core.stats.design import Contrast


def _grid(start=0.0, end=1.0, points=5):
    values = tuple(np.linspace(start, end, points))
    return Contrast("pseudotime", "numeric", values, start, end)


class TestEffectSizes(unittest.TestCase):
    def test_a_shift_toward_the_longest_site(self):
        # Two sites; the log-odds of the shortest against the longest falls with x.
        contrast = _grid()
        at_points = np.array([[[1.0, x], [1.0, x]] for x in contrast.points])
        beta = np.array([[0.0], [-2.0]])
        gene, sites = effects.effect_sizes(beta, at_points, np.array([0.0, 1.0]), contrast)
        start = dirmult.proportions(beta, [[1.0, 0.0]])[0]
        end = dirmult.proportions(beta, [[1.0, 1.0]])[0]
        self.assertAlmostEqual(gene["length_index_from"], start[1])
        self.assertAlmostEqual(gene["length_index_to"], end[1])
        self.assertAlmostEqual(gene["delta_length_index"], end[1] - start[1])
        self.assertGreater(gene["delta_length_index"], 0)
        self.assertAlmostEqual(gene["tvd"], abs(end[0] - start[0]))
        self.assertAlmostEqual(gene["max_abs_delta_proportion"], abs(end[0] - start[0]))
        self.assertAlmostEqual(gene["length_index_range"], gene["delta_length_index"])
        self.assertEqual((gene["min_at"], gene["max_at"]), (0.0, 1.0))
        self.assertAlmostEqual(gene["non_monotonicity"], 0.0)
        np.testing.assert_allclose(sites["delta_proportion"], end - start)
        np.testing.assert_allclose(sites["proportion_from"], start)

    def test_a_rise_and_return_is_not_monotonic(self):
        contrast = _grid(points=21)
        x = np.array(contrast.points)
        # Log-odds of the longest site against the shortest: high in the middle.
        at_points = np.stack(
            [np.column_stack([np.ones(2), np.full(2, v), np.full(2, v**2)]) for v in x]
        )
        beta = np.array([[0.0], [-8.0], [8.0]])  # shortest site's log-odds: low mid-way
        gene, _ = effects.effect_sizes(beta, at_points, np.array([0.0, 1.0]), contrast)
        self.assertAlmostEqual(gene["delta_length_index"], 0.0, places=10)
        self.assertGreater(gene["length_index_range"], 0.3)
        self.assertAlmostEqual(gene["max_at"], 0.5)
        self.assertAlmostEqual(gene["non_monotonicity"], 1.0)

    def test_a_fall_has_a_negative_range(self):
        contrast = _grid()
        at_points = np.array([[[1.0, x]] for x in contrast.points])
        gene, _ = effects.effect_sizes(
            np.array([[0.0], [2.0]]), at_points, np.array([0.0, 1.0]), contrast
        )
        self.assertLess(gene["length_index_range"], 0)
        self.assertAlmostEqual(gene["length_index_range"], gene["delta_length_index"])

    def test_no_change_at_all(self):
        contrast = _grid()
        at_points = np.array([[[1.0, x]] for x in contrast.points])
        gene, _ = effects.effect_sizes(
            np.array([[0.3], [0.0]]), at_points, np.array([0.0, 1.0]), contrast
        )
        self.assertEqual(gene["non_monotonicity"], 0.0)
        self.assertEqual(gene["tvd"], 0.0)

    def test_categories_have_no_trajectory_measures(self):
        contrast = Contrast("condition", "categorical", ("ctl", "trt"), "ctl", "trt")
        at_points = np.array([[[1.0, 0.0]], [[1.0, 1.0]]])
        gene, sites = effects.effect_sizes(
            np.array([[0.0, 0.0], [1.0, -1.0]]), at_points, np.array([0.0, 0.5, 1.0]), contrast
        )
        self.assertNotIn("non_monotonicity", gene)
        self.assertNotIn("length_index_range", gene)
        self.assertEqual(len(sites["delta_proportion"]), 3)
        self.assertAlmostEqual(sites["delta_proportion"].sum(), 0.0)


class TestDescriptive(unittest.TestCase):
    def test_correlations_with_a_numeric_variable(self):
        shares = np.array([[1.0, 0.0], [0.8, 0.2], [0.5, 0.5], [0.1, 0.9]])
        values = np.array([0.1, 0.2, 0.3, 0.4])
        found = effects.descriptive(shares, values, np.array([0.0, 1.0]), _grid())
        self.assertAlmostEqual(found["rho_length_index"], 1.0)
        # Entropy rises then falls: not monotonic in the values.
        self.assertLess(abs(found["rho_entropy"]), 1.0)

    def test_constant_or_too_few_units_give_no_correlation(self):
        shares = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        found = effects.descriptive(
            shares, np.array([1.0, 2.0, 3.0]), np.array([0.0, 1.0]), _grid()
        )
        self.assertTrue(np.isnan(found["rho_length_index"]))
        two = effects.descriptive(shares[:2], np.array([1.0, 2.0]), np.array([0.0, 1.0]), _grid())
        self.assertTrue(np.isnan(two["rho_entropy"]))

    def test_differences_of_means_between_two_levels(self):
        contrast = Contrast("condition", "categorical", ("ctl", "trt"), "ctl", "trt")
        shares = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [0.0, 1.0], [0.3, 0.7]])
        values = np.array(["ctl", "ctl", "trt", "trt", "other"])
        found = effects.descriptive(shares, values, np.array([0.0, 1.0]), contrast)
        self.assertAlmostEqual(found["observed_delta_length_index"], 1.0 - 0.25)
        self.assertAlmostEqual(found["observed_delta_entropy"], 0.0 - 0.5)

    def test_a_missing_level_gives_no_difference(self):
        contrast = Contrast("condition", "categorical", ("ctl", "trt"), "ctl", "trt")
        found = effects.descriptive(
            np.array([[1.0, 0.0]]), np.array(["ctl"]), np.array([0.0, 1.0]), contrast
        )
        self.assertTrue(np.isnan(found["observed_delta_length_index"]))
