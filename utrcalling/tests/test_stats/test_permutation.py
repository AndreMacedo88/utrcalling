"""Tests for utrcalling.core.stats.permutation: shuffles that respect replicates."""

import unittest

import numpy as np
import pandas as pd

from utrcalling.core.stats import permutation


def _cells():
    """Two samples of a condition each, four cells per sample, a pseudotime per cell."""
    return pd.DataFrame(
        {
            "sample": np.repeat(["s1", "s2", "s3", "s4"], 4),
            "condition": np.repeat(["ctl", "ctl", "trt", "trt"], 4),
            "family": np.repeat(["f1", "f2", "f1", "f2"], 4),
            "batch": np.repeat(["b1", "b1", "b2", "b2"], 4),
            "pseudotime": np.arange(16) / 16,
            "phase": np.tile(["G1", "S"], 8),
        }
    )


class TestScheme(unittest.TestCase):
    def test_variables_that_vary_within_replicates_shuffle_within_them(self):
        cells = _cells()
        scheme = permutation.scheme(cells, ["pseudotime"], ["batch"], cells["sample"])
        self.assertEqual(scheme.kind, "within")
        np.testing.assert_array_equal(scheme.replicate, np.repeat([0, 1, 2, 3], 4))

    def test_variables_constant_within_replicates_swap_between_them(self):
        cells = _cells()
        scheme = permutation.scheme(cells, ["condition"], [], cells["sample"], cells["family"])
        self.assertEqual(scheme.kind, "between")
        np.testing.assert_array_equal(scheme.stratum, np.repeat([0, 1, 0, 1], 4))

    def test_units_are_their_own_replicates_by_default(self):
        scheme = permutation.scheme(_cells(), ["pseudotime"])
        self.assertEqual(scheme.kind, "between")
        np.testing.assert_array_equal(scheme.replicate, np.arange(16))
        np.testing.assert_array_equal(scheme.stratum, np.zeros(16))

    def test_warnings_when_shuffles_break_links_with_other_variables(self):
        cells = _cells()
        with self.assertLogs("utrcalling", level="WARNING") as logs:
            permutation.scheme(cells, ["pseudotime"], ["phase"], cells["sample"])
        self.assertIn("phase vary within replicates", logs.output[0])
        with self.assertLogs("utrcalling", level="WARNING") as logs:
            permutation.scheme(cells, ["condition"], ["family"], cells["sample"])
        self.assertIn("family differ between replicates", logs.output[0])
        self.assertIn("--strata family", logs.output[0])

    def test_errors(self):
        cells = _cells()
        with self.assertRaisesRegex(ValueError, "no variable to shuffle"):
            permutation.scheme(cells, [], [], cells["sample"])
        with self.assertRaisesRegex(ValueError, "condition are the same within replicates"):
            permutation.scheme(cells, ["condition", "pseudotime"], [], cells["sample"])
        with self.assertRaisesRegex(ValueError, "--strata applies"):
            permutation.scheme(cells, ["pseudotime"], [], cells["sample"], cells["family"])
        with self.assertRaisesRegex(ValueError, "2 replicates have units in several strata"):
            permutation.scheme(cells, ["condition"], [], cells["condition"], cells["family"])


class TestPermutations(unittest.TestCase):
    def test_counting_distinct_arrangements(self):
        self.assertEqual(permutation.count_permutations([np.array([0, 0, 1, 1])]), 6)
        self.assertEqual(permutation.count_permutations([np.array([0, 1]), np.array([0, 1])]), 4)
        self.assertEqual(permutation.count_permutations([np.arange(5)]), 120)

    def test_all_distinct_arrangements_once_each(self):
        arrangements = permutation._arrangements(np.array([0, 0, 1, 2]))
        self.assertEqual(len(arrangements), 12)
        self.assertEqual(len(set(arrangements)), 12)

    def test_within_replicates_values_stay_in_their_replicate(self):
        cells = _cells()
        scheme = permutation.scheme(cells, ["pseudotime"], [], cells["sample"])
        rows = np.arange(16)
        groups = permutation._groups(scheme, rows, cells)
        sources = permutation.random_permutation(groups, np.random.default_rng(0))
        np.testing.assert_array_equal(
            cells["sample"].to_numpy()[sources], cells["sample"].to_numpy()
        )
        self.assertCountEqual(sources.tolist(), range(16))
        self.assertFalse((sources == rows).all())

    def test_between_replicates_samples_swap_whole(self):
        cells = _cells()
        scheme = permutation.scheme(cells, ["condition"], [], cells["sample"], cells["family"])
        groups = permutation._groups(scheme, np.arange(16), cells)
        swaps = list(permutation.all_permutations(groups))
        self.assertEqual(len(swaps), 4)  # each family's two samples, swapped or not
        condition = cells["condition"].to_numpy()
        for sources in swaps:
            permuted = condition[sources]
            # A sample's cells keep one condition; each family keeps one of each.
            for sample in range(4):
                self.assertEqual(len(set(permuted[sample * 4 : sample * 4 + 4])), 1)
            for family in ("f1", "f2"):
                in_family = (cells["family"] == family).to_numpy()
                self.assertCountEqual(set(permuted[in_family]), {"ctl", "trt"})
        identity = [sources for sources in swaps if (condition[sources] == condition).all()]
        self.assertEqual(len(identity), 1)

    def test_a_utr_with_units_in_some_replicates_only(self):
        cells = _cells()
        scheme = permutation.scheme(cells, ["condition"], [], cells["sample"])
        rows = np.array([0, 1, 5, 9, 10])  # samples s1, s2, s3
        groups = permutation._groups(scheme, rows, cells)
        self.assertEqual(permutation.count_permutations(groups.labels), 3)
        for sources in permutation.all_permutations(groups):
            self.assertTrue(set(sources) <= set(range(5)))


class TestPermutationTest(unittest.TestCase):
    def setUp(self):
        self.cells = _cells()

    def test_exact_when_the_distinct_permutations_are_few(self):
        scheme = permutation.scheme(self.cells, ["condition"], [], self.cells["sample"])
        condition = (self.cells["condition"] == "trt").to_numpy(float)
        response = condition + np.arange(16) * 1e-3  # the observed labels fit best

        def statistic(sources):
            return float(np.corrcoef(condition[sources], response)[0, 1])

        rng = np.random.default_rng(0)
        found = permutation.permutation_test(
            statistic, statistic(np.arange(16)), scheme, np.arange(16), self.cells, rng
        )
        self.assertTrue(found.exact)
        self.assertEqual(found.permutations, 6)
        self.assertAlmostEqual(found.p_value, 1 / 6)

    def test_random_permutations_stop_once_reached_often(self):
        scheme = permutation.scheme(self.cells, ["pseudotime"], [], self.cells["sample"])
        calls = []

        def statistic(sources):
            calls.append(1)
            return 1.0

        rng = np.random.default_rng(0)
        found = permutation.permutation_test(
            statistic, 0.5, scheme, np.arange(16), self.cells, rng, stages=(9, 99), stop_after=5
        )
        self.assertFalse(found.exact)
        self.assertEqual(found.permutations, 9)
        self.assertEqual(len(calls), 9)
        self.assertAlmostEqual(found.p_value, 1.0)

    def test_random_permutations_go_on_while_the_statistic_is_rarely_reached(self):
        scheme = permutation.scheme(self.cells, ["pseudotime"], [], self.cells["sample"])
        rng = np.random.default_rng(0)
        found = permutation.permutation_test(
            lambda sources: 0.0, 1.0, scheme, np.arange(16), self.cells, rng, stages=(9, 99)
        )
        self.assertEqual(found.permutations, 99)
        self.assertAlmostEqual(found.p_value, 1 / 100)

    def test_the_same_seed_gives_the_same_permutations(self):
        scheme = permutation.scheme(self.cells, ["pseudotime"], [], self.cells["sample"])
        groups = permutation._groups(scheme, np.arange(16), self.cells)
        first = permutation.random_permutation(groups, np.random.default_rng([7, 3]))
        second = permutation.random_permutation(groups, np.random.default_rng([7, 3]))
        np.testing.assert_array_equal(first, second)
