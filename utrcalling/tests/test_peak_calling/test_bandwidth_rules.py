"""Tests for utrcalling.core.peak_calling.bandwidth_rules: each UTR's bandwidth by a rule."""

import unittest

import numpy as np

from utrcalling.core.peak_calling import bandwidth_rules, kde
from utrcalling.core.peak_calling.settings import BANDWIDTH_RULES, parse_bandwidth


def range_rule(sizes):
    """A bandwidth rule for the tests: the range of the sizes, over 10."""
    return (sizes.max() - sizes.min()) / 10


class TestUtrBandwidth(unittest.TestCase):
    """Rules set each UTR's bandwidth from the spread of its molecules' sizes."""

    def setUp(self):
        # A 5' UTR with a main start site and a second one 150 nt away.
        rng = np.random.default_rng(0)
        self.sizes = np.concatenate(
            [rng.normal(50, 3, 300).astype(int), rng.normal(200, 3, 100).astype(int)]
        )

    def test_scott_is_scipys_bandwidth(self):
        from scipy.stats import gaussian_kde

        scipy_bandwidth = np.sqrt(gaussian_kde(self.sizes).covariance[0, 0])
        self.assertAlmostEqual(
            bandwidth_rules.utr_bandwidth(self.sizes, "scott"), scipy_bandwidth, places=12
        )
        self.assertAlmostEqual(
            bandwidth_rules.utr_bandwidth(self.sizes, "scott:0.5"), scipy_bandwidth / 2, places=12
        )

    def test_silverman_is_rs_bw_nrd0(self):
        # bw.nrd0 of R 4, for three sets of sizes; in the second, more than
        # half the molecules share a size, so the IQR is 0 and the SD counts.
        cases = {
            (1, 2, 2, 3, 3, 3, 4, 10, 11, 12): 2.371006401380210,
            (5, 5, 5, 5, 5, 6, 9): 0.227555903753893,
            (*[40] * 30, *[44] * 20, 90, 120): 1.218981272992831,
        }
        for sizes, expected in cases.items():
            with self.subTest(sizes=sizes):
                self.assertAlmostEqual(
                    bandwidth_rules.utr_bandwidth(np.array(sizes), "silverman"), expected, 14
                )

    def test_silverman_is_less_swayed_by_a_second_site(self):
        # The distant site inflates the SD (Scott) more than the IQR.
        self.assertLess(
            bandwidth_rules.utr_bandwidth(self.sizes, "silverman"),
            bandwidth_rules.utr_bandwidth(self.sizes, "scott"),
        )

    def test_rules_narrow_with_more_molecules(self):
        deeper = np.repeat(self.sizes, 32)  # the same spread, 32 times the molecules
        for rule in ("scott", "silverman"):
            with self.subTest(rule=rule):
                ratio = bandwidth_rules.utr_bandwidth(deeper, rule) / bandwidth_rules.utr_bandwidth(
                    self.sizes, rule
                )
                self.assertAlmostEqual(ratio, 0.5, places=2)  # 32^-1/5, with ddof=1

    def test_numbers_are_every_utrs_bandwidth(self):
        self.assertEqual(bandwidth_rules.utr_bandwidth(self.sizes, 7), 7.0)

    def test_3_molecules_or_fewer_have_no_kde(self):
        for bandwidth in (7, "scott"):
            self.assertTrue(np.isnan(bandwidth_rules.utr_bandwidth(np.array([1, 2, 3]), bandwidth)))

    def test_molecules_all_of_one_size_are_one_site(self):
        sizes = np.full(20, 61)
        self.assertEqual(bandwidth_rules.utr_bandwidth(sizes, "silverman"), 0)
        for kernel in ("gaussian", "linear"):
            peaks = kde.call_peaks_using_kernel_density_estimator(sizes, "scott", kernel)
            self.assertListEqual(peaks.tolist(), [61] * 20)

    def test_peaks_with_a_rule(self):
        peaks = kde.kde_caller(bandwidth="Scott:0.5", kernel="linear")(self.sizes)
        self.assertEqual(len(peaks), len(self.sizes))
        main = np.bincount(peaks).argmax()
        self.assertLessEqual(abs(main - 50), 3)

    def test_an_invalid_bandwidth_is_refused(self):
        for bandwidth in (0, -2, "wide"):
            with self.subTest(bandwidth=bandwidth), self.assertRaises(ValueError):
                kde.call_peaks_using_kernel_density_estimator(self.sizes, bandwidth)


class TestAnAddedRule(unittest.TestCase):
    """A rule added by name works wherever bandwidths are read."""

    def setUp(self):
        BANDWIDTH_RULES.register("range", range_rule)
        self.addCleanup(BANDWIDTH_RULES.unregister, "range")

    def test_bandwidths_spell_it_with_a_factor(self):
        self.assertEqual(parse_bandwidth("Range:2"), "range:2")
        self.assertEqual(parse_bandwidth("range:1"), "range")

    def test_each_utrs_bandwidth(self):
        sizes = np.array([10, 20, 30, 50, 110])
        self.assertEqual(bandwidth_rules.utr_bandwidth(sizes, "range"), 10.0)
        self.assertEqual(bandwidth_rules.utr_bandwidth(sizes, "range:0.5"), 5.0)

    def test_peaks_with_it(self):
        # Two sites 30 nt apart: a tenth of the range (3 nt) keeps them apart,
        # five times that joins them.
        sizes = np.array([100] * 10 + [130] * 10)
        apart = kde.call_peaks_using_kernel_density_estimator(sizes, "range")
        joined = kde.call_peaks_using_kernel_density_estimator(sizes, "range:5")
        self.assertEqual(len(np.unique(apart)), 2)
        self.assertEqual(len(np.unique(joined)), 1)

    def test_unknown_rules_are_refused(self):
        BANDWIDTH_RULES.unregister("range")
        self.addCleanup(BANDWIDTH_RULES.register, "range", range_rule)
        with self.assertRaisesRegex(ValueError, "expected a number of nt or a rule"):
            parse_bandwidth("range")


if __name__ == "__main__":
    unittest.main()
