"""Tests for utrcalling.core.peak_calling.kde: KDE consensus peaks."""

import unittest

import numpy as np
from sklearn.neighbors import KernelDensity

from utrcalling.core.peak_calling import kde as pc
from utrcalling.core.peak_calling.settings import KERNELS


class TestCallPeaksUsingKDE(unittest.TestCase):
    def test_no_molecules_give_no_peaks(self):
        peaks = pc.call_peaks_using_kernel_density_estimator(np.array([], dtype=np.int32))
        self.assertEqual(peaks.dtype, np.int32)
        self.assertEqual(len(peaks), 0)

    def test_few_umis_returns_median(self):
        """<=3 UMIs should return median for all."""
        sizes = np.array([100, 200, 150])
        peaks = pc.call_peaks_using_kernel_density_estimator(sizes, bandwidth=1)
        self.assertEqual(len(peaks), 3)
        self.assertTrue(np.all(peaks == round(np.median(sizes))))

    def test_single_peak(self):
        """Tightly clustered sizes → all assigned same peak."""
        rng = np.random.RandomState(42)
        sizes = rng.normal(100, 0.5, 50).astype(int)
        peaks = pc.call_peaks_using_kernel_density_estimator(sizes, bandwidth=1)
        self.assertEqual(len(np.unique(peaks)), 1)

    def test_two_separated_peaks(self):
        """Two clearly separated clusters → two distinct peaks."""
        sizes = np.concatenate(
            [
                np.full(30, 50),
                np.full(30, 200),
            ]
        )
        peaks = pc.call_peaks_using_kernel_density_estimator(sizes, bandwidth=1)
        self.assertEqual(len(np.unique(peaks)), 2)

    def test_dtype_is_int32(self):
        """Output should use np.int32 dtype."""
        sizes = np.array([100] * 10 + [200] * 10)
        peaks = pc.call_peaks_using_kernel_density_estimator(sizes, bandwidth=1)
        self.assertEqual(peaks.dtype, np.int32)

    def test_vector_and_column_input_give_the_same_peaks(self):
        sizes = np.array([100] * 10 + [200] * 10 + [205] * 3)
        np.testing.assert_array_equal(
            pc.call_peaks_using_kernel_density_estimator(sizes, bandwidth=3),
            pc.call_peaks_using_kernel_density_estimator(sizes.reshape(-1, 1), bandwidth=3),
        )

    def test_each_molecule_gets_its_own_peak_in_input_order(self):
        sizes = np.array([200, 100, 201, 99, 100, 200])
        peaks = pc.call_peaks_using_kernel_density_estimator(sizes, bandwidth=2)
        self.assertListEqual(peaks.tolist(), [200, 100, 200, 100, 100, 200])

    def test_a_peak_at_size_zero_keeps_its_molecules(self):
        # 5' protocols keep UTR size 0 (transcripts starting at the ATG). Peaks
        # used to be tracked with 0 = "not assigned yet", so these molecules
        # were all moved to the UTR's last peak.
        sizes = np.array([0] * 50 + [200] * 50)
        peaks = pc.call_peaks_using_kernel_density_estimator(sizes, bandwidth=12)
        self.assertListEqual(peaks.tolist(), [0] * 50 + [200] * 50)


def _histogram(counts):
    """Molecules with the given count per size (from 10), and the log of their counts.

    A kernel narrower than the 1 nt between sizes gives this density, with the
    flat stretches that equal counts make. Sizes 0-9 and those after the last
    count are the padding, without molecules (log density minus infinity).
    """
    sizes = np.repeat(np.arange(10, 10 + len(counts)), counts)
    window = np.arange(0, 20 + len(counts))
    density = np.zeros(len(window))
    density[10 : 10 + len(counts)] = counts
    with np.errstate(divide="ignore"):
        return sizes, window, np.log(density)


class TestAssignToPeaks(unittest.TestCase):
    """The density's local maxima take the molecules up to the lowest point between them."""

    def _peak_of_each_size(self, counts):
        sizes, window, log_density = _histogram(counts)
        peaks = pc._assign_to_peaks(sizes, window, log_density)
        self.assertEqual(peaks.dtype, np.int32)
        return dict(zip(sizes.tolist(), peaks.tolist(), strict=True))

    def test_two_peaks_split_at_the_lowest_point_between_them(self):
        # Sizes 10-14 rise to 12 and fall; 15 is the valley; 16-18 a second site.
        peaks = self._peak_of_each_size([1, 3, 9, 3, 2, 1, 4, 8, 2])
        self.assertEqual(
            peaks, {10: 12, 11: 12, 12: 12, 13: 12, 14: 12, 15: 17, 16: 17, 17: 17, 18: 17}
        )

    def test_a_flat_top_is_one_peak_at_its_middle(self):
        # Equal counts at 11-13: one site, not three.
        self.assertEqual(set(self._peak_of_each_size([2, 7, 7, 7, 2]).values()), {12})
        # Two equal sizes: the lower of the two middles, as the Gaussian ties.
        self.assertEqual(set(self._peak_of_each_size([2, 7, 7, 2]).values()), {11})

    def test_a_flat_step_on_a_falling_slope_is_not_a_peak(self):
        # Ranbp3's 5' UTR in Cap-seq data (sizes 15-21 there): 58 molecules at
        # each of two sizes, between 75 and 32, are the flank of the site
        # before. Calling the step a peak made a site out of a flank.
        peaks = self._peak_of_each_size([171, 107, 118, 75, 58, 58, 32])
        self.assertEqual(peaks, {10: 10, 11: 12, 12: 12, 13: 12, 14: 12, 15: 12, 16: 12})

    def test_a_flat_step_on_a_rising_slope_is_not_a_peak(self):
        # Supt5's 5' UTR (sizes 292-296 there): two equal densities below the top.
        self.assertEqual(set(self._peak_of_each_size([3, 5, 5, 9, 4]).values()), {13})

    def test_peaks_apart_split_where_the_density_is_zero(self):
        # Kernels with a bounded support (linear, tophat, ...) give log
        # densities of minus infinity between sites. Their molecules stay apart.
        peaks = self._peak_of_each_size([1, 2, 1, 0, 0, 0, 1, 3, 1])
        self.assertEqual(peaks, {10: 11, 11: 11, 12: 11, 16: 17, 17: 17, 18: 17})

    def test_a_flat_valley_splits_at_its_middle(self):
        peaks = self._peak_of_each_size([5, 9, 5, 1, 1, 1, 5, 9, 5])
        self.assertEqual((peaks[13], peaks[14], peaks[15]), (11, 17, 17))

    def test_a_molecule_at_the_split_goes_to_the_peak_after_it(self):
        # The valley at 13 is a single size: its molecules go up to the next peak.
        peaks = self._peak_of_each_size([1, 6, 2, 1, 5, 1])
        self.assertEqual((peaks[12], peaks[13]), (11, 14))

    def test_beyond_the_window_the_density_counts_as_lower(self):
        # A density that only falls has its peak at the window's first size.
        sizes = np.array([0, 1, 2])
        falling = pc._assign_to_peaks(sizes, np.arange(3), np.array([3.0, 2, 1]))
        rising = pc._assign_to_peaks(sizes, np.arange(3), np.array([1.0, 2, 3]))
        self.assertListEqual(falling.tolist(), [0, 0, 0])
        self.assertListEqual(rising.tolist(), [2, 2, 2])


# The kernels of scikit-learn's KernelDensity, unnormalised, by distance in bandwidths.
KERNEL_SHAPES = {
    "tophat": lambda u: (u < 1).astype(float),
    "linear": lambda u: np.where(u < 1, 1 - u, 0),
    "epanechnikov": lambda u: np.where(u < 1, 1 - u**2, 0),
    "cosine": lambda u: np.where(u < 1, np.cos(np.pi * u / 2), 0),
    "exponential": lambda u: np.exp(-u),
}


class TestKernels(unittest.TestCase):
    """Every kernel of scikit-learn's KernelDensity calls peaks, each its own way."""

    def test_every_kernel_gives_each_molecule_a_peak_among_its_utrs_sizes(self):
        # A 5' UTR: start sites at 30, 34 and 80, and molecules scattered over it.
        rng = np.random.default_rng(1)
        sizes = np.concatenate(
            [np.full(60, 30), np.full(40, 34), np.full(50, 80), rng.integers(0, 120, 30)]
        )
        order = np.argsort(sizes, kind="stable")
        for kernel in KERNELS:
            for bandwidth in (0.5, 2, 8):
                with self.subTest(kernel=kernel, bandwidth=bandwidth):
                    peaks = pc.call_peaks_using_kernel_density_estimator(sizes, bandwidth, kernel)
                    self.assertEqual(len(peaks), len(sizes))
                    self.assertTrue(np.all((peaks >= sizes.min()) & (peaks <= sizes.max())))
                    # A longer UTR never goes to the peak of a shorter one.
                    self.assertTrue(np.all(np.diff(peaks[order]) >= 0))
                    # The site at 80 stays within a bandwidth of where it is.
                    self.assertLessEqual(abs(peaks[100] - 80), bandwidth)

    def test_the_linear_kernel_keeps_two_close_start_sites_apart(self):
        # Two start sites 4 nt apart. A Gaussian of bandwidth 3 joins them at
        # 102, where no molecule starts; a linear kernel of the same bandwidth
        # (a triangle 3 nt wide on each side) keeps them where they are.
        sizes = np.array([100] * 20 + [104] * 20)
        gaussian = pc.kde_caller(bandwidth=3)(sizes)
        linear = pc.kde_caller(bandwidth=3, kernel="linear")(sizes)
        self.assertEqual(set(gaussian.tolist()), {102})
        self.assertListEqual(linear.tolist(), [100] * 20 + [104] * 20)

    def test_a_flat_top_keeps_all_its_molecules_together(self):
        # Molecules spread evenly over 10-30: the linear-kernel density is flat
        # from 11 to 29, up to rounding. Before 5.0.0 the flat top made four
        # peaks, one of them at 33, beyond every molecule.
        sizes = np.repeat(np.arange(10, 31), 5)
        peaks = pc.call_peaks_using_kernel_density_estimator(sizes, 2, "linear")
        self.assertEqual(set(peaks.tolist()), {20})

    def test_an_unknown_kernel_is_refused(self):
        with self.assertRaisesRegex(ValueError, "Unknown kernel 'box': expected one of gaussian"):
            pc.kde_caller(bandwidth=12, kernel="box")
        with self.assertRaisesRegex(ValueError, "Unknown kernel 'box'"):
            pc.call_peaks_using_kernel_density_estimator(np.arange(10), 2, "box")

    def test_a_bandwidth_wider_than_the_utr_joins_its_sites(self):
        # The sizes span 21 nt with their padding. Before the fix, the kernels
        # with an end made a density longer than that, and peak calling failed.
        sizes = np.array([100] * 10 + [104] * 10)
        # Not the exponential kernel: its cusps keep every site a peak, at any
        # bandwidth (between two sizes, a sum of exponentials has no maximum).
        for kernel in set(KERNELS) - {"exponential"}:
            with self.subTest(kernel=kernel):
                peaks = pc.call_peaks_using_kernel_density_estimator(sizes, 60, kernel)
                self.assertEqual(set(peaks.tolist()), {102})

    def test_densities_are_the_kde_up_to_a_constant(self):
        # Summed here so that flat stretches come out exactly flat, but the KDE
        # all the same: the sum of each molecule's kernel, as scikit-learn
        # defines the kernels.
        rng = np.random.default_rng(3)
        sizes = np.concatenate([rng.integers(20, 60, 200), np.full(80, 45), np.full(30, 120)])
        window = np.arange(5, 140)
        for kernel in KERNELS:
            # 200 is wider than the window: every molecule reaches every size.
            for bandwidth in (0.6, 1, 2.5, 7, 200):
                with self.subTest(kernel=kernel, bandwidth=bandwidth):
                    density = pc._density(sizes, window, bandwidth, kernel)
                    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(
                        sizes.reshape(-1, 1)
                    )
                    log_kde = kde.score_samples(window.reshape(-1, 1))
                    if kernel == "gaussian":  # scikit-learn's own, as before
                        np.testing.assert_array_equal(density, log_kde)
                        continue
                    distance = np.abs(window[:, None] - sizes[None, :]) / bandwidth
                    expected = KERNEL_SHAPES[kernel](distance).sum(axis=1)
                    # The exponential sum stops at 40 bandwidths (a weight of e^-40),
                    # which only the deepest valleys, without molecules, notice.
                    checked = expected > np.exp(-15) * expected.max()
                    ratio = density[checked] / expected[checked]
                    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-9)
                    np.testing.assert_array_equal(density[expected == 0], 0)
                    if kernel != "exponential":  # scikit-learn's tree sums approximate tails
                        ratio = density[expected > 0] / np.exp(log_kde[expected > 0])
                        np.testing.assert_allclose(ratio, ratio[0], rtol=1e-9)

    def test_flat_stretches_come_out_exactly_flat(self):
        # Equal counts at sizes 40-49 between two sites: the linear density is
        # flat in the middle of them. scikit-learn's sums differ there in the
        # last digits, and its peaks then split the stretch.
        sizes = np.concatenate([np.full(30, 20), np.repeat(np.arange(40, 50), 4), np.full(30, 70)])
        window = np.arange(10, 80)
        for kernel in ("linear", "tophat", "epanechnikov"):
            density = pc._density(sizes, window, 2, kernel)
            with self.subTest(kernel=kernel):
                self.assertEqual(len(set(density[(window >= 41) & (window <= 48)])), 1)

    def test_mirrored_densities_come_out_equal(self):
        # Two sites of 10 molecules 6 nt apart: the density between them
        # mirrors itself, so its middle is one flat top or valley, not a peak
        # made by rounding.
        sizes = np.array([100] * 10 + [106] * 10)
        window = np.arange(90, 117)
        for kernel in ("cosine", "exponential"):
            density = pc._density(sizes, window, 4, kernel)
            with self.subTest(kernel=kernel):
                np.testing.assert_array_equal(density, density[::-1])


if __name__ == "__main__":
    unittest.main()
