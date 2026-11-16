"""Tests for utrcalling.core.protocols.peak_calling: KDE consensus peaks."""

import unittest

import numpy as np

from utrcalling.core.protocols import peak_calling as pc


class TestGetCaller(unittest.TestCase):
    def test_kde_caller_uses_the_bandwidth(self):
        sizes = np.array([100] * 10 + [130] * 10)
        narrow = pc.get_caller("KDE", bandwidth=2)(sizes)
        wide = pc.get_caller("kde", bandwidth=50)(sizes)
        self.assertEqual(len(np.unique(narrow)), 2)
        self.assertEqual(len(np.unique(wide)), 1)

    def test_unknown_method_raises(self):
        with self.assertRaises(NotImplementedError):
            pc.get_caller("gmm", bandwidth=12)


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


if __name__ == "__main__":
    unittest.main()
