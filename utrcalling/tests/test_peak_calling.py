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


if __name__ == "__main__":
    unittest.main()
