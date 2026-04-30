"""Tests for utrcalling.core.peak_calling_anndata."""
from __future__ import annotations

import os
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse

from utrcalling.core import peak_calling_anndata as pca
from utrcalling.core import module_log

os.chdir(str(Path(__file__).resolve().parents[0]))


def _make_adata(
    count_matrix,
    utr_names,
    utr_sizes,
    gene_names=None,
    n_obs=5,
    sparse=False,
):
    """Build a small synthetic AnnData for testing."""
    obs = pd.DataFrame(
        {"barcode": [f"cell_{i}" for i in range(n_obs)]},
    ).set_index("barcode")

    if gene_names is None:
        gene_names = [f"Gene{i}" for i in range(len(utr_names))]

    var = pd.DataFrame(
        {
            "utr_name": utr_names,
            "utr_size": utr_sizes,
            "gene_name": gene_names,
            "chrom": ["chr1"] * len(utr_names),
            "start": [100] * len(utr_names),
            "end": [200] * len(utr_names),
            "strand": ["+"] * len(utr_names),
        },
    )
    var.index = [f"peak_{i}" for i in range(len(utr_names))]

    X = np.array(count_matrix, dtype=np.int32) if not sparse else count_matrix
    if sparse and not scipy.sparse.issparse(X):
        X = scipy.sparse.csr_matrix(np.array(count_matrix, dtype=np.int32))

    return ad.AnnData(X=X, obs=obs, var=var)


# ---------------------------------------------------------------------------
# _utr_sizes_vector_from_read_counts
# ---------------------------------------------------------------------------


class TestUtrSizesVectorFromReadCounts(unittest.TestCase):
    def test_dense_input(self):
        counts = np.array([[2, 3], [1, 0]])
        sizes = pd.Series([100, 200], index=["A", "B"])
        result = pca._utr_sizes_vector_from_read_counts(counts, sizes)
        # col sums: [3, 3] → repeat [100]*3 + [200]*3
        expected_vals = [100, 100, 100, 200, 200, 200]
        np.testing.assert_array_equal(result.values, expected_vals)

    def test_sparse_input(self):
        counts = scipy.sparse.csr_matrix(np.array([[2, 3], [1, 0]]))
        sizes = pd.Series([100, 200], index=["A", "B"])
        result = pca._utr_sizes_vector_from_read_counts(counts, sizes)
        expected_vals = [100, 100, 100, 200, 200, 200]
        np.testing.assert_array_equal(result.values, expected_vals)

    def test_preserves_index(self):
        counts = np.array([[1, 2]])
        sizes = pd.Series([50, 150], index=["peakA", "peakB"])
        result = pca._utr_sizes_vector_from_read_counts(counts, sizes)
        expected_index = ["peakA", "peakB", "peakB"]
        self.assertEqual(result.index.tolist(), expected_index)


# ---------------------------------------------------------------------------
# _call_peaks_using_kernel_density_estimator
# ---------------------------------------------------------------------------


class TestCallPeaksUsingKDE(unittest.TestCase):
    def test_few_umis_returns_median(self):
        """<=3 UMIs should return median for all."""
        sizes = np.array([100, 200, 150])
        peaks = pca._call_peaks_using_kernel_density_estimator(sizes, bandwidth=1)
        self.assertEqual(len(peaks), 3)
        self.assertTrue(np.all(peaks == round(np.median(sizes))))

    def test_single_peak(self):
        """Tightly clustered sizes → all assigned same peak."""
        rng = np.random.RandomState(42)
        sizes = rng.normal(100, 0.5, 50).astype(int)
        peaks = pca._call_peaks_using_kernel_density_estimator(sizes, bandwidth=1)
        self.assertEqual(len(np.unique(peaks)), 1)

    def test_two_separated_peaks(self):
        """Two clearly separated clusters → two distinct peaks."""
        sizes = np.concatenate([
            np.full(30, 50),
            np.full(30, 200),
        ])
        peaks = pca._call_peaks_using_kernel_density_estimator(sizes, bandwidth=1)
        self.assertEqual(len(np.unique(peaks)), 2)

    def test_dtype_is_int32(self):
        """Output should use np.int32 dtype."""
        sizes = np.array([100] * 10 + [200] * 10)
        peaks = pca._call_peaks_using_kernel_density_estimator(sizes, bandwidth=1)
        self.assertEqual(peaks.dtype, np.int32)


# ---------------------------------------------------------------------------
# remove_peaks_below_count
# ---------------------------------------------------------------------------


class TestRemovePeaksBelowCount(unittest.TestCase):
    def setUp(self):
        # UTR1 has 2 peaks (sizes 100 and 200), UTR2 has 2 peaks (sizes 50 and 150)
        self.adata = _make_adata(
            count_matrix=[
                [10, 1, 10, 1],
                [20, 2, 20, 2],
                [15, 0, 15, 0],
                [10, 1, 10, 1],
                [10, 1, 10, 1],
            ],
            utr_names=["UTR1", "UTR1", "UTR2", "UTR2"],
            utr_sizes=[100, 200, 50, 150],
        )

    def test_peaks_below_threshold_removed(self):
        # col sums: [65, 5, 65, 5]
        # UTR1 total: 70, threshold: max(70*0.1, 10) = 10 → col sums 65 > 10, 5 < 10
        # UTR2 total: 70, threshold: same → 65 > 10, 5 < 10
        result = pca.remove_peaks_below_count(self.adata, threshold=0.1, minimum_counts=10)
        self.assertEqual(result.shape[1], 2)  # only 2 peaks kept

    def test_peaks_above_threshold_kept(self):
        result = pca.remove_peaks_below_count(self.adata, threshold=0.01, minimum_counts=1)
        self.assertEqual(result.shape[1], 4)  # all peaks kept

    def test_obs_preserved(self):
        result = pca.remove_peaks_below_count(self.adata, threshold=0.1, minimum_counts=10)
        self.assertEqual(result.shape[0], 5)

    def test_sparse_input(self):
        self.adata.X = scipy.sparse.csr_matrix(self.adata.X)
        result = pca.remove_peaks_below_count(self.adata, threshold=0.1, minimum_counts=10)
        self.assertEqual(result.shape[1], 2)

    def test_dense_input(self):
        result = pca.remove_peaks_below_count(self.adata, threshold=0.1, minimum_counts=10)
        self.assertFalse(scipy.sparse.issparse(result.X))


# ---------------------------------------------------------------------------
# peak_call_anndata
# ---------------------------------------------------------------------------


class TestPeakCallAnndata(unittest.TestCase):
    def setUp(self):
        # Two UTRs, each with 3 peaks at distinct sizes
        # UTR1: sizes 50, 100, 200  —  UTR2: sizes 60, 120, 300
        self.adata = _make_adata(
            count_matrix=[
                [5, 5, 5, 5, 5, 5],
                [3, 3, 3, 3, 3, 3],
                [4, 4, 4, 4, 4, 4],
                [2, 2, 2, 2, 2, 2],
                [6, 6, 6, 6, 6, 6],
            ],
            utr_names=["UTR1", "UTR1", "UTR1", "UTR2", "UTR2", "UTR2"],
            utr_sizes=[50, 100, 200, 60, 120, 300],
            gene_names=["GeneA", "GeneA", "GeneA", "GeneB", "GeneB", "GeneB"],
        )

    def test_bandwidth_zero_returns_unchanged(self):
        result = pca.peak_call_anndata(self.adata, method="kde", bandwidth=0)
        # Should return the same object
        self.assertIs(result, self.adata)

    def test_output_has_fewer_or_equal_columns(self):
        result = pca.peak_call_anndata(self.adata, method="kde", bandwidth=1)
        self.assertLessEqual(result.shape[1], self.adata.shape[1])

    def test_output_preserves_obs(self):
        result = pca.peak_call_anndata(self.adata, method="kde", bandwidth=1)
        self.assertEqual(result.shape[0], self.adata.shape[0])

    def test_sparse_input_sparse_output(self):
        self.adata.X = scipy.sparse.csr_matrix(self.adata.X)
        result = pca.peak_call_anndata(self.adata, method="kde", bandwidth=1)
        self.assertTrue(scipy.sparse.issparse(result.X))

    def test_min_umis_skips_low_utrs(self):
        # Give UTR1 much more counts so it passes the threshold while UTR2 doesn't
        adata = self.adata.copy()
        adata.X[:, :3] = adata.X[:, :3] * 100  # UTR1 now has 6000 total
        result = pca.peak_call_anndata(adata, method="kde", bandwidth=1, min_umis=100)
        # Only UTR1 peaks should remain (UTR2 skipped)
        utr_names = result.var.utr_name.unique()
        self.assertEqual(len(utr_names), 1)


# ---------------------------------------------------------------------------
# _calculate_umi_distance_to_pas
# ---------------------------------------------------------------------------


class TestCalculateUmiDistanceToPas(unittest.TestCase):
    def test_single_pas_plus_strand(self):
        sizes = pd.Series([100, 150, 200], index=["a", "b", "c"])
        result = pca._calculate_umi_distance_to_pas(sizes, pas_end=[120], seq_strand="+")
        expected = pd.Series([-20, 30, 80], index=["a", "b", "c"])
        pd.testing.assert_series_equal(result, expected)

    def test_single_pas_minus_strand(self):
        sizes = pd.Series([100, 150, 200], index=["a", "b", "c"])
        result = pca._calculate_umi_distance_to_pas(sizes, pas_end=[120], seq_strand="-")
        expected = pd.Series([20, -30, -80], index=["a", "b", "c"])
        pd.testing.assert_series_equal(result, expected)

    def test_multiple_pas_closest_selected(self):
        # Two PAS at positions 100 and 200, + strand
        # UMI at size 115 → distance to PAS1: 15 (in 10-30 range), PAS2: -85 → pick PAS1
        # UMI at size 215 → distance to PAS1: 115, PAS2: 15 (in 10-30 range) → pick PAS2
        sizes = pd.Series([115, 215], index=["a", "b"])
        result = pca._calculate_umi_distance_to_pas(
            sizes, pas_end=[100, 200], seq_strand="+"
        )
        # UMI a: PAS1 distance = 15 (in range), PAS2 = -85 → pick 15
        # UMI b: PAS1 = 115, PAS2 = 15 (in range) → pick 15
        self.assertEqual(result.iloc[0], 15)
        self.assertEqual(result.iloc[1], 15)

    def test_invalid_strand_raises(self):
        sizes = pd.Series([100])
        with self.assertRaises(ValueError):
            pca._calculate_umi_distance_to_pas(sizes, pas_end=[50], seq_strand="?")


# ---------------------------------------------------------------------------
# _peak_call_single_utr
# ---------------------------------------------------------------------------


class TestPeakCallSingleUtr(unittest.TestCase):
    def test_output_shape(self):
        adata = _make_adata(
            count_matrix=[
                [10, 10, 10],
                [5, 5, 5],
                [8, 8, 8],
                [3, 3, 3],
                [7, 7, 7],
            ],
            utr_names=["UTR1", "UTR1", "UTR1"],
            utr_sizes=[50, 100, 200],
            gene_names=["Gene1", "Gene1", "Gene1"],
        )
        mtx, var_df = pca._peak_call_single_utr(adata, method="kde", bandwidth=1)
        # mtx should have same number of rows as obs
        self.assertEqual(mtx.shape[0], 5)
        # var_df should have same number of rows as mtx columns
        self.assertEqual(var_df.shape[0], mtx.shape[1])

    def test_var_has_utr_size(self):
        adata = _make_adata(
            count_matrix=[[10, 10], [5, 5], [8, 8], [3, 3], [7, 7]],
            utr_names=["UTR1", "UTR1"],
            utr_sizes=[50, 200],
            gene_names=["Gene1", "Gene1"],
        )
        _, var_df = pca._peak_call_single_utr(adata, method="kde", bandwidth=1)
        self.assertIn("utr_size", var_df.columns)


class TestPeakCallAnndataParallel(unittest.TestCase):
    """Verify the batched parallel path matches the sequential path."""

    def setUp(self):
        # Build ~10 UTRs each with 3 peaks. Small enough to run fast, large
        # enough to exercise multiple batches when n_jobs > 1.
        n_utrs = 10
        peaks_per_utr = 3
        utr_names = []
        utr_sizes = []
        gene_names = []
        sizes_options = [[50, 100, 200], [60, 120, 250], [70, 140, 280]]
        for i in range(n_utrs):
            base = sizes_options[i % len(sizes_options)]
            utr_names.extend([f"UTR{i}"] * peaks_per_utr)
            utr_sizes.extend(base)
            gene_names.extend([f"Gene{i}"] * peaks_per_utr)

        rng = np.random.RandomState(42)
        n_obs = 6
        n_var = n_utrs * peaks_per_utr
        counts = rng.randint(0, 10, size=(n_obs, n_var)).tolist()

        self.adata = _make_adata(
            count_matrix=counts,
            utr_names=utr_names,
            utr_sizes=utr_sizes,
            gene_names=gene_names,
            n_obs=n_obs,
        )

    def test_parallel_matches_sequential(self):
        seq = pca.peak_call_anndata(self.adata, method="kde", bandwidth=1)
        par = pca.peak_call_anndata_parallel(
            self.adata, method="kde", bandwidth=1, n_jobs=2,
        )

        # Same shape
        self.assertEqual(seq.shape, par.shape)
        # Same X (peak_call_anndata orders UTRs as they appear; parallel sorts
        # batches back to original order, so columns should align)
        np.testing.assert_array_equal(seq.X, par.X)
        # Same var (peak names/sizes derived deterministically per UTR)
        pd.testing.assert_frame_equal(seq.var, par.var)

    def test_n_jobs_invariance(self):
        """Different worker counts should produce identical output."""
        out1 = pca.peak_call_anndata_parallel(
            self.adata, method="kde", bandwidth=1, n_jobs=1,
        )
        out4 = pca.peak_call_anndata_parallel(
            self.adata, method="kde", bandwidth=1, n_jobs=4,
        )
        np.testing.assert_array_equal(out1.X, out4.X)
        pd.testing.assert_frame_equal(out1.var, out4.var)

    def test_sparse_input_matches_dense(self):
        """Sparse X (which exercises the CSC conversion in _extract_per_utr_inputs)
        produces identical output to dense X."""
        dense_out = pca.peak_call_anndata_parallel(
            self.adata, method="kde", bandwidth=1, n_jobs=2,
        )
        sparse_adata = self.adata.copy()
        sparse_adata.X = scipy.sparse.csr_matrix(sparse_adata.X)
        sparse_out = pca.peak_call_anndata_parallel(
            sparse_adata, method="kde", bandwidth=1, n_jobs=2,
        )
        # Sparse output stays sparse; densify to compare values.
        np.testing.assert_array_equal(
            dense_out.X,
            sparse_out.X.toarray() if scipy.sparse.issparse(sparse_out.X) else sparse_out.X,
        )
        pd.testing.assert_frame_equal(dense_out.var, sparse_out.var)


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2, buffer=True)
    module_log.log_end_run("Finished testing")
