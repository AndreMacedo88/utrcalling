"""Tests for utrcalling.core.peak_calling_anndata."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse

from utrcalling.core import module_log, provenance
from utrcalling.core import peak_calling_anndata as pca
from utrcalling.core.peak_calling import bandwidth_rules

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

    def test_no_pandas_future_warning(self):
        """`groupby` must pass `observed=True` so the pandas 2.x default
        deprecation does not surface as a warning to users."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            pca.remove_peaks_below_count(self.adata, threshold=0.1, minimum_counts=10)


def _eugenias_filter(adata, threshold, minimum_counts, group_by):
    """The grouped filter of Eugenia Guerrero Aruffo's thesis code, as she wrote it.

    Her remove_peaks_below_count (projects/tss_anno/scripts/functions), for var
    sorted by UTR name, as her call-peaks script sorted it.
    """
    utr_names = adata.var.loc[:, "utr_name"].values
    group_names = adata.obs[group_by].values
    counts = adata.to_df()
    utrs_thresholds = counts.groupby(group_names, observed=False).sum().T.groupby(utr_names).sum()
    utrs_thresholds = utrs_thresholds.apply(
        lambda x: x.apply(lambda y: max(y * threshold, minimum_counts))
    )
    utr_site_counts = counts.groupby(group_names, observed=False).sum()
    nsites_per_utr = adata.var["utr_name"].groupby(adata.var["utr_name"], observed=False).count()
    new_cols = np.repeat(nsites_per_utr.index.to_numpy(), nsites_per_utr.values)
    mask_group = utrs_thresholds.T.loc[:, new_cols]
    mask_group.columns = utr_site_counts.columns
    return (
        np.greater_equal(utr_site_counts.to_numpy(), mask_group).sum(axis=0).astype(bool).to_numpy()
    )


class TestRemovePeaksBelowCountByGroup(unittest.TestCase):
    """With group_by, a site stays if it passes the threshold in one group of cells."""

    def setUp(self):
        # One UTR, two sites, two treatments. Site B is 2.9% of all molecules
        # (30 of 1020), but 18% of the IFN-treated cells' (20 of 110).
        self.adata = _make_adata(
            count_matrix=[[450, 5], [450, 5], [45, 10], [45, 10]],
            utr_names=["U", "U"],
            utr_sizes=[60, 96],
            n_obs=4,
        )
        self.adata.obs["Treatment"] = ["ctrl", "ctrl", "IFN", "IFN"]

    def _kept(self, adata=None, **settings):
        adata = self.adata if adata is None else adata
        return pca.remove_peaks_below_count(adata, **settings).var.utr_size.tolist()

    def test_a_site_a_treatment_raises_stays(self):
        self.assertListEqual(self._kept(threshold=0.1, minimum_counts=10), [60])
        self.assertListEqual(
            self._kept(threshold=0.1, minimum_counts=10, group_by="Treatment"), [60, 96]
        )

    def test_the_minimum_count_holds_in_each_group(self):
        # 20 molecules in the IFN cells, but at least 25 needed.
        self.assertListEqual(
            self._kept(threshold=0.1, minimum_counts=25, group_by="Treatment"), [60]
        )

    def test_sparse_counts_give_the_same(self):
        adata = self.adata.copy()
        adata.X = scipy.sparse.csr_matrix(adata.X)
        self.assertListEqual(self._kept(adata, threshold=0.1, group_by="Treatment"), [60, 96])

    def test_cells_without_a_group_count_in_none(self):
        adata = self.adata.copy()
        adata.obs["Treatment"] = ["ctrl", "ctrl", None, None]
        self.assertListEqual(self._kept(adata, threshold=0.1, group_by="Treatment"), [60])
        adata.obs["Treatment"] = None
        self.assertListEqual(self._kept(adata, threshold=0.1, group_by="Treatment"), [])

    def test_an_unknown_column_is_refused(self):
        with self.assertRaisesRegex(ValueError, "No column 'Genotype' in the cell annotations"):
            pca.remove_peaks_below_count(self.adata, group_by="Genotype")

    def test_the_filter_of_eugenias_thesis(self):
        rng = np.random.default_rng(7)
        counts = rng.poisson(rng.gamma(0.5, 8, size=40), size=(12, 40))
        adata = _make_adata(
            count_matrix=counts,
            utr_names=sorted([f"UTR{i}" for i in range(8)] * 5),
            utr_sizes=list(range(40)),
            n_obs=12,
        )
        adata.obs["Treatment"] = ["ctrl", "30 mins", "2 hours", "16 hours"] * 3
        for threshold, minimum in ((0.1, 10), (0.05, 5), (0.2, 1)):
            kept = pca.remove_peaks_below_count(adata, threshold, minimum, group_by="Treatment")
            expected = _eugenias_filter(adata, threshold, minimum, "Treatment")
            with self.subTest(threshold=threshold, minimum=minimum):
                self.assertListEqual(list(kept.var_names), list(adata.var_names[expected]))


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
        for call_peaks in (pca.peak_call_anndata, pca.peak_call_anndata_parallel):
            with self.subTest(caller=call_peaks.__name__):
                self.assertIs(call_peaks(self.adata, method="kde", bandwidth=0), self.adata)

    def test_unknown_method_raises(self):
        # Used to fail with an UnboundLocalError deep in the per-UTR code. Since
        # 5.0.0 methods are looked up by name, as every named method is, and an
        # unknown name is a ValueError that lists the known ones.
        for call_peaks in (pca.peak_call_anndata, pca.peak_call_anndata_parallel):
            with (
                self.subTest(caller=call_peaks.__name__),
                self.assertRaisesRegex(
                    ValueError, "Unknown peak caller 'gmm': expected one of kde"
                ),
            ):
                call_peaks(self.adata, method="gmm", bandwidth=1)

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

    def test_returns_actual_not_view(self):
        """Returned AnnData must be an actual object, not a view, so that
        downstream `.var` mutations on disk write don't trigger
        ImplicitModificationWarning per column."""
        result = pca.peak_call_anndata(self.adata, method="kde", bandwidth=1)
        self.assertFalse(result.is_view)


# ---------------------------------------------------------------------------
# _calculate_umi_distance_to_pas
# ---------------------------------------------------------------------------


class TestCalculateUmiDistanceToPas(unittest.TestCase):
    """The Fig. 1B definition: utr_size - hexamer end, to the nearest hexamer."""

    def test_single_pas(self):
        sizes = pd.Series([100, 150, 200], index=["a", "b", "c"])
        result = pca._calculate_umi_distance_to_pas(sizes, pas_end=[120])
        expected = pd.Series([-20, 30, 80], index=["a", "b", "c"])
        pd.testing.assert_series_equal(result, expected)

    def test_multiple_pas_closest_selected(self):
        # UMI a: 15 nt after the hexamer at 100, 85 nt before the one at 200.
        # UMI b: 115 nt after the first, 15 nt after the second.
        sizes = pd.Series([115, 215], index=["a", "b"])
        result = pca._calculate_umi_distance_to_pas(sizes, pas_end=[100, 200])
        self.assertListEqual(result.tolist(), [15, 15])

    def test_nearest_hexamer_wins_over_one_10_to_30_nt_upstream(self):
        # 18 nt after the hexamer at 100 but 2 nt before the one at 120: the
        # nearest counts. The package used to prefer distances in [10, 30],
        # which inflates the share of molecules found in that window.
        sizes = pd.Series([118], index=["a"])
        result = pca._calculate_umi_distance_to_pas(sizes, pas_end=[100, 120])
        self.assertListEqual(result.tolist(), [-2])

    def test_equally_near_hexamers_take_the_first(self):
        sizes = pd.Series([110], index=["a"])
        result = pca._calculate_umi_distance_to_pas(sizes, pas_end=[100, 120])
        self.assertListEqual(result.tolist(), [10])


# ---------------------------------------------------------------------------
# _peak_call_single_utr
# ---------------------------------------------------------------------------


class TestPeakCallingDoesNotDependOnSiteOrder(unittest.TestCase):
    """Each peak gets the counts of its own sites, whatever order they come in.

    Both peak callers summed the counts in size order but labelled the peaks in
    the order the sites came in, so unsorted input swapped counts between peaks.
    """

    def setUp(self):
        # One UTR: the sites at 100 and 101 form one peak, the site at 300 another.
        self.adata = _make_adata(
            count_matrix=[[1, 2, 7], [3, 0, 5], [0, 4, 6], [2, 2, 0], [1, 1, 9]],
            utr_names=["UTR1"] * 3,
            utr_sizes=[100, 101, 300],
            gene_names=["GeneA"] * 3,
        )

    def _check(self, call_peaks):
        for order in ([0, 1, 2], [2, 1, 0]):
            with self.subTest(sizes=[self.adata.var.utr_size.iloc[i] for i in order]):
                result = call_peaks(self.adata[:, order].copy())
                self.assertEqual(result.n_vars, 2)
                long_peak = int(np.argmax(result.var.utr_size.to_numpy()))
                self.assertEqual(result.var.utr_size.iloc[long_peak], 300)
                counts = result[self.adata.obs_names].X
                counts = counts.toarray() if scipy.sparse.issparse(counts) else counts
                np.testing.assert_array_equal(counts[:, long_peak], [7, 5, 6, 0, 9])
                np.testing.assert_array_equal(counts[:, 1 - long_peak], [3, 3, 4, 4, 2])

    def test_parallel_caller(self):
        self._check(lambda a: pca.peak_call_anndata_parallel(a, "kde", bandwidth=5, n_jobs=1))

    def test_sequential_caller(self):
        self._check(lambda a: pca.peak_call_anndata(a, "kde", bandwidth=5))


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
        mtx, var_df = pca._peak_call_single_utr(adata, bandwidth=1)
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
        _, var_df = pca._peak_call_single_utr(adata, bandwidth=1)
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
            self.adata,
            method="kde",
            bandwidth=1,
            n_jobs=2,
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
            self.adata,
            method="kde",
            bandwidth=1,
            n_jobs=1,
        )
        out4 = pca.peak_call_anndata_parallel(
            self.adata,
            method="kde",
            bandwidth=1,
            n_jobs=4,
        )
        np.testing.assert_array_equal(out1.X, out4.X)
        pd.testing.assert_frame_equal(out1.var, out4.var)

    def test_sparse_input_matches_dense(self):
        """Sparse X (which exercises the CSC conversion in _extract_per_utr_inputs)
        produces identical output to dense X."""
        dense_out = pca.peak_call_anndata_parallel(
            self.adata,
            method="kde",
            bandwidth=1,
            n_jobs=2,
        )
        sparse_adata = self.adata.copy()
        sparse_adata.X = scipy.sparse.csr_matrix(sparse_adata.X)
        sparse_out = pca.peak_call_anndata_parallel(
            sparse_adata,
            method="kde",
            bandwidth=1,
            n_jobs=2,
        )
        # Sparse output stays sparse; densify to compare values.
        np.testing.assert_array_equal(
            dense_out.X,
            sparse_out.X.toarray() if scipy.sparse.issparse(sparse_out.X) else sparse_out.X,
        )
        pd.testing.assert_frame_equal(dense_out.var, sparse_out.var)

    def test_sparse_input_stays_sparse_end_to_end(self):
        """Sparse input must produce sparse output without ever materializing a
        dense intermediate. Regression guard: a dense intermediate at this scale
        OOMs the parent process (each per-UTR mtx becomes (n_obs, n_peaks) dense,
        and ``np.hstack`` over thousands of UTRs blows up to multiple GB)."""
        sparse_adata = self.adata.copy()
        sparse_adata.X = scipy.sparse.csr_matrix(sparse_adata.X)
        out = pca.peak_call_anndata_parallel(
            sparse_adata,
            method="kde",
            bandwidth=1,
            n_jobs=2,
        )
        self.assertTrue(
            scipy.sparse.issparse(out.X),
            "Sparse input must yield sparse output; got dense, which would "
            "OOM at production scale.",
        )

    def test_returns_actual_not_view(self):
        """Returned AnnData must be an actual object, not a view, so that
        downstream `.var` mutations on disk write don't trigger
        ImplicitModificationWarning per column."""
        out = pca.peak_call_anndata_parallel(
            self.adata,
            method="kde",
            bandwidth=1,
            n_jobs=2,
        )
        self.assertFalse(out.is_view)


class TestPeakCallSingleUtrCompact(unittest.TestCase):
    """Direct tests for the matmul-aggregator implementation in
    ``_peak_call_single_utr_compact`` to lock in correctness independently of
    the surrounding parallel orchestration."""

    def _make_inputs(self, sparse=False):
        # 3 peaks within one UTR: sizes 50, 100, 200; obs counts chosen so KDE
        # produces multiple peak groups deterministically.
        var_sub = pd.DataFrame(
            {
                "utr_name": ["UTR1", "UTR1", "UTR1"],
                "utr_size": [50, 100, 200],
                "gene_name": ["GeneA", "GeneA", "GeneA"],
                "chrom": ["chr1"] * 3,
                "start": [100] * 3,
                "end": [200] * 3,
                "strand": ["+"] * 3,
            },
            index=["peak_a", "peak_b", "peak_c"],
        )
        X = np.array(
            [
                [10, 0, 5],
                [3, 7, 0],
                [0, 4, 8],
                [2, 0, 1],
                [6, 6, 6],
            ],
            dtype=np.int32,
        )
        if sparse:
            X = scipy.sparse.csc_matrix(X)
        return X, var_sub

    def test_dense_input_returns_dense(self):
        X, var_sub = self._make_inputs(sparse=False)
        mtx, var_df = pca._peak_call_single_utr_compact(X, var_sub, 1)
        self.assertFalse(scipy.sparse.issparse(mtx))
        self.assertEqual(mtx.shape[0], 5)
        self.assertEqual(var_df.shape[0], mtx.shape[1])

    def test_sparse_input_returns_sparse(self):
        X, var_sub = self._make_inputs(sparse=True)
        mtx, var_df = pca._peak_call_single_utr_compact(X, var_sub, 1)
        self.assertTrue(scipy.sparse.issparse(mtx))
        self.assertEqual(mtx.shape[0], 5)
        self.assertEqual(var_df.shape[0], mtx.shape[1])

    def test_sparse_and_dense_produce_same_values(self):
        X_dense, var_sub = self._make_inputs(sparse=False)
        X_sparse, _ = self._make_inputs(sparse=True)
        mtx_dense, _ = pca._peak_call_single_utr_compact(X_dense, var_sub, 1)
        mtx_sparse, _ = pca._peak_call_single_utr_compact(X_sparse, var_sub, 1)
        np.testing.assert_array_equal(mtx_dense, mtx_sparse.toarray())

    def test_column_sums_preserved(self):
        """Total UMI count across all peaks must equal total in original X.
        (Aggregator routes every input column to exactly one output column.)"""
        X, var_sub = self._make_inputs(sparse=True)
        mtx, _ = pca._peak_call_single_utr_compact(X, var_sub, 1)
        total_in = X.sum()
        total_out = mtx.sum()
        self.assertEqual(int(total_in), int(total_out))


# ---------------------------------------------------------------------------
# estimate_best_bandwidth
# ---------------------------------------------------------------------------


class TestEstimateBestBandwidthWithGenome(unittest.TestCase):
    """Bandwidth scores come from PAS hexamers found in the genome.

    The scoring path used to call the PAS finder without the genome, so
    estimate_best_bandwidth always raised a TypeError.
    """

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmpdir)
        # One UTR on chr1:100-200 (+ strand, see _make_adata), with UTR sizes
        # around 41 nt.
        self.adata = _make_adata(
            count_matrix=[[2, 5, 2], [1, 6, 1], [2, 4, 3], [0, 5, 2]],
            utr_names=["UTR0"] * 3,
            utr_sizes=[40, 41, 42],
            n_obs=4,
        )

    def _genome(self, chr1: str) -> str:
        path = self.tmpdir / "genome.fa"
        path.write_text(f">chr1\n{chr1}\n")
        return str(path)

    def _scores(self, chr1: str, bandwidths: list[float]) -> list:
        return pca.estimate_best_bandwidth(
            self.adata, genome=self._genome(chr1), hamming=0, bandwidths=bandwidths, n_jobs=1
        )

    def test_hexamer_upstream_of_the_peaks_scores_above_zero(self):
        # The UTR sequence starts at 0-based position 99, so an AATAAA at 119
        # ends 26 nt into the UTR: ~15 nt before the reads end, inside the
        # 10-30 nt window that counts.
        scores = self._scores("C" * 119 + "AATAAA" + "C" * 175, bandwidths=[1, 3])
        self.assertEqual(len(scores), 2)
        for score in scores:
            self.assertGreater(score, 0)

    def test_genome_without_a_hexamer_scores_zero(self):
        self.assertListEqual(self._scores("C" * 300, bandwidths=[1]), [0.0])

    def test_a_single_bandwidth_can_be_given_as_a_number(self):
        hexamer_upstream = "C" * 119 + "AATAAA" + "C" * 175
        self.assertEqual(len(self._scores(hexamer_upstream, bandwidths=3)), 1)

    def test_a_hexamer_away_from_the_molecules_scores_zero(self):
        # AATAAA ending 80 nt into the UTR: the molecules end ~40 nt before it.
        self.assertListEqual(self._scores("C" * 173 + "AATAAA" + "C" * 121, bandwidths=[1]), [0.0])

    def test_distances_are_returned_on_request(self):
        ((score, distances),) = pca.estimate_best_bandwidth(
            self.adata,
            genome=self._genome("C" * 119 + "AATAAA" + "C" * 175),
            hamming=0,
            bandwidths=[1],
            return_distances_to_hex=True,
            n_jobs=1,
        )
        self.assertEqual(len(distances), 33)  # one per molecule
        self.assertEqual(score, np.mean((distances >= 10) & (distances <= 30)))


class TestPasDistanceHamming(unittest.TestCase):
    """The hexamer search allows `hamming` mismatches to AATAAA."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmpdir)
        self.adata = _make_adata(
            count_matrix=[[2, 5, 2], [1, 6, 1], [2, 4, 3], [0, 5, 2]],
            utr_names=["UTR0"] * 3,
            utr_sizes=[40, 41, 42],
            n_obs=4,
        )
        # AATACA, one mismatch from AATAAA, ending 26 nt into the UTR.
        self.genome = self.tmpdir / "genome.fa"
        self.genome.write_text(">chr1\n" + "C" * 119 + "AATACA" + "C" * 175 + "\n")

    def _distances(self, hamming):
        return pca.pas_distance(self.adata, genome=str(self.genome), hamming=hamming, n_jobs=1)

    def test_one_mismatch_is_found_with_hamming_1_only(self):
        self.assertEqual(self._distances(0).size, 0)
        self.assertListEqual(sorted(self._distances(1).tolist()), [14] * 5 + [15] * 20 + [16] * 8)

    def test_negative_hamming_raises(self):
        with self.assertRaisesRegex(ValueError, "Hamming distance of -1"):
            pca._find_pas(self.adata.var, genome=str(self.genome), hamming=-1)


class TestPasDistanceTable(unittest.TestCase):
    """pas_distance_table: each site's molecules and distance to its nearest hexamer."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmpdir)
        # Sites of UTR0 (chr1:100-200, + strand) at UTR sizes 40 and 80, and of
        # UTR1 (chr1:300-400) at 50, with UTR1 in between in .var.
        self.adata = _make_adata(
            count_matrix=[[2, 1, 0], [3, 0, 4], [0, 2, 1]],
            utr_names=["UTR0", "UTR1", "UTR0"],
            utr_sizes=[40, 50, 80],
            n_obs=3,
        )
        self.adata.var.loc["peak_1", ["start", "end"]] = [300, 400]

    def _table(self, chr1, hamming=0, adata=None):
        genome = self.tmpdir / "genome.fa"
        genome.write_text(f">chr1\n{chr1}\n")
        adata = self.adata if adata is None else adata
        return pca.pas_distance_table(adata, genome=str(genome), hamming=hamming, n_jobs=1)

    def _assert_values(self, column, expected, dtype):
        pd.testing.assert_extension_array_equal(
            pd.array(column, dtype=dtype), pd.array(expected, dtype=dtype)
        )

    def test_rows_follow_var_and_utrs_without_a_hexamer_have_no_distance(self):
        # AATAAA ending 26 nt into UTR0 (see TestPasDistanceHamming); none in UTR1.
        table = self._table("C" * 119 + "AATAAA" + "C" * 375)
        self.assertListEqual(list(table.index), ["peak_0", "peak_1", "peak_2"])
        self.assertEqual(table.index.name, "site")
        self.assertListEqual(table.utr_name.tolist(), ["UTR0", "UTR1", "UTR0"])
        self.assertListEqual(table.molecules.tolist(), [5, 3, 5])
        self._assert_values(table.pas_distance, [14, None, 54], "Int64")
        self._assert_values(table.pas_hexamer, ["AATAAA", None, "AATAAA"], "string")
        self.assertEqual(table.pas_distance.dtype, "Int64")
        self.assertEqual(table.pas_hexamer.dtype, "string")

    def test_each_site_takes_its_nearest_hexamer(self):
        # A second hexamer, one mismatch from AATAAA, ends 66 nt into UTR0: 14 nt
        # before the site at 80, which is 54 nt after the first hexamer.
        chr1 = "C" * 119 + "AATAAA" + "C" * 34 + "ATTAAA" + "C" * 335
        table = self._table(chr1, hamming=1)
        self._assert_values(table.pas_distance, [14, None, 14], "Int64")
        self._assert_values(table.pas_hexamer, ["AATAAA", None, "ATTAAA"], "string")

    def test_sparse_counts_give_the_same_table(self):
        sparse = self.adata.copy()
        sparse.X = scipy.sparse.csr_matrix(sparse.X)
        chr1 = "C" * 119 + "AATAAA" + "C" * 375
        pd.testing.assert_frame_equal(self._table(chr1, adata=sparse), self._table(chr1))

    def test_without_gene_names_the_table_has_no_gene_column(self):
        adata = self.adata.copy()
        del adata.var["gene_name"]
        table = self._table("C" * 500, adata=adata)
        self.assertListEqual(
            list(table.columns),
            ["utr_name", "utr_size", "molecules", "pas_hexamer", "pas_distance"],
        )


class TestUtrWithoutMolecules(unittest.TestCase):
    """A UTR without molecules, as in a subset of the cells, has no peaks.

    Peak calling used to crash on it: the median of no sizes is not a number.
    """

    def setUp(self):
        self.adata = _make_adata(
            count_matrix=[[2, 5, 0, 0], [1, 6, 0, 0]],
            utr_names=["UTR0", "UTR0", "UTR1", "UTR1"],
            utr_sizes=[40, 41, 90, 91],
            n_obs=2,
        )

    def test_both_callers_leave_it_out(self):
        parallel = pca.peak_call_anndata_parallel(self.adata, "kde", bandwidth=12, n_jobs=1)
        sequential = pca.peak_call_anndata(self.adata, "kde", bandwidth=12)
        for peaks in (parallel, sequential):
            self.assertListEqual(peaks.var.utr_name.tolist(), ["UTR0"])
            self.assertEqual(peaks.X.sum(), 14)


class TestPeakTable(unittest.TestCase):
    """peak_table: the peaks of a bandwidth, and the spread of the raw sizes in each."""

    def setUp(self):
        # Two UTRs, each with sizes 40-42 (8, 20 and 8 molecules) and 90-92 (12 each).
        self.adata = _make_adata(
            count_matrix=[[2, 5, 2, 3, 3, 3] * 2] * 4,
            utr_names=["UTR0"] * 6 + ["UTR1"] * 6,
            utr_sizes=[40, 41, 42, 90, 91, 92] * 2,
            gene_names=["GeneA"] * 6 + ["GeneB"] * 6,
            n_obs=4,
        )

    def test_peaks_and_their_spread(self):
        table = pca.peak_table(self.adata, bandwidth=1, n_jobs=1)
        self.assertEqual(table.index.name, "site")
        utr0 = table[table.utr_name == "UTR0"]
        self.assertListEqual(utr0.utr_size.tolist(), [41, 91])
        self.assertListEqual(utr0.molecules.tolist(), [36, 36])
        self.assertListEqual(utr0.utr_share.tolist(), [0.5, 0.5])
        self.assertListEqual(utr0.utr_sites.tolist(), [2, 2])
        self.assertListEqual(utr0.size_mean.tolist(), [41.0, 91.0])
        # 8 molecules 1 nt below and 8 above the 41-nt peak; 12 and 12 around 91.
        np.testing.assert_allclose(utr0.size_sd, [np.sqrt(16 / 36), np.sqrt(24 / 36)])
        self.assertListEqual(utr0.size_min.tolist(), [40, 90])
        self.assertListEqual(utr0.size_max.tolist(), [42, 92])
        self.assertListEqual(table.gene_name.unique().tolist(), ["GeneA", "GeneB"])

    def test_the_peaks_are_those_of_call_peaks(self):
        for bandwidth in (1, 60):
            peaks = pca.peak_call_anndata_parallel(self.adata, "kde", bandwidth=bandwidth, n_jobs=1)
            table = pca.peak_table(self.adata, bandwidth=bandwidth, n_jobs=1)
            with self.subTest(bandwidth=bandwidth):
                self.assertListEqual(list(table.index), list(peaks.var_names))
                np.testing.assert_array_equal(
                    table.molecules.to_numpy(), np.asarray(peaks.X.sum(axis=0)).ravel()
                )

    def test_a_wide_bandwidth_joins_the_two_groups_of_sizes(self):
        table = pca.peak_table(self.adata, bandwidth=60, n_jobs=1)
        self.assertListEqual(table.utr_sites.tolist(), [1, 1])
        self.assertListEqual(table.size_min.tolist(), [40, 40])
        self.assertListEqual(table.size_max.tolist(), [92, 92])
        self.assertGreater(table.size_sd.min(), 20)

    def test_sizes_without_molecules_are_left_out(self):
        adata = self.adata.copy()
        adata.X[:, 6:] = 0  # UTR1
        table = pca.peak_table(adata, bandwidth=1, n_jobs=1)
        self.assertListEqual(table.utr_name.unique().tolist(), ["UTR0"])
        adata.X[:] = 0
        empty = pca.peak_table(adata, bandwidth=1, n_jobs=1)
        self.assertEqual(len(empty), 0)
        self.assertListEqual(list(empty.columns), list(table.columns))


class TestPeakStability(unittest.TestCase):
    """peak_table with bootstrap: how often each peak comes back in resamples of its UTR."""

    def setUp(self):
        # UTR0: a strong site at 100 (200 molecules) and a weak one at 140 (6);
        # UTR1: one site of 3 molecules, which take their median (no density).
        self.adata = _make_adata(
            count_matrix=[[100, 3, 2], [100, 3, 1]],
            utr_names=["UTR0", "UTR0", "UTR1"],
            utr_sizes=[100, 140, 30],
            n_obs=2,
        )

    def test_strong_peaks_always_come_back_and_weak_ones_less(self):
        table = pca.peak_table(self.adata, bandwidth=5, n_jobs=1, bootstrap=200, seed=1)
        self.assertEqual(table.columns[-1], "stability")
        strong, weak, median = table.stability
        self.assertEqual(strong, 1.0)
        # The weak site keeps a peak while a resample draws any of its 6
        # molecules: 1 - (200/206)^206, about 0.998. More than 1 nt off it,
        # never: all its molecules are at 140.
        self.assertGreater(weak, 0.98)
        self.assertTrue(np.isnan(median))  # 3 molecules: no density, no stability

    def test_a_peak_of_one_molecule_often_misses(self):
        adata = self.adata.copy()
        adata.X = np.array([[100, 1, 2], [100, 0, 1]])
        table = pca.peak_table(adata, bandwidth=5, n_jobs=1, bootstrap=300, seed=2)
        # Drawn at least once with probability 1 - (200/201)^201, about 0.63.
        self.assertAlmostEqual(table.stability.iloc[1], 1 - (200 / 201) ** 201, delta=0.08)

    def test_the_same_seed_gives_the_same_stabilities(self):
        stabilities = [
            pca.peak_table(self.adata, bandwidth=5, n_jobs=n_jobs, bootstrap=50, seed=3).stability
            for n_jobs in (1, 2, 1)
        ]
        for other in stabilities[1:]:
            pd.testing.assert_series_equal(other, stabilities[0])
        other_seed = pca.peak_table(self.adata, bandwidth=1, n_jobs=1, bootstrap=50, seed=4)
        self.assertEqual(len(other_seed), 3)

    def test_without_bootstrap_no_stability(self):
        table = pca.peak_table(self.adata, bandwidth=5, n_jobs=1)
        self.assertNotIn("stability", table.columns)
        with self.assertRaisesRegex(ValueError, "bootstrap -1: give 0 resamples or more"):
            pca.peak_table(self.adata, bandwidth=5, bootstrap=-1)

    def test_the_tolerance_is_the_utrs_bandwidth(self):
        # A peak that moves by up to its bandwidth in resamples still comes back.
        sizes, counts, peaks = np.array([100, 101, 102, 103]), np.full(4, 30), np.array([101])

        def shifted(molecules):  # a caller whose peak is 3 nt from 101
            return np.full(len(molecules), 104)

        rng = np.random.default_rng(0)
        for tolerance, expected in ((3, 1.0), (2, 0.0)):
            with self.subTest(tolerance=tolerance):
                stability = pca._peak_stability(sizes, counts, peaks, tolerance, shifted, 10, rng)
                self.assertListEqual(stability.tolist(), [expected])


class TestKernelReachesEveryCaller(unittest.TestCase):
    """Each way of calling peaks uses the kernel it is given.

    Two start sites 4 nt apart: a Gaussian of bandwidth 3 joins them at 102, a
    linear kernel of bandwidth 3 keeps them apart.
    """

    def setUp(self):
        self.adata = _make_adata(
            count_matrix=[[5, 5, 3]] * 4,
            utr_names=["UTR0"] * 3,
            utr_sizes=[100, 104, 300],
            gene_names=["GeneA"] * 3,
            n_obs=4,
        )

    def test_peak_calling(self):
        for call_peaks in (pca.peak_call_anndata, pca.peak_call_anndata_parallel):
            with self.subTest(caller=call_peaks.__name__):
                gaussian = call_peaks(self.adata, "kde", bandwidth=3)
                linear = call_peaks(self.adata, "kde", bandwidth=3, kernel="linear")
                self.assertListEqual(gaussian.var.utr_size.tolist(), [102, 300])
                self.assertListEqual(linear.var.utr_size.tolist(), [100, 104, 300])
                np.testing.assert_array_equal(linear.X, self.adata.X)

    def test_peak_table(self):
        table = pca.peak_table(self.adata, bandwidth=3, n_jobs=1, kernel="linear")
        self.assertListEqual(table.utr_size.tolist(), [100, 104, 300])
        self.assertListEqual(table.size_sd.tolist(), [0.0, 0.0, 0.0])

    def test_an_unknown_kernel_is_refused_before_any_work(self):
        callers = {
            "peak_call_anndata": lambda: pca.peak_call_anndata(self.adata, "kde", 3, kernel="box"),
            "parallel": lambda: pca.peak_call_anndata_parallel(self.adata, "kde", 3, kernel="box"),
            "peak_table": lambda: pca.peak_table(self.adata, 3, kernel="box"),
            "score_bandwidths": lambda: pca.score_bandwidths(
                self.adata, genome="missing.fa", bandwidths=[3], kernel="box"
            ),
        }
        for name, call in callers.items():
            with self.subTest(caller=name), self.assertRaisesRegex(ValueError, "'box'"):
                call()


class TestBandwidthRules(unittest.TestCase):
    """A rule sets each UTR's bandwidth, which its sites keep in .var["bandwidth"]."""

    def setUp(self):
        # UTR0: two sites with spread; UTR1: one size only; UTR2: 3 molecules.
        self.adata = _make_adata(
            count_matrix=[[3, 6, 3, 2, 5, 2, 4, 1, 0], [2, 5, 2, 3, 4, 3, 4, 1, 1]],
            utr_names=["UTR0"] * 6 + ["UTR1", "UTR2", "UTR2"],
            utr_sizes=[40, 41, 42, 90, 91, 92, 70, 10, 20],
            n_obs=2,
        )
        sizes = np.repeat([40, 41, 42, 90, 91, 92], [5, 11, 5, 5, 9, 5])
        self.expected = {
            "UTR0": bandwidth_rules.utr_bandwidth(sizes, "scott:0.5"),
            "UTR1": 0.0,  # all 8 molecules at 70
            "UTR2": np.nan,  # 3 molecules take their median
        }

    def test_both_callers_record_each_utrs_bandwidth(self):
        for call_peaks in (pca.peak_call_anndata, pca.peak_call_anndata_parallel):
            with self.subTest(caller=call_peaks.__name__):
                peaks = call_peaks(self.adata, "kde", bandwidth="scott:0.5", kernel="linear")
                expected = peaks.var.utr_name.map(self.expected).to_numpy(np.float64)
                np.testing.assert_array_equal(peaks.var.bandwidth.to_numpy(), expected)
                self.assertListEqual(peaks.var.utr_size.tolist(), [41, 91, 70, 10])
                self.assertEqual(peaks.X.sum(), self.adata.X.sum())

    def test_a_number_adds_no_column(self):
        peaks = pca.peak_call_anndata_parallel(self.adata, "kde", bandwidth="2", n_jobs=1)
        self.assertNotIn("bandwidth", peaks.var)

    def test_the_peak_table_gives_each_utrs_bandwidth(self):
        for bandwidth, expected in (
            (2, {"UTR0": 2.0, "UTR1": 2.0, "UTR2": np.nan}),
            ("scott:0.5", self.expected),
        ):
            table = pca.peak_table(self.adata, bandwidth=bandwidth, n_jobs=1, kernel="linear")
            with self.subTest(bandwidth=bandwidth):
                np.testing.assert_array_equal(
                    table.bandwidth.to_numpy(), table.utr_name.map(expected).to_numpy(np.float64)
                )
        self.assertListEqual(list(table.columns)[4:7], ["utr_share", "utr_sites", "bandwidth"])

    def test_bandwidths_are_checked_before_any_work(self):
        with self.assertRaisesRegex(ValueError, "bandwidth 'wide'"):
            pca.peak_call_anndata_parallel(self.adata, "kde", bandwidth="wide")
        with self.assertRaisesRegex(ValueError, "must be above 0"):
            pca.peak_table(self.adata, bandwidth=0)
        self.assertIs(pca.peak_call_anndata(self.adata, "kde", bandwidth="0"), self.adata)


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2, buffer=True)
    module_log.log_end_run("Finished testing")
