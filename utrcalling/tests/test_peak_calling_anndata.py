"""Tests for utrcalling.core.peak_calling_anndata."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse

from utrcalling.core import module_log
from utrcalling.core import peak_calling_anndata as pca

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
        # Used to fail with an UnboundLocalError deep in the per-UTR code.
        for call_peaks in (pca.peak_call_anndata, pca.peak_call_anndata_parallel):
            with (
                self.subTest(caller=call_peaks.__name__),
                self.assertRaisesRegex(NotImplementedError, "'gmm'"),
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


class TestScoreBandwidths(unittest.TestCase):
    """score_bandwidths: estimate_best_bandwidth's scores, with the sites each bandwidth makes."""

    def test_scores_and_sites_per_bandwidth(self):
        tmpdir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmpdir)
        genome = tmpdir / "genome.fa"
        genome.write_text(">chr1\n" + "C" * 119 + "AATAAA" + "C" * 175 + "\n")
        # Two UTRs, each with sizes 40-42 and 90-92, on the same coordinates.
        adata = _make_adata(
            count_matrix=[[2, 5, 2, 3, 3, 3] * 2] * 4,
            utr_names=["UTR0"] * 6 + ["UTR1"] * 6,
            utr_sizes=[40, 41, 42, 90, 91, 92] * 2,
            n_obs=4,
        )
        bandwidths = [1, 60]
        table = pca.score_bandwidths(adata, genome=str(genome), bandwidths=bandwidths, n_jobs=1)

        self.assertEqual(table.index.name, "bandwidth")
        self.assertListEqual(table.index.tolist(), bandwidths)
        expected = pca.estimate_best_bandwidth(
            adata, genome=str(genome), hamming=1, bandwidths=bandwidths, n_jobs=1
        )
        np.testing.assert_allclose(table.score.to_numpy(), expected)
        sites = [
            pca.peak_call_anndata_parallel(adata, "kde", bandwidth=bw, n_jobs=1).n_vars
            for bw in bandwidths
        ]
        self.assertListEqual(table.sites.tolist(), sites)
        self.assertListEqual(table.sites_per_utr.tolist(), [s / 2 for s in sites])
        # A narrow bandwidth keeps the two groups of sizes of each UTR apart.
        self.assertListEqual(sites, [4, 2])


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


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2, buffer=True)
    module_log.log_end_run("Finished testing")
