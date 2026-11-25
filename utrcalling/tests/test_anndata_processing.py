# import general purpose tools
import os
import shutil
import tempfile
import unittest
import warnings
from pathlib import Path

import anndata as ad

# import data wrangling tools
import numpy as np
import pandas as pd
import scipy.sparse

from utrcalling.core import anndata_processing as ap
from utrcalling.core import module_log

# make sure file paths resolve correctly for the test case
os.chdir(str(Path(__file__).resolve().parents[0]))
# import internal tools


class TestComputeMeanSizes(unittest.TestCase):
    def setUp(self) -> None:
        # create synthetic count matrix
        count_matrix = np.array(
            [
                [1, 0, 2, 2, 1, 1, 3, 3],
                [4, 4, 4, 0, 1, 0, 1, 3],
                [0, 4, 2, 3, 1, 1, 1, 4],
                [4, 3, 1, 3, 2, 2, 4, 1],
                [0, 2, 0, 0, 0, 1, 0, 2],
            ]
        )

        # create synthetic obs
        obs = pd.DataFrame(
            columns=["celltype", "barcodes", "cool_annotation"],
            data=list(
                zip(
                    ["A", "B", "A", "C", "B"],
                    [
                        "AAAAAAAAAAAAAAAA-1",
                        "AAAAAAAAAAAAAAAG-1",
                        "GAAAAAAAAAAAAAAT-2",
                        "AAAAAAAAAAAAATTT-1",
                        "AAAAAAAAAAAAAAGG-2",
                    ],
                    [2.3, 4.4, 5.0, 2.1, 3.0],
                    strict=True,
                )
            ),
        )
        obs = obs.set_index("barcodes")

        # create synthetic var
        utr_regions = {
            "polyA": [
                "UTR1-1",
                "UTR2-1",
                "UTR2-2",
                "UTR3-1",
                "UTR3-2",
                "UTR3-3",
                "UTR3-4",
                "UTR4-1",
            ],
            "utr_name": [
                "UTR1",
                "UTR2",
                "UTR2",
                "UTR3",
                "UTR3",
                "UTR3",
                "UTR3",
                "UTR4",
            ],
            "utr_size": [100, 100, 200, 100, 200, 300, 400, 100],
            "chrom": [1, 1, 1, 1, 1, 1, 1, 1],
            "start": [10, 100, 100, 500, 500, 500, 500, 900],
            "end": [100, 400, 400, 800, 800, 800, 800, 1200],
            "name": ["UTR1", "UTR2", "UTR2", "UTR3", "UTR3", "UTR3", "UTR3", "UTR4"],
            "score": [0.0] * 8,
            "strand": ["+", "-", "-", "+", "+", "+", "+", "-"],
            "transcript_id": [
                "UTR1_t",
                "UTR2_t",
                "UTR2_t",
                "UTR3_t",
                "UTR3_t",
                "UTR3_t",
                "UTR3_t",
                "UTR4_t",
            ],
            "gene_name": [
                "Gene1",
                "Gene1",
                "Gene1",
                "Gene2",
                "Gene2",
                "Gene2",
                "Gene2",
                "Gene3",
            ],
        }
        var = pd.DataFrame(utr_regions).set_index("polyA")

        self.adata = ad.AnnData(X=count_matrix, obs=obs, var=var)
        self.adata_means = ap.compute_utr_size_means(
            adata=self.adata.copy(),
            utr_name_column="utr_name",
        )

    def test_columns(self):
        "Columns should be the utr_name"
        self.assertListEqual(
            list(self.adata.var.utr_name.unique()), self.adata_means.var_names.tolist()
        )

    def test_shape(self):
        "Shape should be: rows = number of unique barcodes (check obs: 5):"
        "cols = number of UTRs (check var: 4)"
        self.assertTupleEqual(self.adata_means.shape, (5, 4))

    def test_mean(self):
        expected = np.array(
            [
                [100, 200, (200 + 200 + 300 + (400 * 3)) / (2 + 1 + 1 + 3), 100],
                [100, 150, (200 + 400) / 2, 100],
                [
                    np.nan,
                    ((4 * 100) + (2 * 200)) / 6,
                    (300 + 200 + 300 + 400) / (3 + 1 + 1 + 1),
                    100,
                ],
                [
                    100,
                    (300 + 200) / 4,
                    (300 + 400 + 600 + (4 * 400)) / (3 + 2 + 2 + 4),
                    100,
                ],
                [np.nan, 100, 300, 100],
            ]
        )
        np.testing.assert_array_almost_equal(self.adata_means.X, expected)


class TestSiteNames(unittest.TestCase):
    """Sites are named <gene>_<n>-<size>, n numbering each gene's sites."""

    @staticmethod
    def _var(utr_names, utr_sizes, gene_names):
        return pd.DataFrame(
            {"utr_name": utr_names, "utr_size": utr_sizes, "gene_name": gene_names},
            index=[f"site{i}" for i in range(len(utr_names))],
        )

    def test_numbered_by_utr_and_size_whatever_the_row_order(self):
        # Gene A has two UTRs: ENSG1 (sites at 100 and 300) and ENSG1_2 (at 50).
        var = self._var(["ENSG1", "ENSG1_2", "ENSG1"], [300, 50, 100], ["A", "A", "A"])
        self.assertListEqual(list(ap.site_names(var)), ["A_2-300", "A_3-50", "A_1-100"])

    def test_genes_without_a_name_use_the_ensembl_id(self):
        var = self._var(
            ["ENSG1", "ENSG2_2", "ENSG3-x", "ENSG4"], [10, 20, 30, 40], [np.nan, "None", None, "D"]
        )
        self.assertListEqual(
            list(ap.site_names(var)), ["ENSG1_1-10", "ENSG2_1-20", "ENSG3_1-30", "D_1-40"]
        )

    def test_categorical_columns_from_h5ad_files(self):
        var = self._var(["U1", "U1", "U2"], [5, 7, 9], ["A", "A", "B"]).astype(
            {"utr_name": "category", "gene_name": "category"}
        )
        self.assertListEqual(list(ap.site_names(var)), ["A_1-5", "A_2-7", "B_1-9"])

    def test_without_a_gene_name_column(self):
        var = self._var(["ENSG1", "ENSG1"], [5, 7], [None, None]).drop(columns="gene_name")
        self.assertListEqual(list(ap.site_names(var)), ["ENSG1_1-5", "ENSG1_2-7"])

    def test_a_gene_named_like_another_genes_ensembl_id_shares_its_numbers(self):
        # Both resolve to "ENSG1", so they are numbered together and stay unique.
        var = self._var(["ENSG1", "ENSG9"], [5, 5], [np.nan, "ENSG1"])
        self.assertListEqual(list(ap.site_names(var)), ["ENSG1_1-5", "ENSG1_2-5"])


class TestGenerateUtrSiteNamesFromGeneName(unittest.TestCase):
    def setUp(self) -> None:
        adata = ad.read_h5ad("./test_data/anndata_test.h5ad")
        adata_2 = adata.copy()
        adata_2.var_names = [
            "gene3_1-1",
            "gene3_1-2",
            "gene3_2-1",
            "gene4_1-1",
            "gene4_2-1",
            "gene4_2-2",
        ]
        self.adata = ad.concat([adata, adata_2], axis=1, merge="same")
        gene_names = [["AA"] * 3, [np.nan] * 3, ["CC"] * 3, [np.nan] * 3]
        gene_names = [item for gene_name in gene_names for item in gene_name]
        self.adata.var["utr_name"] = self.adata.var.index.str.split("-").str[0]
        self.adata.var["gene_name"] = gene_names
        self.adata_named = ap.generate_utr_site_names_from_gene_name(
            self.adata, utr_name_column="utr_name", gene_name_column="gene_name"
        )

    def test_new_names(self):
        names_new = [
            "AA_1-300",
            "AA_2-399",
            "AA_3-500",
            "ENSMUSG00000000002_1-100",
            "ENSMUSG00000000002_2-201",
            "ENSMUSG00000000002_3-300",
            "CC_1-300",
            "CC_2-399",
            "CC_3-500",
            "gene4_1-100",
            "gene4_2-201",
            "gene4_3-300",
        ]
        self.assertListEqual(self.adata_named.var_names.tolist(), names_new)

    def test_data_not_changed(self):
        np.testing.assert_array_equal(self.adata_named.X, self.adata.X)

    def test_obs_not_changed(self):
        pd.testing.assert_frame_equal(self.adata_named.obs, self.adata.obs)

    def test_var_not_changed(self):
        pd.testing.assert_frame_equal(
            self.adata_named.var[["utr_size", "utr_name", "gene_name"]].reset_index(drop=True),
            self.adata.var[["utr_size", "utr_name", "gene_name"]].reset_index(drop=True),
        )

    def test_utr_site_column(self):
        pd.testing.assert_series_equal(
            self.adata_named.var.UTR_site,
            self.adata.var_names.to_series(),
            check_index=False,
            check_names=False,
        )


# ---------------------------------------------------------------------------
# New test classes for previously untested functions
# ---------------------------------------------------------------------------


def _make_multi_site_adata():
    """Build a small AnnData with 2 UTRs: UTR1 has 3 sites, UTR2 has 1 site."""
    X = np.array(
        [
            [5, 3, 2, 4],
            [1, 1, 1, 2],
            [3, 2, 0, 1],
        ]
    )
    obs = pd.DataFrame({"batch": [1, 1, 2]}, index=["c1", "c2", "c3"])
    var = pd.DataFrame(
        {
            "utr_name": ["UTR1", "UTR1", "UTR1", "UTR2"],
            "utr_size": [100, 200, 300, 150],
            "gene_name": ["GeneA", "GeneA", "GeneA", "GeneB"],
        },
        index=["p1", "p2", "p3", "p4"],
    )
    return ad.AnnData(X=X, obs=obs, var=var)


class TestSelectAlternativeTranscriptionEvents(unittest.TestCase):
    def test_single_site_utrs_removed(self):
        adata = _make_multi_site_adata()
        result = ap.select_alternative_transcription_events(adata)
        # UTR2 has only 1 site → removed; UTR1 has 3 → kept
        self.assertNotIn("UTR2", result.var.utr_name.values)

    def test_multi_site_utrs_kept(self):
        adata = _make_multi_site_adata()
        result = ap.select_alternative_transcription_events(adata)
        self.assertEqual(result.shape[1], 3)  # 3 peaks from UTR1

    def test_obs_preserved(self):
        adata = _make_multi_site_adata()
        result = ap.select_alternative_transcription_events(adata)
        self.assertEqual(result.shape[0], 3)

    def test_no_pandas_future_warning(self):
        """`groupby` must pass `observed=True` so the pandas 2.x default
        deprecation does not surface as a warning to users."""

        adata = _make_multi_site_adata()
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            ap.select_alternative_transcription_events(adata)


class TestSelectUtrsWithMolecules(unittest.TestCase):
    def setUp(self):
        # UTR A: 3 + 6 = 9 molecules; UTR B: 10; UTR C: 2.
        var = pd.DataFrame(
            {"utr_name": ["A", "A", "B", "C"], "utr_size": [10, 20, 30, 40]},
            index=["A_1-10", "A_2-20", "B_1-30", "C_1-40"],
        )
        counts = np.array([[1, 4, 10, 0], [2, 2, 0, 2]], dtype=np.int32)
        self.adata = ad.AnnData(X=counts, var=var)

    def test_utrs_with_enough_molecules_over_all_cells(self):
        for minimum, utrs in ((10, ["B"]), (9, ["A", "A", "B"]), (0, ["A", "A", "B", "C"])):
            with self.subTest(minimum=minimum):
                kept = ap.select_utrs_with_molecules(self.adata, minimum)
                self.assertListEqual(kept.var.utr_name.tolist(), utrs)
                self.assertFalse(kept.is_view)
        self.assertEqual(ap.select_utrs_with_molecules(self.adata, 11).n_vars, 0)

    def test_sparse_counts(self):
        self.adata.X = scipy.sparse.csr_matrix(self.adata.X)
        kept = ap.select_utrs_with_molecules(self.adata, 9)
        self.assertTrue(scipy.sparse.issparse(kept.X))
        self.assertListEqual(list(kept.var_names), ["A_1-10", "A_2-20", "B_1-30"])


class TestCountsToPercentages(unittest.TestCase):
    def test_fractions_sum_to_one(self):
        adata = _make_multi_site_adata()
        adata = adata.copy()
        adata.X = adata.X.astype(float)
        result = ap.counts_to_percentages(adata, utr_name_column="utr_name")
        # For UTR1 (3 columns), each row's values should sum to 1
        utr1_mask = result.var.utr_name == "UTR1"
        utr1_X = result[:, utr1_mask].X
        row_sums = np.sum(utr1_X, axis=1)
        np.testing.assert_array_almost_equal(row_sums, [1.0, 1.0, 1.0])

    def test_single_peak_utr_is_one(self):
        adata = _make_multi_site_adata()
        adata = adata.copy()
        adata.X = adata.X.astype(float)
        result = ap.counts_to_percentages(adata, utr_name_column="utr_name")
        # UTR2 has single column → all values become 1.0
        utr2_mask = result.var.utr_name == "UTR2"
        utr2_X = result[:, utr2_mask].X
        np.testing.assert_array_almost_equal(utr2_X.flatten(), [1.0, 1.0, 1.0])

    def test_zero_row_handled(self):
        """A row with all zeros should not produce NaN/inf."""
        X = np.array([[0, 0], [3, 1]])
        obs = pd.DataFrame(index=["c1", "c2"])
        var = pd.DataFrame(
            {"utr_name": ["UTR1", "UTR1"], "utr_size": [100, 200]},
            index=["p1", "p2"],
        )
        adata = ad.AnnData(X=X.astype(float), obs=obs, var=var)
        result = ap.counts_to_percentages(adata, utr_name_column="utr_name")
        # Zero-sum row should produce zeros, not NaN
        self.assertFalse(np.any(np.isnan(result.X)))

    def test_sparse_input(self):
        adata = _make_multi_site_adata()
        adata = adata.copy()
        adata.X = scipy.sparse.csr_matrix(adata.X.astype(float))
        result = ap.counts_to_percentages(adata, utr_name_column="utr_name")
        utr1_mask = result.var.utr_name == "UTR1"
        utr1_X = result[:, utr1_mask].X
        if scipy.sparse.issparse(utr1_X):
            utr1_X = utr1_X.toarray()
        row_sums = np.sum(utr1_X, axis=1)
        np.testing.assert_array_almost_equal(row_sums, [1.0, 1.0, 1.0])

    def test_int32_sparse_input_does_not_truncate(self):
        """Regression: int32 sparse counts must produce float percentages.

        Previously, view-based assignment into the int32 parent matrix
        silently downcast fractional percentages (0 < x < 1) to 0,
        corrupting any UTR row whose source had multiple peaks per cell.
        """
        X = scipy.sparse.csr_matrix(np.array([[1, 2, 1], [3, 6, 3], [1, 0, 0]], dtype=np.int32))
        var = pd.DataFrame(
            {"utr_name": ["U1", "U1", "U1"], "utr_size": [100, 200, 300]},
            index=["p1", "p2", "p3"],
        )
        adata = ad.AnnData(
            X=X,
            obs=pd.DataFrame(index=["c1", "c2", "c3"]),
            var=var,
        )
        result = ap.counts_to_percentages(adata, utr_name_column="utr_name")
        out = result.X.toarray() if scipy.sparse.issparse(result.X) else result.X
        np.testing.assert_array_almost_equal(
            out,
            [[0.25, 0.5, 0.25], [0.25, 0.5, 0.25], [1.0, 0.0, 0.0]],
        )
        # Output dtype must be float (so fractions are representable).
        self.assertTrue(np.issubdtype(result.X.dtype, np.floating))

    def test_no_future_warnings(self):
        """Vectorized implementation must not trigger anndata/pandas FutureWarnings."""

        adata = _make_multi_site_adata()
        adata.X = scipy.sparse.csr_matrix(adata.X.astype(np.int32))
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            warnings.simplefilter("error", ad.ImplicitModificationWarning)
            ap.counts_to_percentages(adata, utr_name_column="utr_name")

    def test_matches_loop_reference(self):
        """Vectorized output must match a per-UTR loop reference."""
        rng = np.random.RandomState(0)
        n_obs, n_utrs, peaks_per_utr = 20, 8, 4
        n_vars = n_utrs * peaks_per_utr
        X = scipy.sparse.csr_matrix(rng.randint(0, 5, size=(n_obs, n_vars)).astype(np.int32))
        utr_names = np.repeat([f"U{i}" for i in range(n_utrs)], peaks_per_utr)
        var = pd.DataFrame(
            {"utr_name": utr_names, "utr_size": np.tile([100, 200, 300, 400], n_utrs)},
            index=[f"p{i}" for i in range(n_vars)],
        )
        adata = ad.AnnData(
            X=X.copy(),
            obs=pd.DataFrame(index=[f"c{i}" for i in range(n_obs)]),
            var=var,
        )
        result = ap.counts_to_percentages(adata, utr_name_column="utr_name")

        # Reference: scalar per-(obs, utr) loop on a dense float copy.
        ref = X.toarray().astype(np.float64)
        for utr in np.unique(utr_names):
            cols = np.where(utr_names == utr)[0]
            block = ref[:, cols]
            row_sums = block.sum(axis=1, keepdims=True)
            ref[:, cols] = np.divide(
                block,
                row_sums,
                out=np.zeros_like(block),
                where=row_sums != 0,
            )
        out = result.X.toarray() if scipy.sparse.issparse(result.X) else result.X
        np.testing.assert_allclose(out, ref, atol=1e-12)

    def test_non_unique_var_names(self):
        """Duplicate var_names (real-data property) must not corrupt percentages."""
        X = scipy.sparse.csr_matrix(np.array([[1, 1, 2, 2], [0, 1, 0, 0]], dtype=np.int32))
        var = pd.DataFrame(
            {"utr_name": ["U1", "U1", "U2", "U2"], "utr_size": [100, 200, 100, 200]},
            index=["dup", "dup", "dup", "dup"],  # all four var names identical
        )
        adata = ad.AnnData(X=X, obs=pd.DataFrame(index=["c1", "c2"]), var=var)
        result = ap.counts_to_percentages(adata, utr_name_column="utr_name")
        out = result.X.toarray() if scipy.sparse.issparse(result.X) else result.X
        np.testing.assert_array_almost_equal(out, [[0.5, 0.5, 0.5, 0.5], [0.0, 1.0, 0.0, 0.0]])


class TestConcatenateAdata(unittest.TestCase):
    def test_overlapping_vars(self):
        adata1 = _make_multi_site_adata()
        adata2 = _make_multi_site_adata()
        adata2.obs.index = ["c4", "c5", "c6"]
        result = ap.concatenate_adata([adata1, adata2])
        self.assertEqual(result.shape[0], 6)  # 3 + 3 obs
        self.assertEqual(result.shape[1], 4)  # same vars

    def test_non_overlapping_obs(self):
        adata1 = _make_multi_site_adata()
        adata2 = _make_multi_site_adata()
        adata2.obs.index = ["c4", "c5", "c6"]
        result = ap.concatenate_adata([adata1, adata2])
        # All obs should be present
        self.assertEqual(set(result.obs.index), {"c1", "c2", "c3", "c4", "c5", "c6"})

    def test_fills_missing_with_zero(self):
        # adata1 has UTR1 (3 peaks) + UTR2 (1 peak)
        adata1 = _make_multi_site_adata()
        # adata2 has only UTR1 (3 peaks) — different from adata1 due to different var
        X2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        obs2 = pd.DataFrame({"batch": [3, 3, 3]}, index=["c4", "c5", "c6"])
        var2 = pd.DataFrame(
            {
                "utr_name": ["UTR1", "UTR1", "UTR1"],
                "utr_size": [100, 200, 300],
                "gene_name": ["GeneA", "GeneA", "GeneA"],
            },
            index=["p1", "p2", "p3"],
        )
        adata2 = ad.AnnData(X=X2, obs=obs2, var=var2)
        result = ap.concatenate_adata([adata1, adata2])
        # adata2 doesn't have UTR2, so those values should be 0 for c4-c6
        self.assertEqual(result.shape[0], 6)


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2, buffer=True)
    module_log.log_end_run("Finished testing")


def _sites_matrix():
    """Six cells of two samples: UTR A with 3 sites, B with 2, C with 1."""
    counts = np.array(
        [
            [2, 0, 2, 1, 1, 5],
            [0, 0, 4, 0, 0, 1],
            [1, 1, 0, 3, 0, 0],
            [0, 0, 0, 0, 0, 2],
            [3, 1, 0, 0, 4, 0],
            [1, 2, 1, 2, 2, 1],
        ],
        dtype=np.int32,
    )
    var = pd.DataFrame(
        {
            "utr_name": ["A", "A", "A", "B", "B", "C"],
            # Sites out of order: ranks follow sizes, not columns.
            "utr_size": [10, 200, 50, 300, 30, 5],
            "gene_name": ["gA", "gA", "gA", "gB", "gB", "gC"],
            "strand": ["+", "+", "+", "-", "-", "+"],
        },
        index=[f"site{i}" for i in range(6)],
    )
    obs = pd.DataFrame(
        {
            "sample": ["s1", "s1", "s1", "s2", "s2", "s2"],
            "condition": ["ctl", "ctl", "ctl", "trt", "trt", "trt"],
            "pseudotime": [0.1, 0.5, 0.9, 0.2, 0.6, 0.8],
            "phase": ["G1", "S", "G1", "G1", "G2M", "S"],
            "flag": [True, False, True, True, False, True],
        },
        index=[f"cell{i}" for i in range(6)],
    )
    return ad.AnnData(X=scipy.sparse.csr_matrix(counts), obs=obs, var=var)


class TestUtrAnnotations(unittest.TestCase):
    def test_one_row_per_utr_with_the_columns_its_sites_share(self):
        utrs = ap.utr_annotations(_sites_matrix().var)
        self.assertListEqual(list(utrs.index), ["A", "B", "C"])
        self.assertListEqual(list(utrs.columns), ["gene_name", "strand", "sites"])
        self.assertListEqual(utrs["sites"].tolist(), [3, 2, 1])
        self.assertListEqual(utrs["gene_name"].tolist(), ["gA", "gB", "gC"])

    def test_without_shared_columns_only_the_number_of_sites(self):
        var = pd.DataFrame({"utr_name": ["A", "A"], "utr_size": [1, 2]})
        self.assertListEqual(list(ap.utr_annotations(var).columns), ["sites"])


class TestLengthIndexMatrix(unittest.TestCase):
    def setUp(self):
        self.counts = _sites_matrix()
        self.index = ap.length_index(self.counts)

    def test_a_column_per_multi_site_utr(self):
        self.assertListEqual(list(self.index.var_names), ["A", "B"])
        self.assertListEqual(self.index.var["sites"].tolist(), [3, 2])
        pd.testing.assert_index_equal(self.index.obs_names, self.counts.obs_names)

    def test_values_are_the_mean_rank_of_each_cells_molecules(self):
        dense = self.counts.X.toarray()
        # A: sizes 10, 200, 50 give ranks 0, 1, 0.5; B: 300, 30 give 1, 0.
        a = dense[:, :3]
        with np.errstate(invalid="ignore"):
            expected_a = (a @ np.array([0.0, 1.0, 0.5])) / a.sum(axis=1)
        b = dense[:, 3:5]
        with np.errstate(invalid="ignore"):
            expected_b = (b @ np.array([1.0, 0.0])) / b.sum(axis=1)
        np.testing.assert_allclose(self.index.X[:, 0], expected_a, rtol=1e-6)
        np.testing.assert_allclose(self.index.X[:, 1], expected_b, rtol=1e-6)
        # Cell 3 has no molecules in A or B.
        self.assertTrue(np.isnan(self.index.X[3]).all())

    def test_entropy_layer(self):
        entropy = self.index.layers["entropy"]
        # Cell 0, UTR A: shares 0.5, 0, 0.5 of three sites.
        self.assertAlmostEqual(float(entropy[0, 0]), np.log(2) / np.log(3), places=6)
        # Cell 1, UTR A: all at one site.
        self.assertEqual(float(entropy[1, 0]), 0.0)
        self.assertTrue(np.isnan(entropy[1, 1]))

    def test_global_index_is_the_mean_over_measured_utrs(self):
        obs = self.index.obs
        self.assertListEqual(obs["utrs_measured"].tolist(), [2, 1, 2, 0, 2, 2])
        np.testing.assert_allclose(obs["length_index"].iloc[0], np.nanmean(self.index.X[0]))
        np.testing.assert_allclose(
            obs["entropy"].iloc[2], np.nanmean(self.index.layers["entropy"][2])
        )
        self.assertTrue(np.isnan(obs["length_index"].iloc[3]))

    def test_smoothing_averages_the_counts_over_neighbours_first(self):
        graph = scipy.sparse.csr_matrix(np.ones((6, 6)) - np.eye(6))
        self.counts.obsp["knn"] = graph
        smoothed = ap.length_index(self.counts, smooth_graph="knn", smooth_rounds=1)
        dense = self.counts.X.toarray().astype(float)
        mean_of_others = (dense.sum(axis=0) - dense[3]) / 5
        a = mean_of_others[:3]
        self.assertAlmostEqual(float(smoothed.X[3, 0]), a @ [0.0, 1.0, 0.5] / a.sum(), places=6)
        self.assertEqual(smoothed.obs["utrs_measured"].iloc[3], 2)

    def test_no_multi_site_utr_is_refused(self):
        with self.assertRaisesRegex(ValueError, "two or more sites"):
            ap.length_index(self.counts[:, [5]].copy())


class TestSmoothCounts(unittest.TestCase):
    def test_each_round_takes_the_weighted_mean_of_the_neighbours(self):
        counts = np.array([[1.0, 0.0], [3.0, 2.0], [5.0, 4.0]])
        graph = scipy.sparse.csr_matrix([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=float)
        once = ap.smooth_counts(counts, graph, rounds=1).toarray()
        np.testing.assert_allclose(once, [[4.0, 3.0], [1.0, 0.0], [1.0, 0.0]])
        twice = ap.smooth_counts(counts, graph, rounds=2).toarray()
        np.testing.assert_allclose(twice, [[1.0, 0.0], [4.0, 3.0], [4.0, 3.0]])

    def test_cells_without_neighbours_keep_their_counts(self):
        counts = np.array([[1.0], [3.0], [7.0]])
        graph = scipy.sparse.csr_matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=float)
        np.testing.assert_allclose(ap.smooth_counts(counts, graph, 1).toarray().ravel(), [3, 1, 7])

    def test_zero_rounds_and_errors(self):
        counts = np.array([[1.0], [3.0]])
        graph = scipy.sparse.csr_matrix([[0, 1], [1, 0]], dtype=float)
        np.testing.assert_allclose(ap.smooth_counts(counts, graph, 0).toarray(), counts)
        with self.assertRaisesRegex(ValueError, "0 or more"):
            ap.smooth_counts(counts, graph, -1)
        with self.assertRaisesRegex(ValueError, "a row and a column per cell"):
            ap.smooth_counts(counts, scipy.sparse.eye(3), 1)


class TestNeighbourGraph(unittest.TestCase):
    def test_reads_obsp_or_says_what_is_there(self):
        adata = _sites_matrix()
        adata.obsp["connectivities"] = scipy.sparse.csr_matrix(np.eye(6))
        self.assertEqual(ap.neighbour_graph(adata, "connectivities").shape, (6, 6))
        with self.assertRaisesRegex(ValueError, "it has: connectivities"):
            ap.neighbour_graph(adata, "distances")
        with self.assertRaisesRegex(ValueError, "it has: none"):
            ap.neighbour_graph(_sites_matrix(), "connectivities")


class TestPseudobulk(unittest.TestCase):
    def setUp(self):
        self.cells = _sites_matrix()

    def test_sums_the_counts_of_each_samples_cells(self):
        units = ap.pseudobulk(self.cells, by=["sample"], min_cells=1)
        self.assertListEqual(list(units.obs_names), ["s1", "s2"])
        dense = self.cells.X.toarray()
        np.testing.assert_array_equal(units.X.toarray(), [dense[:3].sum(0), dense[3:].sum(0)])
        self.assertTrue(scipy.sparse.issparse(units.X))
        self.assertEqual(units.X.dtype, np.int32)
        self.assertListEqual(units.obs["n_cells"].tolist(), [3, 3])
        pd.testing.assert_frame_equal(units.var, self.cells.var)

    def test_keeps_shared_columns_and_means_of_numbers(self):
        units = ap.pseudobulk(self.cells, by=["sample"], min_cells=1)
        self.assertListEqual(units.obs["condition"].tolist(), ["ctl", "trt"])
        np.testing.assert_allclose(units.obs["pseudotime"], [0.5, (0.2 + 0.6 + 0.8) / 3])
        # Text and booleans that differ within a unit are left out.
        self.assertNotIn("phase", units.obs)
        self.assertNotIn("flag", units.obs)

    def test_categorical_columns_stay_categorical(self):
        self.cells.obs["condition"] = pd.Categorical(self.cells.obs["condition"])
        units = ap.pseudobulk(self.cells, by=["sample"], min_cells=1)
        self.assertIsInstance(units.obs["condition"].dtype, pd.CategoricalDtype)

    def test_bins_cut_a_numeric_column_into_equal_frequency_bins(self):
        units = ap.pseudobulk(self.cells, by=["sample"], bins={"pseudotime": 2}, min_cells=1)
        self.assertListEqual(list(units.obs_names), ["s1|1", "s1|2", "s2|1", "s2|2"])
        self.assertListEqual(units.obs["pseudotime_bin"].tolist(), [1, 2, 1, 2])
        self.assertListEqual(units.obs["n_cells"].tolist(), [2, 1, 1, 2])
        np.testing.assert_allclose(units.obs["pseudotime"], [0.3, 0.9, 0.2, 0.7])

    def test_small_units_and_cells_without_a_value_are_left_out(self):
        self.cells.obs.loc["cell0", "sample"] = np.nan
        units = ap.pseudobulk(self.cells, by=["sample"], min_cells=3)
        self.assertListEqual(list(units.obs_names), ["s2"])

    def test_repeated_values_make_fewer_bins(self):
        self.cells.obs["pseudotime"] = [0.0, 0.0, 0.0, 0.0, 1.0, 2.0]
        with self.assertLogs("utrcalling", level="WARNING") as logs:
            units = ap.pseudobulk(self.cells, by=["sample"], bins={"pseudotime": 3}, min_cells=1)
        self.assertIn("made 2", logs.output[0])
        self.assertListEqual(sorted(units.obs["pseudotime_bin"].unique()), [1, 2])

    def test_errors(self):
        with self.assertRaisesRegex(ValueError, "No column donor"):
            ap.pseudobulk(self.cells, by=["donor"])
        with self.assertRaisesRegex(ValueError, "phase is not numeric"):
            ap.pseudobulk(self.cells, by=["sample"], bins={"phase": 2})
        with self.assertRaisesRegex(ValueError, "flag is not numeric"):
            ap.pseudobulk(self.cells, by=["sample"], bins={"flag": 2})
        with self.assertRaisesRegex(ValueError, "1 or more bins"):
            ap.pseudobulk(self.cells, by=["sample"], bins={"pseudotime": 0})
        with self.assertRaisesRegex(ValueError, "the largest has 3"):
            ap.pseudobulk(self.cells, by=["sample"], min_cells=4)
