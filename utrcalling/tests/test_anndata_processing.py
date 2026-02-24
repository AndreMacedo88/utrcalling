# import general purpose tools
from utrcalling.core import anndata_processing as ap
from utrcalling.core import module_log
import os
from pathlib import Path
import tempfile
import unittest

# import data wrangling tools
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse

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

        self.adata = ad.AnnData(
            X=count_matrix, obs=obs, var=var, dtype=count_matrix.dtype
        )
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
            "AA_site1",
            "AA_site2",
            "AA_site3",
            "ENSMUSG00000000002_site1",
            "ENSMUSG00000000002_site2",
            "ENSMUSG00000000002_site3",
            "CC_UTR1_site1",
            "CC_UTR1_site2",
            "CC_UTR2_site1",
            "gene4_UTR1_site1",
            "gene4_UTR2_site1",
            "gene4_UTR2_site2",
        ]
        self.assertListEqual(self.adata_named.var_names.tolist(), names_new)

    def test_data_not_changed(self):
        np.testing.assert_array_equal(self.adata_named.X, self.adata.X)

    def test_obs_not_changed(self):
        pd.testing.assert_frame_equal(self.adata_named.obs, self.adata.obs)

    def test_var_not_changed(self):
        pd.testing.assert_frame_equal(
            self.adata_named.var[["utr_size", "utr_name", "gene_name"]].reset_index(
                drop=True
            ),
            self.adata.var[["utr_size", "utr_name", "gene_name"]].reset_index(
                drop=True
            ),
        )

    def test_utr_site_column(self):
        pd.testing.assert_series_equal(
            self.adata_named.var.UTR_site,
            self.adata.var_names.to_series(),
            check_index=False,
            check_names=False,
        )


class TestSplitTrainTestData(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        adata = ad.read_h5ad("./test_data/anndata_test.h5ad")
        adata.X = adata.X * np.array([1, 1, 60, 1, 100])[:, np.newaxis]
        adata.obs["batch_new"] = [2021, 2021, 2021, 2022, 2022]
        cls.adata = ad.concat([adata, adata, adata], merge="same")
        cls.adata.obs_names_make_unique()

    def test_split_fraction(self):
        train, test = ap.split_train_test_data(self.adata, train_fraction=0.66, seed=1)
        self.assertEqual(train.shape[0], test.shape[0] * 2)

    def test_keep_balance(self):
        train, test = ap.split_train_test_data(self.adata, train_fraction=0.66, seed=1)

        orig_batch_counts = self.adata.obs.batch_new.value_counts().to_list()
        orig_prop = orig_batch_counts[0] / orig_batch_counts[1]

        train_batch_counts = train.obs.batch_new.value_counts().to_list()
        train_prop = train_batch_counts[0] / train_batch_counts[1]

        test_batch_counts = test.obs.batch_new.value_counts().to_list()
        test_prop = test_batch_counts[0] / test_batch_counts[1]

        self.assertTrue(train_prop == test_prop == orig_prop)


class TestSplitTrainTestDataUmiCol(TestSplitTrainTestData):
    def setUp(self) -> None:
        self.adata.obs["umi_test_bin"] = [1, 1, 2, 1, 2] * 3


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
    return ad.AnnData(X=X, obs=obs, var=var, dtype=X.dtype)


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


class TestSubsetAdata(unittest.TestCase):
    def setUp(self):
        self.adata = _make_multi_site_adata()
        self.tmpdir = tempfile.mkdtemp()

    def _write_csv(self, index_values, filename):
        path = os.path.join(self.tmpdir, filename)
        pd.DataFrame(index=index_values, data={"val": range(len(index_values))}).to_csv(
            path
        )
        return path

    def test_var_subset(self):
        csv_path = self._write_csv(["p1", "p3"], "var_keep.csv")
        result = ap.subset_adata(self.adata, [csv_path], ["var"])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape[1], 2)

    def test_obs_subset(self):
        csv_path = self._write_csv(["c1", "c3"], "obs_keep.csv")
        result = ap.subset_adata(self.adata, [csv_path], ["obs"])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape[0], 2)

    def test_invalid_type_raises_value_error(self):
        csv_path = self._write_csv(["c1"], "dummy.csv")
        with self.assertRaises(ValueError):
            ap.subset_adata(self.adata, [csv_path], ["invalid"])


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
        adata2 = ad.AnnData(X=X2, obs=obs2, var=var2, dtype=X2.dtype)
        result = ap.concatenate_adata([adata1, adata2])
        # adata2 doesn't have UTR2, so those values should be 0 for c4-c6
        self.assertEqual(result.shape[0], 6)


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2, buffer=True)
    module_log.log_end_run("Finished testing")
