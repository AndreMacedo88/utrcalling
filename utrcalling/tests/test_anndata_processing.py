# import general purpose tools
import os
from pathlib import Path
import unittest
# import data wrangling tools
import numpy as np
import pandas as pd
import anndata as ad
# make sure file paths resolve correctly for the test case
os.chdir(str(Path(__file__).resolve().parents[0]))
# import internal tools
from utrcalling.core import module_log
from utrcalling.core import anndata_processing as ap


class TestComputeMeanSizes(unittest.TestCase):
    def setUp(self) -> None:
        # create synthetic count matrix
        count_matrix = np.array([
            [1, 0, 2, 2, 1, 1, 3, 3],
            [4, 4, 4, 0, 1, 0, 1, 3],
            [0, 4, 2, 3, 1, 1, 1, 4],
            [4, 3, 1, 3, 2, 2, 4, 1],
            [0, 2, 0, 0, 0, 1, 0, 2]])

        # create synthetic obs
        obs = pd.DataFrame(columns=["celltype", "barcodes",
                                    "cool_annotation"],
                           data=list(
            zip(["A", "B", "A", "C", "B"],
                ["AAAAAAAAAAAAAAAA-1", "AAAAAAAAAAAAAAAG-1", "GAAAAAAAAAAAAAAT-2",
                 "AAAAAAAAAAAAATTT-1", "AAAAAAAAAAAAAAGG-2"],
                [2.3, 4.4, 5.0, 2.1, 3.0])))
        obs = obs.set_index("barcodes")

        # create synthetic var
        utr_regions = {
            "polyA": [
                "UTR1-1", "UTR2-1", "UTR2-2", "UTR3-1",
                "UTR3-2", "UTR3-3", "UTR3-4", "UTR4-1"
            ],
            "utr_name": ["UTR1", "UTR2", "UTR2", "UTR3", "UTR3", "UTR3", "UTR3", "UTR4"],
            "utr_size": [100, 100, 200, 100, 200, 300, 400, 100],
            "chrom": [1, 1, 1, 1, 1, 1, 1, 1],
            "start": [10, 100, 100, 500, 500, 500, 500, 900],
            "end": [100, 400, 400, 800, 800, 800, 800, 1200],
            "name": ["UTR1", "UTR2", "UTR2", "UTR3", "UTR3", "UTR3", "UTR3", "UTR4"],
            "score": [0.0] * 8,
            "strand": ["+", "-", "-", "+", "+", "+", "+", "-"],
            "transcript_id": ["UTR1_t", "UTR2_t", "UTR2_t", "UTR3_t", "UTR3_t", "UTR3_t",
                              "UTR3_t", "UTR4_t"],
            "gene_name": ["Gene1", "Gene1", "Gene1", "Gene2", "Gene2", "Gene2", "Gene2",
                          "Gene3"]
        }
        var = pd.DataFrame(utr_regions).set_index("polyA")

        self.adata = ad.AnnData(X=count_matrix, obs=obs, var=var,
                                dtype=count_matrix.dtype)
        self.adata_means = ap.compute_utr_size_means(
            adata=self.adata.copy(),
            utr_name_column="utr_name",
        )

    def test_columns(self):
        "Columns should be the utr_name"
        self.assertListEqual(list(self.adata.var.utr_name.unique()),
                             self.adata_means.var_names.tolist())

    def test_shape(self):
        "Shape should be: rows = number of unique barcodes (check obs: 5): "
        "cols = number of UTRs (check var: 4)"
        self.assertTupleEqual(self.adata_means.shape, (5, 4))

    def test_mean(self):
        expected = np.array([
            [100, 200, (200 + 200 + 300 + (400 * 3)) / (2 + 1 + 1 + 3), 100],
            [100, 150, (200 + 400) / 2, 100],
            [np.nan, ((4 * 100) + (2 * 200)) / 6,
             (300 + 200 + 300 + 400) / (3 + 1 + 1 + 1), 100],
            [100, (300 + 200) / 4, (300 + 400 + 600 + (4 * 400)) / (3 + 2 + 2 + 4), 100],
            [np.nan, 100, 300, 100]
        ])
        np.testing.assert_array_almost_equal(self.adata_means.X, expected)


class TestGenerateUtrSiteNamesFromGeneName(unittest.TestCase):
    def setUp(self) -> None:
        adata = ad.read_h5ad("./test_data/anndata_test.h5ad")
        adata_2 = adata.copy()
        adata_2.var_names = [
            "gene3_1-1", "gene3_1-2", "gene3_2-1", "gene4_1-1", "gene4_2-1", "gene4_2-2"
        ]
        self.adata = ad.concat([adata, adata_2], axis=1, merge="same")
        gene_names = [["AA"] * 3,
                      [np.nan] * 3,
                      ["CC"] * 3,
                      [np.nan] * 3]
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
            self.adata_named.var[["utr_size", "utr_name",
                                  "gene_name"]].reset_index(drop=True),
            self.adata.var[["utr_size", "utr_name",
                            "gene_name"]].reset_index(drop=True)
        )

    def test_utr_site_column(self):
        pd.testing.assert_series_equal(
            self.adata_named.var.UTR_site,
            self.adata.var_names.to_series(),
            check_index=False, check_names=False
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
        train, test = ap.split_train_test_data(
            self.adata,
            train_fraction=0.66,
            seed=1)
        self.assertEqual(train.shape[0], test.shape[0] * 2)

    def test_keep_balance(self):
        train, test = ap.split_train_test_data(
            self.adata,
            train_fraction=0.66,
            seed=1)

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


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2, buffer=True)
    module_log.log_end_run("Finished testing")
