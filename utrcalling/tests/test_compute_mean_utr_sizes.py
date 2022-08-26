import os
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import anndata as ad

# make sure file paths resolve correctly for the test case
os.chdir(str(Path(__file__).resolve().parents[0]))

from utrcalling.tools import compute_mean_utr_sizes as tool
from utrcalling.core import module_log


class TestCaseBase(unittest.TestCase):
    def assertIsFile(self, path):
        if not Path(path).resolve().is_file():
            raise AssertionError(f"File does not exist: {str(path)}")


class TestComputeMeanSizes(TestCaseBase):
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
        self.adata_path = "test_compute_mean_sizes.h5ad"
        tool.compute_mean_utr_sizes(
            self.adata,
            self.adata_path,
            utr_name_column="utr_name",
        )

    def tearDown(self) -> None:
        os.remove(self.adata_path)

    def test_columns(self):
        "Columns should be the utr_name"
        adata = ad.read_h5ad(self.adata_path)
        self.assertListEqual(list(self.adata.var.utr_name.unique()),
                             adata.var_names.tolist())

    def test_shape(self):
        "Shape should be: rows = number of unique barcodes (check obs: 5): "
        "cols = number of UTRs (check var: 4)"
        adata = ad.read_h5ad(self.adata_path)
        self.assertTupleEqual(adata.shape, (5, 4))

    def test_mean(self):
        expected = np.array([
            [100, 200, (200 + 200 + 300 + (400 * 3)) / (2 + 1 + 1 + 3), 100],
            [100, 150, (200 + 400) / 2, 100],
            [np.nan, ((4 * 100) + (2 * 200)) / 6,
             (300 + 200 + 300 + 400) / (3 + 1 + 1 + 1), 100],
            [100, (300 + 200) / 4, (300 + 400 + 600 + (4 * 400)) / (3 + 2 + 2 + 4), 100],
            [np.nan, 100, 300, 100]
        ])
        adata_X = ad.read_h5ad(self.adata_path).X
        np.testing.assert_array_almost_equal(adata_X, expected)

    def test_file_creation(self):
        self.assertIsFile(self.adata_path)


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2, buffer=True)
    module_log.log_end_run("Finished testing")
