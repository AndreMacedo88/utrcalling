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
        obs.set_index("barcodes")

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
        print(count_matrix, obs, var, sep="\n\n")

        self.data = ad.AnnData(X=count_matrix, )

    def test_index(self):
        "index should be the utr_name"
        pass

    def test_shape(self):
        "shape should be the same as utr_regions except for dropping the polyA column"
        pass

    def test_mean(self):
        pass


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2, buffer=False)
    module_log.log_end_run("Finished testing")
