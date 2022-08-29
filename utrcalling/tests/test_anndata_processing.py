import os
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import anndata as ad

# make sure file paths resolve correctly for the test case
os.chdir(str(Path(__file__).resolve().parents[0]))

from utrcalling.core import module_log
from utrcalling.core import anndata_processing as ap


class TestSplitTrainTestData(unittest.TestCase):
    def setUp(self) -> None:
        adata = ad.read_h5ad("./test_data/anndata_test.h5ad")
        adata.X = adata.X * np.array([1, 1, 60, 1, 100])[:, np.newaxis]
        adata.obs["batch_new"] = [2021, 2021, 2021, 2022, 2022]
        self.adata = ad.concat([adata, adata, adata], merge="same")
        self.adata.obs_names_make_unique()

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

if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2, buffer=False)
    module_log.log_end_run("Finished testing")
