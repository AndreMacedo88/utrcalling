import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from utrcalling.core import storage


class TestStorageBase(unittest.TestCase):
    """Base class that creates and cleans up a temp directory."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)


class TestStoreReadMappingsSingle(TestStorageBase):

    def test_creates_parquet_file(self):
        df = pd.DataFrame({
            "utr_name": pd.Categorical(["UTR1", "UTR2"]),
            "observation_id": pd.Categorical(["CELL1", "CELL2"]),
            "utr_size": np.array([100, 200], dtype=np.int32),
        })
        storage.store_read_mappings_single(df, directory=self.tmpdir)
        expected_path = os.path.join(self.tmpdir, "UTR_sizes_counts.parquet")
        self.assertTrue(os.path.exists(expected_path))
        loaded = pd.read_parquet(expected_path)
        self.assertEqual(len(loaded), 2)


class TestStoreReadMappingsMulti(TestStorageBase):

    def test_creates_per_sample_parquets(self):
        dfs = [
            pd.DataFrame({
                "utr_name": pd.Categorical(["UTR1"]),
                "observation_id": pd.Categorical(["CELL1"]),
                "utr_size": np.array([100], dtype=np.int32),
            }),
            pd.DataFrame({
                "utr_name": pd.Categorical(["UTR2"]),
                "observation_id": pd.Categorical(["CELL2"]),
                "utr_size": np.array([200], dtype=np.int32),
            }),
        ]
        storage.store_read_mappings_multi(dfs, directory=self.tmpdir)
        self.assertTrue(os.path.exists(
            os.path.join(self.tmpdir, "UTR_sizes_counts_1.parquet")))
        self.assertTrue(os.path.exists(
            os.path.join(self.tmpdir, "UTR_sizes_counts_2.parquet")))


class TestStoreMergedCounts(TestStorageBase):

    def test_creates_merged_parquet(self):
        df = pd.DataFrame({
            "utr_name": pd.Categorical(["UTR1"]),
            "observation_id": pd.Categorical(["CELL1"]),
            "utr_size": np.array([100], dtype=np.int32),
        })
        storage.store_merged_counts(df, merge_method="intersection", directory=self.tmpdir)
        expected = os.path.join(self.tmpdir, "UTR_sizes_counts_merged_intersection.parquet")
        self.assertTrue(os.path.exists(expected))


class TestStorePeakCallingResults(TestStorageBase):

    def test_creates_all_output_files(self):
        matrix = csr_matrix(np.array([[1, 0], [0, 2]], dtype=np.int32))
        var = pd.DataFrame({"utr_size": [100, 200]}, index=["UTR1-100", "UTR2-200"])
        obs = pd.DataFrame({"batch": ["A", "B"]}, index=["CELL1", "CELL2"])
        ref = pd.DataFrame({
            "chrom": ["1", "1"], "start": [100, 200], "end": [200, 300],
        }, index=pd.Index(["UTR1", "UTR2"], name="utr_name"))

        storage.store_peak_calling_results(
            matrix, var, obs, ref,
            merge_method="union", directory=self.tmpdir,
        )
        self.assertTrue(os.path.exists(
            os.path.join(self.tmpdir, "peaks_count_matrix_union.npz")))
        self.assertTrue(os.path.exists(
            os.path.join(self.tmpdir, "peaks_var_union.parquet")))
        self.assertTrue(os.path.exists(
            os.path.join(self.tmpdir, "peaks_obs_union.parquet")))
        self.assertTrue(os.path.exists(
            os.path.join(self.tmpdir, "UTR_reference_subset_union.parquet")))


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2)
