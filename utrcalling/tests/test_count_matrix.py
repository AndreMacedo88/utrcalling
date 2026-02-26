from test_data import synthetic_data
from utrcalling.core import (
    barcode_filtering,
    count_matrix,
    protocol,
    utr_mapping,
    utr_reference,
    module_log,
    process_mapped_reads,
)

import os
from pathlib import Path
import unittest

import pandas as pd
import numpy as np
from numpy import testing as npt
import copy

# make sure file paths resolve correctly for the test case
os.chdir(str(Path(__file__).resolve().parents[0]))


class TestPeakCalling(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.utr_regions = pd.read_parquet("./test_data/test_UTR_reference.parquet")
        intervals = utr_mapping.generate_utr_intervals(utr_regions=cls.utr_regions)
        alignment_1 = "./test_data/test_UTRoldNSCgenome_bwt2_102_sort.bam"
        alignment_2 = "./test_data/test_UTRyoungNSCgenome_bwt2_102_sort.bam"
        cls.annotations = pd.read_csv("./test_data/vivo_anno.csv")
        cls.protocol_ = protocol.create(
            platform="10x",
            version=2,
            annotations=cls.annotations,
            merge_method="intersection",
        )
        cls.protocol_union = protocol.create(
            platform="10x",
            version=2,
            annotations=cls.annotations,
            merge_method="union",
        )
        cls.mappings = [
            utr_mapping.map_reads_to_utr(
                bam=alignment, intervals=intervals, protocol=cls.protocol_, read="first",
                batch_idx=str(i + 1)
            )
            for i, alignment in enumerate([alignment_1, alignment_2])
        ]


class TestGeneratePeakCounts(TestPeakCalling):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.merged = process_mapped_reads.merge_samples_utr_sizes(
            utr_sizes=copy.deepcopy(cls.mappings)
        )
        cls.merged = barcode_filtering.filter_barcodes(
            cls.merged, whitelist=cls.protocol_.barcode_whitelist
        )


class TestMatrix(TestGeneratePeakCounts):
    def setUp(self):
        self.matrix, _, _ = count_matrix.generate_counts_and_anno(
            utr_sizes=self.merged,
            protocol=self.protocol_,
            size_transform=self.protocol_.peak_calling,
        )

    def test_length(self):
        self.assertTupleEqual(self.matrix.shape, (670, 99))

    def test_matrix_values_count(self):
        non_zeros_count = np.count_nonzero(self.matrix.toarray())
        self.assertEqual(non_zeros_count, 755)

    def test_matrix_maximum(self):
        self.assertEqual(self.matrix.data.max(), 3)


class TestVar(TestGeneratePeakCounts):
    def setUp(self):
        _, self.var, _ = count_matrix.generate_counts_and_anno(
            utr_sizes=self.merged,
            protocol=self.protocol_,
            size_transform=self.protocol_.peak_calling,
        )

    def test_length(self):
        self.assertTupleEqual(self.var.shape, (99, 1))

    def test_vars_values_count(self):
        non_zeros_count = np.count_nonzero(self.var)
        self.assertEqual(non_zeros_count, 99)


class TestObs(TestGeneratePeakCounts):
    def setUp(self):
        _, _, self.obs = count_matrix.generate_counts_and_anno(
            utr_sizes=self.merged,
            protocol=self.protocol_,
            size_transform=self.protocol_.peak_calling,
        )

    def test_length(self):
        self.assertTupleEqual(self.obs.shape, (670, 4))

    def test_obs_values_count(self):
        non_zeros_count = np.count_nonzero(self.obs)
        self.assertEqual(non_zeros_count, 2680)


class TestGeneratePeakCountsSynthetic(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        data_filtered = synthetic_data.MappedDataThreePrimeTenXFilteredBarcodes()
        cls.data = data_filtered._data_filtered
        anno = data_filtered._anno
        cls.protocol_ = protocol.create(
            platform="10x",
            version=2,
            annotations=anno,
            merge_method="intersection",
        )


class TestMatrixSynthetic(TestGeneratePeakCountsSynthetic):
    def setUp(self) -> None:
        self.matrix, _, _ = count_matrix.generate_counts_and_anno(
            utr_sizes=self.data,
            protocol=self.protocol_,
            size_transform=self.protocol_.peak_calling,
        )

    def test_length(self):
        self.assertTupleEqual(self.matrix.shape, (5, 6))

    def test_matrix_values_count(self):
        non_zeros_count = np.count_nonzero(self.matrix.toarray())
        self.assertEqual(non_zeros_count, 9)

    def test_matrix_maximum(self):
        self.assertEqual(self.matrix.data.max(), 2)

    def test_matrix_composition(self):
        reality = np.array(
            [
                [0, 0, 2, 0, 0, 0],
                [1, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
                [0, 1, 1, 0, 1, 0],
            ]
        )
        npt.assert_allclose(self.matrix.toarray(), reality)


class TestVarSynthetic(TestGeneratePeakCountsSynthetic):
    def setUp(self):
        _, self.var, _ = count_matrix.generate_counts_and_anno(
            utr_sizes=self.data,
            protocol=self.protocol_,
            size_transform=self.protocol_.peak_calling,
        )

    def test_length(self):
        self.assertTupleEqual(self.var.shape, (6, 1))

    def test_vars_values_count(self):
        non_zeros_count = np.count_nonzero(self.var)
        self.assertEqual(non_zeros_count, 6)

    def test_utr_names(self):
        lengths_names = self.var.index.str.split("-").str[1].astype(int).to_list()
        self.assertListEqual(lengths_names, self.var.utr_size.to_list())


class TestObsSynthetic(TestGeneratePeakCountsSynthetic):
    def setUp(self):
        _, _, self.obs = count_matrix.generate_counts_and_anno(
            utr_sizes=self.data,
            protocol=self.protocol_,
            size_transform=self.protocol_.peak_calling,
        )

    def test_length(self):
        self.assertTupleEqual(self.obs.shape, (5, 2))

    def test_obs_values_count(self):
        non_zeros_count = np.count_nonzero(self.obs)
        self.assertEqual(non_zeros_count, 10)


class TestGenerateCountMatrixAnndata(TestPeakCalling):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        merged = process_mapped_reads.merge_samples_utr_sizes(cls.mappings)
        subsetted = utr_reference.subset_reference_data(
            utr_sizes=merged,
            utr_reference_regions=cls.utr_regions,
        )
        filtered = barcode_filtering.filter_barcodes(
            utr_sizes=merged,
            whitelist=cls.protocol_.barcode_whitelist,
        )
        cls.matrix, var, cls.obs = count_matrix.generate_counts_and_anno(
            utr_sizes=filtered,
            protocol=cls.protocol_,
            size_transform=cls.protocol_.peak_calling,
        )
        cls.var = count_matrix.add_to_peaks_var(var, subsetted)


class TestAnnData(TestGenerateCountMatrixAnndata):
    def setUp(self) -> None:
        self.matrix_anndata = count_matrix.generate_count_matrix_anndata(
            self.matrix, self.var, self.obs
        )

    def test_matrix_values_count(self):
        non_zeros_count = np.count_nonzero(self.matrix_anndata.X.toarray())
        self.assertEqual(non_zeros_count, 755)

    def test_vars_values_count(self):
        non_zeros_count = np.count_nonzero(self.matrix_anndata.var)
        self.assertEqual(non_zeros_count, 99 * 9)

    def test_obs_values_count(self):
        non_zeros_count = np.count_nonzero(self.matrix_anndata.obs)
        self.assertEqual(non_zeros_count, 670 * 4)


class TestCountMatrixEdgeCases(unittest.TestCase):
    """Edge case tests for generate_counts_and_anno()."""

    @classmethod
    def setUpClass(cls):
        cls.protocol_ = protocol.create(
            platform="10x",
            version=2,
            annotations=synthetic_data.ANNOTATION_THREE_PRIME_TENX,
        )

    def test_empty_dataframe(self):
        """Empty input should produce an empty sparse matrix and empty var."""
        empty_df = pd.DataFrame({
            "utr_name": pd.Categorical([]),
            "observation_id": pd.Categorical([]),
            "utr_size": np.array([], dtype=np.int32),
        })
        matrix, var, obs = count_matrix.generate_counts_and_anno(
            utr_sizes=empty_df, protocol=self.protocol_,
        )
        self.assertEqual(matrix.shape, (0, 0))
        self.assertEqual(len(var), 0)

    def test_single_observation(self):
        """Single observation should produce a 1-row matrix."""
        single_obs = pd.DataFrame({
            "utr_name": pd.Categorical(["UTR1", "UTR1"]),
            "observation_id": pd.Categorical([
                "AAAAAAAAAAAAAAAA-1", "AAAAAAAAAAAAAAAA-1",
            ]),
            "utr_size": np.array([100, 200], dtype=np.int32),
        })
        matrix, var, obs = count_matrix.generate_counts_and_anno(
            utr_sizes=single_obs, protocol=self.protocol_,
        )
        self.assertEqual(matrix.shape[0], 1)
        self.assertGreater(matrix.shape[1], 0)

    def test_single_utr(self):
        """Single UTR with multiple observations should produce peaks from that UTR only."""
        single_utr = pd.DataFrame({
            "utr_name": pd.Categorical(["UTR1"] * 3),
            "observation_id": pd.Categorical([
                "AAAAAAAAAAAAAAAA-1",
                "AAAAAAAAAAAAAAAG-1",
                "GAAAAAAAAAAAAAAT-2",
            ]),
            "utr_size": np.array([100, 100, 200], dtype=np.int32),
        })
        matrix, var, obs = count_matrix.generate_counts_and_anno(
            utr_sizes=single_utr, protocol=self.protocol_,
        )
        self.assertEqual(matrix.shape[0], 3)
        # All peak names should start with "UTR1-"
        for name in var.index:
            self.assertTrue(name.startswith("UTR1-"), f"Unexpected peak name: {name}")


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2, buffer=True)
    module_log.log_end_run("Finished testing")
