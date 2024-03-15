from test_data import synthetic_data
from utrcalling.core import (
    peak_calling,
    protocol,
    utr_mapping,
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
                bam=alignment, intervals=intervals, protocol=cls.protocol_, read="first"
            )
            for alignment in [alignment_1, alignment_2]
        ]


class TestGeneratePeakCounts(TestPeakCalling):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.merged = process_mapped_reads.merge_samples_utr_sizes(
            utr_sizes=copy.deepcopy(cls.mappings)
        )
        cls.merged = process_mapped_reads.filter_barcodes(
            cls.merged, whitelist=cls.protocol_.barcode_whitelist
        )


class TestMatrix(TestGeneratePeakCounts):
    def setUp(self):
        self.matrix, _, _ = peak_calling.generate_peak_counts_and_anno(
            utr_sizes=self.merged,
            protocol=self.protocol_,
        )

    def test_length(self):
        self.assertTupleEqual(self.matrix.shape, (670, 99))

    def test_matrix_values_count(self):
        non_zeros_count = np.count_nonzero(self.matrix)
        self.assertEqual(non_zeros_count, 755)

    def test_matrix_maximum(self):
        self.assertEqual(np.max(self.matrix.values), 3)


class TestVar(TestGeneratePeakCounts):
    def setUp(self):
        _, self.var, _ = peak_calling.generate_peak_counts_and_anno(
            utr_sizes=self.merged,
            protocol=self.protocol_,
        )

    def test_length(self):
        self.assertTupleEqual(self.var.shape, (99, 1))

    def test_vars_values_count(self):
        non_zeros_count = np.count_nonzero(self.var)
        self.assertEqual(non_zeros_count, 99)


class TestObs(TestGeneratePeakCounts):
    def setUp(self):
        _, _, self.obs = peak_calling.generate_peak_counts_and_anno(
            utr_sizes=self.merged,
            protocol=self.protocol_,
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
        self.matrix, _, _ = peak_calling.generate_peak_counts_and_anno(
            utr_sizes=self.data,
            protocol=self.protocol_,
        )

    def test_length(self):
        self.assertTupleEqual(self.matrix.shape, (5, 6))

    def test_matrix_values_count(self):
        non_zeros_count = np.count_nonzero(self.matrix)
        self.assertEqual(non_zeros_count, 9)

    def test_matrix_maximum(self):
        self.assertEqual(np.max(self.matrix.values), 2)

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
        npt.assert_allclose(self.matrix.sort_index().to_numpy(), reality)


class TestVarSynthetic(TestGeneratePeakCountsSynthetic):
    def setUp(self):
        _, self.var, _ = peak_calling.generate_peak_counts_and_anno(
            utr_sizes=self.data,
            protocol=self.protocol_,
        )

    def test_length(self):
        self.assertTupleEqual(self.var.shape, (6, 1))

    def test_vars_values_count(self):
        non_zeros_count = np.count_nonzero(self.var)
        self.assertEqual(non_zeros_count, 6)

    def test_utr_names(self):
        lengths_names = self.var.index.str.split("-").str[1].astype(int).to_list()
        print(lengths_names)
        self.assertListEqual(lengths_names, self.var.utr_size.to_list())


class TestObsSynthetic(TestGeneratePeakCountsSynthetic):
    def setUp(self):
        _, _, self.obs = peak_calling.generate_peak_counts_and_anno(
            utr_sizes=self.data,
            protocol=self.protocol_,
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
        subsetted = process_mapped_reads.subset_reference_data(
            utr_sizes=merged,
            utr_reference_regions=cls.utr_regions,
        )
        filtered = process_mapped_reads.filter_barcodes(
            utr_sizes=merged,
            whitelist=cls.protocol_.barcode_whitelist,
        )
        cls.matrix, var, cls.obs = peak_calling.generate_peak_counts_and_anno(
            utr_sizes=filtered,
            protocol=cls.protocol_,
        )
        cls.var = peak_calling.add_to_peaks_var(var, subsetted)


class TestAnnData(TestGenerateCountMatrixAnndata):
    def setUp(self) -> None:
        self.matrix_anndata = peak_calling.generate_count_matrix_anndata(
            self.matrix, self.var, self.obs
        )

    def test_matrix_values_count(self):
        non_zeros_count = np.count_nonzero(self.matrix_anndata.X)
        self.assertEqual(non_zeros_count, 755)

    def test_vars_values_count(self):
        non_zeros_count = np.count_nonzero(self.matrix_anndata.var)
        self.assertEqual(non_zeros_count, 99 * 9)

    def test_obs_values_count(self):
        non_zeros_count = np.count_nonzero(self.matrix_anndata.obs)
        self.assertEqual(non_zeros_count, 670 * 4)


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2, buffer=False)
    module_log.log_end_run("Finished testing")
