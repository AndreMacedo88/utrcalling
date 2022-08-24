import os
from pathlib import Path
import unittest

import pandas as pd
import numpy as np
from numpy import testing as npt
import copy

# make sure file paths resolve correctly for the test case
os.chdir(str(Path(__file__).resolve().parents[0]))

from utrcalling.core import peak_calling, protocols, utr_mapping, module_log, helpers


ANNOTATION = pd.DataFrame(columns=["celltype", "barcodes",
                                   "cool_annotation"],
                          data=list(
    zip(
        ["A", "B", "A", "C", "B"],
        ["AAAAAAAAAAAAAAAA-1", "AAAAAAAAAAAAAAAG-1", "GAAAAAAAAAAAAAAT-2",
         "AAAAAAAAAAAAATTT-1", "AAAAAAAAAAAAAAGG-2"],
        [2.3, 4.4, 5.0, 2.1, 3.0]
    )
)
)


class DataSynthetic:
    def __init__(self) -> None:
        read_start_positions = [[100, 101, 100, 200, 202, 300],
                                [1000, 1100, 1102, 1200, 1200]]
        read_strands = [["+", "+", "+", "+", "+", "+"], ["-", "-", "-", "-", "+"]]
        read_barcodes = [["AAAAAAAAAAAAAAAA-1", "AAAAAAAAAAAAAAAA-1",
                          "GAAAAAAAAAAAAAAT-2", "GAAAAAAAAAAAAAAT-2",
                          "AAAAAAAAAAAAAAAG-2", "AAAAAAAAAAAAAAAG-1"],
                         ["AAAAAAAAAAAAAAAG-1", "AAAAAAAAAAAAAGGG-2",
                          "AAAAAAAAAAAAAAAT-2", "AAAAAAAAAAAAAATT-1",
                          "ATATATATATATATAT-1"]]
        read_umis = [
            ["CCCCCCCCGG", "CCCCCCCCTT", "CCCCCCCCAA", "CCCCCCCCAA", "CCCCCCCGGC",
             "CCCCCCCTTC"],
            ["AAAAAAAAAA", "AAAAAAAAAT", "AAAAAAAAAG", "AAAAAAAAAC", "CTCTCTCTCT"]
        ]
        read_lengths = [
            [49, 59, 51, 52, 48, 52],
            [50, 52, 53, 52, 70]
        ]
        utr_sizes_from_reads = [
            [500, 499, 500, 400, 398, 300],
            [100, 200, 202, 300, 300]
        ]

        self._data_synthetic = pd.DataFrame(columns=['read_start_positions', 'read_strands',
                                                     'read_barcodes', 'read_umis',
                                                     'read_lengths', 'utr_sizes_from_reads'],
                                            index=["ENSMUSG00000000001",
                                                   "ENSMUSG00000000002"],
                                            data=list(
            zip(
                read_start_positions, read_strands, read_barcodes, read_umis, read_lengths,
                utr_sizes_from_reads
            )
        )
        )

        self._anno = ANNOTATION

        @property
        def data_synthetic(self):
            return copy.deepcopy(self._data_synthetic)

        @property
        def anno(self):
            return copy.deepcopy(self._anno)


class DataFiltered:
    def __init__(self) -> None:
        read_start_positions = [[100, 101, 100, 200, 202, 300],
                                [1000, 1100, 1102, 1200]]
        read_strands = [["+", "+", "+", "+", "+", "+"], ["-", "-", "-", "-"]]
        read_barcodes = [
            ['AAAAAAAAAAAAAAAA-1', 'AAAAAAAAAAAAAAAA-1',
             'GAAAAAAAAAAAAAAT-2', 'GAAAAAAAAAAAAAAT-2',
             'AAAAAAAAAAAAAAGG-2', 'AAAAAAAAAAAAAAAG-1'],
            ['AAAAAAAAAAAAAAAG-1', 'AAAAAAAAAAAAAAGG-2',
             'GAAAAAAAAAAAAAAT-2', 'AAAAAAAAAAAAATTT-1']
        ]
        read_umis = [
            ["CCCCCCCCGG", "CCCCCCCCTT", "CCCCCCCCAA", "CCCCCCCCAA", "CCCCCCCGGC",
             "CCCCCCCTTC"],
            ["AAAAAAAAAA", "AAAAAAAAAT", "AAAAAAAAAG", "AAAAAAAAAC"]
        ]
        read_lengths = [
            [49, 59, 51, 52, 48, 52],
            [50, 52, 53, 52]
        ]
        utr_sizes_from_reads = [
            [500, 499, 500, 400, 398, 300],
            [100, 200, 202, 300]
        ]

        self._data_filtered = pd.DataFrame(columns=['read_start_positions', 'read_strands',
                                                    'read_barcodes', 'read_umis',
                                                    'read_lengths', 'utr_sizes_from_reads'],
                                           index=["ENSMUSG00000000001",
                                                  "ENSMUSG00000000002"],
                                           data=list(
            zip(
                read_start_positions, read_strands, read_barcodes, read_umis, read_lengths,
                utr_sizes_from_reads
            )
        )
        )

        self._anno = ANNOTATION

        @property
        def data_filtered(self):
            return copy.deepcopy(self._data_filtered)

        @property
        def anno(self):
            return copy.deepcopy(self._anno)


class TestPeakCalling(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.utr_regions = pd.read_pickle('./test_data/test_UTR_reference.pkl')
        intervals = utr_mapping.generate_utr_intervals(utr_regions=cls.utr_regions)
        alignment_1 = './test_data/test_UTRoldNSCgenome_bwt2_102_sort.bam'
        alignment_2 = './test_data/test_UTRyoungNSCgenome_bwt2_102_sort.bam'
        cls.protocol = protocols.protocol(platform='10x', version=2,
                                          merge_method="intersection")
        cls.protocol_union = protocols.protocol(platform='10x', version=2,
                                                merge_method="union")
        cls.mappings = [
            utr_mapping.map_reads_to_utr(alignment, intervals, cls.protocol,
                                         read='first') for alignment in [alignment_1,
                                                                         alignment_2]
        ]


class TestMergeSamplesUtrSizesIntersection(TestPeakCalling):
    def setUp(self) -> None:
        self.maxDiff = None
        self.merged = peak_calling.merge_samples_utr_sizes(
            self.mappings,
            method=self.protocol.merge_method)

    def test_details(self):
        details_true = {
            'read_start_positions': [10024824, 10024829, 10024829, 10024833, 10024602,
                                     10024826, 10024829, 10024829, 10024835, 10024839],
            'read_strands': ['+', '+', '+', '+', '+', '+', '+', '+', '+', '+'],
            'read_barcodes': ['CGTTGGGCAGTCAGAG-1', 'CCGTGGAGTCTCGTTC-1',
                              'GTAGTCAAGCTATGCT-1', 'TGCGCAGAGGCATGTG-1', 'CGACCTTTCTGCTGCT-2',
                              'TGTTCCGAGGGTCGAT-2', 'GTACGTAAGCGTTTAC-2', 'GTGCGGTAGGACAGCT-2',
                              'GCTGCAGCACTGTGTA-2', 'GTCCTCAGTAAACGCG-2'],
            'read_umis': ['CTGGCTTATA', 'CAACGTAACA', 'GACCTCCGGT', 'AACCCGCACT',
                          'GTCACGTGAG', 'TTCCTAAATG', 'CATGATCGGT', 'CTCCAGGAAT',
                          'CTTTAGAACT', 'TGCTAGGACG'],
            'read_lengths': [46, 50, 48, 46, 43, 45, 54, 50, 41, 41],
            'utr_sizes_from_reads': [153, 148, 148, 144, 375, 151, 148, 148, 142, 138]
        }
        self.assertDictEqual(
            self.merged.loc["ENSMUSG00000025917"].to_dict(), details_true)

    def test_length(self):
        self.assertEqual(self.merged.shape[0], 63)


class TestMergeSamplesUtrSizesUnion(TestPeakCalling):
    def setUp(self) -> None:
        self.maxDiff = None
        self.merged = peak_calling.merge_samples_utr_sizes(
            copy.deepcopy(self.mappings),
            method=self.protocol_union.merge_method)

    def test_identical_index(self):
        details_true = {
            'read_start_positions': [10024824, 10024829, 10024829, 10024833, 10024602,
                                     10024826, 10024829, 10024829, 10024835, 10024839],
            'read_strands': ['+', '+', '+', '+', '+', '+', '+', '+', '+', '+'],
            'read_barcodes': ['CGTTGGGCAGTCAGAG-1', 'CCGTGGAGTCTCGTTC-1',
                              'GTAGTCAAGCTATGCT-1', 'TGCGCAGAGGCATGTG-1', 'CGACCTTTCTGCTGCT-2',
                              'TGTTCCGAGGGTCGAT-2', 'GTACGTAAGCGTTTAC-2', 'GTGCGGTAGGACAGCT-2',
                              'GCTGCAGCACTGTGTA-2', 'GTCCTCAGTAAACGCG-2'],
            'read_umis': ['CTGGCTTATA', 'CAACGTAACA', 'GACCTCCGGT', 'AACCCGCACT',
                          'GTCACGTGAG', 'TTCCTAAATG', 'CATGATCGGT', 'CTCCAGGAAT',
                          'CTTTAGAACT', 'TGCTAGGACG'],
            'read_lengths': [46, 50, 48, 46, 43, 45, 54, 50, 41, 41],
            'utr_sizes_from_reads': [153, 148, 148, 144, 375, 151, 148, 148, 142, 138]
        }
        self.assertDictEqual(
            self.merged.loc["ENSMUSG00000025917"].to_dict(), details_true)

    def test_index_on_first_df(self):
        details_true = {
            'read_start_positions': [136881390, 136881415],
            'read_strands': ['+', '+'],
            'read_barcodes': ['CGAGAAGGTGCAACGA-1', 'CAGTCCTAGAAGGACA-1'],
            'read_umis': ['CCCACAGCCA', 'TGCCTCACGG'],
            'read_lengths': [53, 48],
            'utr_sizes_from_reads': [53, 28]
        }
        self.assertDictEqual(
            self.merged.loc["ENSMUSG00000115423"].to_dict(), details_true)

    def test_index_on_second_df(self):
        details_true = {
            'read_start_positions': [107584044],
            'read_strands': ['-'],
            'read_barcodes': ['ACAGCTATCCTTAATC-2'],
            'read_umis': ['AATAGTCCAT'],
            'read_lengths': [50],
            'utr_sizes_from_reads': [192]
        }
        self.assertDictEqual(
            self.merged.loc["ENSMUSG00000079334"].to_dict(), details_true)

    def test_length(self):
        self.assertEqual(
            self.merged.shape[0], self.mappings[0].index.union(
                self.mappings[1].index).shape[0]
        )


class TestSubsetReferenceData(TestPeakCalling):
    def setUp(self) -> None:
        merged = peak_calling.merge_samples_utr_sizes(copy.deepcopy(self.mappings))
        self.subsetted = peak_calling.subset_reference_data(
            merged, self.utr_regions)

    def test_number_of_features(self) -> None:
        self.assertEqual(self.subsetted.shape[0], 63)

    def test_number_of_fields(self) -> None:
        """
        Number of fields in the annotation BED file should be equal to the optional
        fields (in variable 'annotations') plus the mandatory fields 'seqname', 'start',
        'end' and finally the count field, that tell us how many features were merged
        at each new feature (row).
        """
        self.assertEqual(self.subsetted.shape[1], 8)

    def test_values_are_proper(self) -> None:
        proper_values = ['1', 10024601, 10024978, 'ENSMUSG00000025917', 0.0, '-',
                         'ENSMUST00000027050', 'Cops5']
        self.assertListEqual(
            self.subsetted.loc["ENSMUSG00000025917"].values.tolist(), proper_values)

    def test_strand_is_correct(self) -> None:
        self.assertEqual(self.subsetted.iloc[0].strand, '-')


class TestFilterBarcodes(TestPeakCalling):
    def setUp(self) -> None:
        self.merged = peak_calling.merge_samples_utr_sizes(copy.deepcopy(self.mappings))
        self.annotations = './test_data/vivo_anno.csv'

    @staticmethod
    def are_all_barcodes_in_anno(data, annotations):
        counter = 0
        control = 0
        barcodes_real = annotations.CellBarcode
        for barcodes in data.read_barcodes:
            control += 1
            if set(barcodes).issubset(barcodes_real):
                counter += 1
        return counter, control

    def test_before_filter(self):
        annotations = pd.read_csv(self.annotations)
        barcodes_in_data, barcodes_in_annotations = self.are_all_barcodes_in_anno(
            self.merged, annotations
        )
        self.assertNotEqual(barcodes_in_data, barcodes_in_annotations)

    def test_filter(self):
        filtered = peak_calling.filter_barcodes(
            self.merged, annotations=self.annotations)
        annotations = pd.read_csv(self.annotations)
        barcodes_in_data, barcodes_in_annotations = self.are_all_barcodes_in_anno(
            filtered, annotations
        )
        self.assertEqual(barcodes_in_data, barcodes_in_annotations)

    def test_specific_utr(self):
        filtered = peak_calling.filter_barcodes(
            self.merged, annotations=self.annotations)
        self.assertNotEqual(len(self.merged.loc["ENSMUSG00000025917"]['read_barcodes']),
                            len(filtered.loc["ENSMUSG00000025917"]['read_barcodes']))

    def test_specific_utr_2(self):
        barcodes_real = ['CGTTGGGCAGTCAGAG-1', 'CCGTGGAGTCTCGTTC-1', 'TGCGCAGAGGCATGTG-1',
                         'CGACCTTTCTGCTGCT-2', 'TGTTCCGAGGGTCGAT-2', 'GTACGTAAGCGTTTAC-2',
                         'GTGCGGTAGGACAGCT-2', 'GCTGCAGCACTGTGTA-2', 'GTCCTCAGTAAACGCG-2']
        filtered = peak_calling.filter_barcodes(
            self.merged, annotations=self.annotations)
        self.assertListEqual(
            filtered.loc["ENSMUSG00000025917"]['read_barcodes'], barcodes_real)

    def test_using_annotations_as_dataframe(self):
        annotations = pd.read_csv(self.annotations)
        filtered = peak_calling.filter_barcodes(
            self.merged, annotations=annotations)
        barcodes_in_data, barcodes_in_annotations = self.are_all_barcodes_in_anno(
            filtered, annotations
        )
        self.assertEqual(barcodes_in_data, barcodes_in_annotations)


class TestFilterBarcodesSynthetic(unittest.TestCase):
    def setUp(self) -> None:
        data_synthetic = DataSynthetic()
        self.data = data_synthetic._data_synthetic
        self.anno = data_synthetic._anno
        self.filtered = peak_calling.filter_barcodes(self.data, annotations=self.anno)
        self.barcodes_filtered = list(helpers.flatten(self.filtered.read_barcodes))

    def test_barcodes_in_anno(self):
        self.assertFalse(
            [barcode for barcode in self.barcodes_filtered
             if barcode not in self.anno.barcodes.tolist()]
        )

    def test_filtered_len(self):
        self.assertEqual(len(self.barcodes_filtered), 10)

    def test_filtered_id(self):
        correct = ["AAAAAAAAAAAAAAAA-1", "AAAAAAAAAAAAAAAA-1",
                   "GAAAAAAAAAAAAAAT-2", "GAAAAAAAAAAAAAAT-2",
                   "AAAAAAAAAAAAAAGG-2", "AAAAAAAAAAAAAAAG-1",
                   "AAAAAAAAAAAAAAAG-1", "AAAAAAAAAAAAAAGG-2",
                   "GAAAAAAAAAAAAAAT-2", "AAAAAAAAAAAAATTT-1"]
        self.assertListEqual(correct, self.barcodes_filtered)


class TestGeneratePeakCounts(TestPeakCalling):
    @ classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.merged = peak_calling.merge_samples_utr_sizes(copy.deepcopy(cls.mappings))
        cls.annotations = './test_data/vivo_anno.csv'
        cls.merged = peak_calling.filter_barcodes(cls.merged, cls.annotations)


class TestMatrix(TestGeneratePeakCounts):
    def setUp(self):
        self.matrix, _, _ = peak_calling.generate_peak_counts(self.merged,
                                                              annotations=self.annotations)

    def test_length(self):
        self.assertTupleEqual(self.matrix.shape, (670, 173))

    def test_matrix_values_count(self):
        non_zeros_count = np.count_nonzero(self.matrix)
        self.assertEqual(non_zeros_count, 763)

    def test_matrix_maximum(self):
        self.assertEqual(np.max(self.matrix.values), 3)


class TestVar(TestGeneratePeakCounts):
    def setUp(self):
        _, self.var, _ = peak_calling.generate_peak_counts(self.merged,
                                                           annotations=self.annotations)

    def test_length(self):
        self.assertTupleEqual(self.var.shape, (173, 1))

    def test_vars_values_count(self):
        non_zeros_count = np.count_nonzero(self.var)
        self.assertEqual(non_zeros_count, 173)


class TestObs(TestGeneratePeakCounts):
    def setUp(self):
        _, _, self.obs = peak_calling.generate_peak_counts(self.merged,
                                                           annotations=self.annotations)

    def test_length(self):
        self.assertTupleEqual(self.obs.shape, (670, 3))

    def test_obs_values_count(self):
        non_zeros_count = np.count_nonzero(self.obs)
        self.assertEqual(non_zeros_count, 2010)


class TestNoAnno(TestGeneratePeakCounts):
    def setUp(self):
        self.matrix, self.var = peak_calling.generate_peak_counts(self.merged,
                                                                  annotations=None)

    def test_matrix_values_count(self):
        non_zeros_count = np.count_nonzero(self.matrix)
        self.assertEqual(non_zeros_count, 763)

    def test_vars_values_count(self):
        non_zeros_count = np.count_nonzero(self.var)
        self.assertEqual(non_zeros_count, 173)


class TestAnnoDf(TestGeneratePeakCounts):
    def setUp(self):
        annotations = pd.read_csv(self.annotations)
        self.matrix, self.var, self.obs = peak_calling.generate_peak_counts(
            self.merged,
            annotations=annotations
        )

    def test_matrix_values_count(self):
        non_zeros_count = np.count_nonzero(self.matrix)
        self.assertEqual(non_zeros_count, 763)

    def test_vars_values_count(self):
        non_zeros_count = np.count_nonzero(self.var)
        self.assertEqual(non_zeros_count, 173)

    def test_obs_values_count(self):
        non_zeros_count = np.count_nonzero(self.obs)
        self.assertEqual(non_zeros_count, 2010)


class TestGeneratePeakCountsSynthetic(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        data_filtered = DataFiltered()
        cls.data = data_filtered._data_filtered
        cls.anno = data_filtered._anno
        cls.protocol = protocols.protocol(platform='10x', version=2,
                                          merge_method="intersection")


class TestMatrixSynthetic(TestGeneratePeakCountsSynthetic):
    def setUp(self) -> None:
        self.matrix, _, _ = peak_calling.generate_peak_counts(
            self.data,
            annotations=self.anno,
            func=self.protocol.peak_calling
        )

    def test_length(self):
        self.assertTupleEqual(self.matrix.shape, (5, 6))

    def test_matrix_values_count(self):
        non_zeros_count = np.count_nonzero(self.matrix)
        self.assertEqual(non_zeros_count, 9)

    def test_matrix_maximum(self):
        self.assertEqual(np.max(self.matrix.values), 2)

    def test_matrix_composition(self):
        reality = np.array([[0, 0, 2, 0, 0, 0],
                            [1, 0, 0, 1, 0, 0],
                            [0, 1, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 1],
                            [0, 1, 1, 0, 1, 0]])
        npt.assert_allclose(self.matrix.sort_index().to_numpy(), reality)


class TestVarSynthetic(TestGeneratePeakCountsSynthetic):
    def setUp(self):
        _, self.var, _ = peak_calling.generate_peak_counts(
            self.data,
            annotations=self.anno,
            func=self.protocol.peak_calling
        )

    def test_length(self):
        self.assertTupleEqual(self.var.shape, (6, 1))

    def test_vars_values_count(self):
        non_zeros_count = np.count_nonzero(self.var)
        self.assertEqual(non_zeros_count, 6)


class TestObsSynthetic(TestGeneratePeakCountsSynthetic):
    def setUp(self):
        _, _, self.obs = peak_calling.generate_peak_counts(
            self.data,
            annotations=self.anno,
            func=self.protocol.peak_calling
        )

    def test_length(self):
        self.assertTupleEqual(self.obs.shape, (5, 2))

    def test_obs_values_count(self):
        non_zeros_count = np.count_nonzero(self.obs)
        self.assertEqual(non_zeros_count, 10)


class TestNoAnnoSynthetic(TestGeneratePeakCountsSynthetic):
    def setUp(self):
        self.matrix, self.var = peak_calling.generate_peak_counts(
            self.data, annotations=None, func=self.protocol.peak_calling
        )

    def test_matrix_values_count(self):
        non_zeros_count = np.count_nonzero(self.matrix)
        self.assertEqual(non_zeros_count, 9)

    def test_vars_values_count(self):
        non_zeros_count = np.count_nonzero(self.var)
        self.assertEqual(non_zeros_count, 6)


class TestGenerateCountMatrixAnndata(TestPeakCalling):
    @ classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        merged = peak_calling.merge_samples_utr_sizes(cls.mappings)
        subsetted = peak_calling.subset_reference_data(
            merged, cls.utr_regions)
        annotations = './test_data/vivo_anno.csv'
        filtered = peak_calling.filter_barcodes(merged, annotations)
        cls.matrix, var, cls.obs = peak_calling.generate_peak_counts(filtered,
                                                                     annotations=annotations)
        cls.var = peak_calling.add_to_peaks_var(var, subsetted)


class TestWithObs(TestGenerateCountMatrixAnndata):
    def setUp(self) -> None:
        self.matrix_anndata = peak_calling.generate_count_matrix_anndata(
            self.matrix, self.var, self.obs)

    def test_matrix_values_count(self):
        non_zeros_count = np.count_nonzero(self.matrix_anndata.X)
        self.assertEqual(non_zeros_count, 763)

    def test_vars_values_count(self):
        non_zeros_count = np.count_nonzero(self.matrix_anndata.var)
        self.assertEqual(non_zeros_count, 173 * 9)

    def test_obs_values_count(self):
        non_zeros_count = np.count_nonzero(self.matrix_anndata.obs)
        self.assertEqual(non_zeros_count, 670 * 3)


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2, buffer=True)
    module_log.log_end_run("Finished testing")
