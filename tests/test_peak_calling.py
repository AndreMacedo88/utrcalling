import os
import sys
from pathlib import Path
import unittest

import pandas as pd
import numpy as np

# make sure main module gets imported
file_dir = str(Path(__file__).resolve().parents[1])
sys.path.append(file_dir)
# make sure file paths resolve correctly for the test case
os.chdir(str(Path(__file__).resolve().parents[0]))

from main_module import peak_calling, protocols, utr_mapping, storage


class TestPeakCalling(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.utr_regions = pd.read_pickle('./test_data/test_UTR_reference.pkl')
        intervals = utr_mapping.generate_utr_intervals(utr_regions=cls.utr_regions)
        alignment_1 = './test_data/test_UTRoldNSCgenome_bwt2_102_sort.bam'
        alignment_2 = './test_data/test_UTRyoungNSCgenome_bwt2_102_sort.bam'
        cls.protocol = protocols.protocol(platform='10x', version=2)
        cls.mappings = [
            utr_mapping.map_reads_to_utr(alignment, intervals, cls.protocol,
                                         read='first') for alignment in [alignment_1,
                                                                         alignment_2]
        ]


class TestMergeSamplesUtrSizes(TestPeakCalling):
    def setUp(self) -> None:
        self.maxDiff = None
        self.merged = peak_calling.merge_samples_utr_sizes(self.mappings,
                                                           method=self.protocol)

    def test_details(self):
        details_true = {
            'utr_name': 'ENSMUSG00000025917',
            'read_start_positions': [10024879, 10024880, 10024882, 10024888, 10024660,
                                     10024882, 10024876, 10024880, 10024893, 10024899],
            'read_strands': ['+', '+', '+', '+', '+', '+', '+', '+', '+', '+'],
            'read_barcodes': ['CGTTGGGCAGTCAGAG-1', 'CCGTGGAGTCTCGTTC-1',
                              'GTAGTCAAGCTATGCT-1', 'TGCGCAGAGGCATGTG-1', 'CGACCTTTCTGCTGCT-2',
                              'TGTTCCGAGGGTCGAT-2', 'GTACGTAAGCGTTTAC-2', 'GTGCGGTAGGACAGCT-2',
                              'GCTGCAGCACTGTGTA-2', 'GTCCTCAGTAAACGCG-2'],
            'read_umis': ['CTGGCTTATA', 'CAACGTAACA', 'GACCTCCGGT', 'AACCCGCACT',
                          'GTCACGTGAG', 'TTCCTAAATG', 'CATGATCGGT', 'CTCCAGGAAT',
                          'CTTTAGAACT', 'TGCTAGGACG'],
            'read_lengths': [46, 50, 48, 46, 43, 45, 54, 50, 41, 41],
            'utr_sizes_from_reads': [98, 97, 95, 89, 317, 95, 101, 97, 84, 78]
        }
        self.assertDictEqual(
            self.merged.ENSMUSG00000025917.todict(), details_true)

    def test_length(self):
        self.assertEqual(len(self.merged), 66)


class TestMergeSamplesUtrSizesExclude(TestPeakCalling):
    def setUp(self) -> None:
        self.maxDiff = None
        self.merged = peak_calling.merge_samples_utr_sizes(self.mappings,
                                                           method=self.protocol,
                                                           exclude=['read_start_positions',
                                                                    'read_lengths',
                                                                    'read_strands'])

    def test_details(self):
        details_true = {
            'utr_name': 'ENSMUSG00000025917',
            'read_start_positions': [10024879, 10024880, 10024882, 10024888],
            'read_strands': ['+', '+', '+', '+'],
            'read_barcodes': ['CGTTGGGCAGTCAGAG-1', 'CCGTGGAGTCTCGTTC-1',
                              'GTAGTCAAGCTATGCT-1', 'TGCGCAGAGGCATGTG-1', 'CGACCTTTCTGCTGCT-2',
                              'TGTTCCGAGGGTCGAT-2', 'GTACGTAAGCGTTTAC-2', 'GTGCGGTAGGACAGCT-2',
                              'GCTGCAGCACTGTGTA-2', 'GTCCTCAGTAAACGCG-2'],
            'read_umis': ['CTGGCTTATA', 'CAACGTAACA', 'GACCTCCGGT', 'AACCCGCACT',
                          'GTCACGTGAG', 'TTCCTAAATG', 'CATGATCGGT', 'CTCCAGGAAT',
                          'CTTTAGAACT', 'TGCTAGGACG'],
            'read_lengths': [46, 50, 48, 46],
            'utr_sizes_from_reads': [98, 97, 95, 89, 317, 95, 101, 97, 84, 78]
        }
        self.assertDictEqual(
            self.merged.ENSMUSG00000025917.todict(), details_true)

    def test_length(self):
        self.assertEqual(len(self.merged), 66)


class TestSubsetReferenceData(TestPeakCalling):
    def setUp(self) -> None:
        merged = peak_calling.merge_samples_utr_sizes(self.mappings)
        self.subsetted = peak_calling.subset_reference_data(
            merged, self.utr_regions)

    def test_number_of_features(self) -> None:
        self.assertEqual(self.subsetted.shape[0], 66)

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
            self.subsetted.iloc[0].values.tolist(), proper_values)

    def test_strand_is_correct(self) -> None:
        self.assertEqual(self.subsetted.iloc[0].strand, '-')


class TestFilterBarcodes(TestPeakCalling):
    def setUp(self) -> None:
        self.merged = peak_calling.merge_samples_utr_sizes(self.mappings)
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
        self.assertNotEqual(len(self.merged.ENSMUSG00000025917['read_barcodes']),
                            len(filtered.ENSMUSG00000025917['read_barcodes']))

    def test_specific_utr_2(self):
        barcodes_real = ['CGTTGGGCAGTCAGAG-1', 'CCGTGGAGTCTCGTTC-1', 'TGCGCAGAGGCATGTG-1',
                         'CGACCTTTCTGCTGCT-2', 'TGTTCCGAGGGTCGAT-2', 'GTACGTAAGCGTTTAC-2',
                         'GTGCGGTAGGACAGCT-2', 'GCTGCAGCACTGTGTA-2', 'GTCCTCAGTAAACGCG-2']
        filtered = peak_calling.filter_barcodes(
            self.merged, annotations=self.annotations)
        self.assertListEqual(
            filtered.ENSMUSG00000025917['read_barcodes'], barcodes_real)

    def test_using_annotations_as_dataframe(self):
        annotations = pd.read_csv(self.annotations)
        filtered = peak_calling.filter_barcodes(
            self.merged, annotations=annotations)
        barcodes_in_data, barcodes_in_annotations = self.are_all_barcodes_in_anno(
            filtered, annotations
        )
        self.assertEqual(barcodes_in_data, barcodes_in_annotations)

    def test_filter_parallel(self):
        filtered = peak_calling.filter_barcodes(self.merged, annotations=self.annotations,
                                                parallel=True)
        annotations = pd.read_csv(self.annotations)
        barcodes_in_data, barcodes_in_annotations = self.are_all_barcodes_in_anno(
            filtered, annotations
        )
        self.assertEqual(barcodes_in_data, barcodes_in_annotations)


class TestGeneratePeakCounts(TestPeakCalling):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.merged = peak_calling.merge_samples_utr_sizes(cls.mappings)
        cls.annotations = './test_data/vivo_anno.csv'
        cls.merged = peak_calling.filter_barcodes(cls.merged, cls.annotations)


class TestMatrix(TestGeneratePeakCounts):
    def setUp(self):
        self.matrix, _, _ = peak_calling.generate_peak_counts(self.merged,
                                                              annotations=self.annotations)

    def test_length(self):
        self.assertTupleEqual(self.matrix.shape, (695, 160))

    def test_matrix_values_count(self):
        non_zeros_count = np.count_nonzero(self.matrix)
        self.assertEqual(non_zeros_count, 792)

    def test_matrix_maximum(self):
        self.assertEqual(np.max(self.matrix.values), 3.0)


class TestVar(TestGeneratePeakCounts):
    def setUp(self):
        _, self.var, _ = peak_calling.generate_peak_counts(self.merged,
                                                           annotations=self.annotations)

    def test_length(self):
        self.assertTupleEqual(self.var.shape, (160, 1))

    def test_vars_values_count(self):
        non_zeros_count = np.count_nonzero(self.var)
        self.assertEqual(non_zeros_count, 160)


class TestObs(TestGeneratePeakCounts):
    def setUp(self):
        _, _, self.obs = peak_calling.generate_peak_counts(self.merged,
                                                           annotations=self.annotations)

    def test_length(self):
        self.assertTupleEqual(self.obs.shape, (695, 3))

    def test_obs_values_count(self):
        non_zeros_count = np.count_nonzero(self.obs)
        self.assertEqual(non_zeros_count, 2085)


class TestNoAnno(TestGeneratePeakCounts):
    def setUp(self):
        self.matrix, self.var = peak_calling.generate_peak_counts(self.merged,
                                                                  annotations=None)

    def test_matrix_values_count(self):
        non_zeros_count = np.count_nonzero(self.matrix)
        self.assertEqual(non_zeros_count, 792)

    def test_vars_values_count(self):
        non_zeros_count = np.count_nonzero(self.var)
        self.assertEqual(non_zeros_count, 160)


class TestAnnoDf(TestGeneratePeakCounts):
    def setUp(self):
        annotations = pd.read_csv(self.annotations)
        self.matrix, self.var, self.obs = peak_calling.generate_peak_counts(self.merged,
                                                                            annotations=annotations)

    def test_matrix_values_count(self):
        non_zeros_count = np.count_nonzero(self.matrix)
        self.assertEqual(non_zeros_count, 792)

    def test_vars_values_count(self):
        non_zeros_count = np.count_nonzero(self.var)
        self.assertEqual(non_zeros_count, 160)

    def test_obs_values_count(self):
        non_zeros_count = np.count_nonzero(self.obs)
        self.assertEqual(non_zeros_count, 2085)


class TestGenerateCountMatrixAnndata(TestPeakCalling):
    @classmethod
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
        self.assertEqual(non_zeros_count, 792)

    def test_vars_values_count(self):
        non_zeros_count = np.count_nonzero(self.matrix_anndata.var)
        self.assertEqual(non_zeros_count, 160 * 9)

    def test_obs_values_count(self):
        non_zeros_count = np.count_nonzero(self.matrix_anndata.obs)
        self.assertEqual(non_zeros_count, 2085)


class TestCallPeaks():
    def setUp(self) -> None:
        st_pos = [20, 20, 21, 50, 50, 50, 70, 70, 60, 55, 70, 19, 49]

        read_information = storage.MappedReadInformation()
        read_information.read_start_positions.append()
        read_information.read_strands.append()
        read_information.read_barcodes.append()
        read_information.read_umis.append()
        read_information.read_lengths.append()
        read_information.utr_sizes_from_reads.append()
        self.utr = storage.UtrMappings()

        self.func = peak_calling.call_peaks_using_kernel_density_estimator


if __name__ == '__main__':
    unittest.main()
