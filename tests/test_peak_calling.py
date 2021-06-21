import os
import sys
from pathlib import Path
import unittest

import pybedtools

# make sure main module gets imported
file_dir = str(Path(__file__).resolve().parents[1])
sys.path.append(file_dir)
# make sure file paths resolve correctly for the test case
os.chdir(str(Path(__file__).resolve().parents[0]))

from utrcalling import peak_calling, utr_mapping


class TestPeakCalling(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.utr_regions = pybedtools.BedTool('./test_data/test_bed_merged.bed')
        intervals = utr_mapping.generate_utr_intervals(utr_regions=cls.utr_regions)
        alignment_1 = './test_data/test_UTRoldNSCgenome_bwt2_102_sort.bam'
        alignment_2 = './test_data/test_UTRyoungNSCgenome_bwt2_102_sort.bam'
        cls.mappings = [
            utr_mapping.map_reads_to_utr(alignment, intervals, 
            read='first', version=2) for alignment in [alignment_1, alignment_2]
            ]
    

class TestMergeSamplesUtrSizes(TestPeakCalling):
    def setUp(self) -> None:
        self.merged = peak_calling.merge_samples_utr_sizes(self.mappings)
    
    def test_details(self):
        details_true = {
            'utr_name': 'ENSMUSG00000025917', 
            'read_start_positions': [10024824, 10024829, 10024829, 10024833, 10024602, 
                10024826, 10024829, 10024829, 10024835, 10024839], 
            'read_strands': ['+', '+', '+', '+', '+', '+', '+', '+', '+', '+'], 
            'read_barcodes': ['CGTTGGGCAGTCAGAG-1', 'CCGTGGAGTCTCGTTC-1', 
                'GTAGTCAAGCTATGCT-1', 'TGCGCAGAGGCATGTG-1', 'CGACCTTTCTGCTGCT-2', 
                'TGTTCCGAGGGTCGAT-2', 'GTACGTAAGCGTTTAC-2', 'GTGCGGTAGGACAGCT-2', 
                'GCTGCAGCACTGTGTA-2', 'GTCCTCAGTAAACGCG-2'], 
            'read_umis': ['TGGCTTATA', 'AACGTAACA', 'ACCTCCGGT', 'ACCCGCACT', 
                'TCACGTGAG', 'TCCTAAATG', 'ATGATCGGT', 'TCCAGGAAT', 'TTTAGAACT', 
                'GCTAGGACG'], 
            'read_lengths': [46, 50, 48, 46, 43, 45, 54, 50, 41, 41], 
            'utr_sizes_from_reads': [153, 148, 148, 144, 375, 151, 148, 148, 142, 138]
        }
        self.assertDictEqual(self.merged.ENSMUSG00000025917.todict(), details_true)
    
    def test_length(self):
        self.assertEqual(len(self.merged), 71)


class TestSubsetReferenceData(TestPeakCalling):
    def setUp(self) -> None:
        merged = peak_calling.merge_samples_utr_sizes(self.mappings)
        self.subsetted = peak_calling.subset_reference_data(merged, self.utr_regions)
    
    def test_number_of_features(self) -> None:
        self.assertEqual(self.subsetted.shape[0], 71)
    
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
        'ENSMUST00000027050', 1]
        self.assertListEqual(self.subsetted.iloc[0].values.tolist(), proper_values)
    
    def test_strand_is_correct(self) -> None:
        self.assertEqual(self.subsetted.iloc[0].strand, '-')


class TestFilterUtrSizes(TestPeakCalling):
    def setUp(self) -> None:
        merged = peak_calling.merge_samples_utr_sizes(self.mappings)
        merged.ENSMUSG00000001036.utr_sizes_from_reads = [1, 798, 1, 798, 1, 798]
        self.filtered = peak_calling.filter_utr_sizes(merged)

    def test_number_of_features(self) -> None:
        self.assertEqual(len(self.filtered), 71)
    
    def test_is_filtered(self):
        dict_true = {
            'utr_name': 'ENSMUSG00000001036',
            'read_start_positions': [61518618, 61518618, 61518618],
            'read_strands': ['+', '+', '+'],
            'read_barcodes': ['CTAATGGCATTCGACA-1',
            'CCCAGTTTCGCTTGTC-1',
            'TATTACCAGGATTCGG-2'],
            'read_umis': ['ACCGAGCGT', 'ATGGTAACG', 'GCGATAGGA'],
            'read_lengths': [59, 52, 52],
            'utr_sizes_from_reads': [798, 798, 798]
            }
        self.assertDictEqual(self.filtered.ENSMUSG00000001036.todict(), dict_true)
