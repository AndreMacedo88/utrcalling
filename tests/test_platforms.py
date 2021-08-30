import os
import sys
from pathlib import Path
import unittest

import pandas as pd

# make sure main module gets imported
file_dir = str(Path(__file__).resolve().parents[1])
sys.path.append(file_dir)
# make sure file paths resolve correctly for the test case
os.chdir(str(Path(__file__).resolve().parents[0]))

from utrcalling import platforms, peak_calling, utr_mapping


class Test10x(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.utr_regions = pd.read_pickle('./test_data/test_UTR_reference.pkl')
        intervals = utr_mapping.generate_utr_intervals(utr_regions=cls.utr_regions)
        alignment_1 = './test_data/test_UTRoldNSCgenome_bwt2_102_sort.bam'
        alignment_2 = './test_data/test_UTRyoungNSCgenome_bwt2_102_sort.bam'
        cls.protocol = platforms.Protocol(platform='10x', version=2)
        cls.mappings = [
            utr_mapping.map_reads_to_utr(alignment, intervals, cls.protocol,
                                         read='first') for alignment in [alignment_1,
                                                                         alignment_2]
        ]


class TestFilterUtrSizes(Test10x):
    def setUp(self) -> None:
        merged = peak_calling.merge_samples_utr_sizes(self.mappings)
        merged.ENSMUSG00000001036.utr_sizes_from_reads = [1, 798, 1, 798, 1, 798]
        self.filtered = self.protocol._filter_utr_sizes(merged)

    def test_number_of_features(self) -> None:
        self.assertEqual(len(self.filtered), 66)

    def test_is_filtered(self):
        dict_true = {
            'utr_name': 'ENSMUSG00000001036',
            'read_start_positions': [61518660, 61518663, 61518665],
            'read_strands': ['+', '+', '+'],
            'read_barcodes': ['CTAATGGCATTCGACA-1',
                              'CCCAGTTTCGCTTGTC-1',
                              'TATTACCAGGATTCGG-2'],
            'read_umis': ['ACCGAGCGT', 'ATGGTAACG', 'GCGATAGGA'],
            'read_lengths': [59, 52, 52],
            'utr_sizes_from_reads': [798, 798, 798]
        }
        self.assertDictEqual(
            self.filtered.ENSMUSG00000001036.todict(), dict_true)


class TestSmartSeqCapPhospho(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.utr_regions = pd.read_pickle('./test_data/test_UTR_reference_5prime.pkl')
        intervals = utr_mapping.generate_utr_intervals(utr_regions=cls.utr_regions)
        alignment = './test_data/test_CapPhospho_genome_102_coordsort.bam'
        cls.protocol = platforms.Protocol(platform='smart-seq_cap', version=3)
        cls.mappings = \
            utr_mapping.map_reads_to_utr(alignment, intervals, cls.protocol, read='first')


class TestFilterUtrSizesSmartSeqCap(TestSmartSeqCapPhospho):
    def setUp(self) -> None:
        self.mappings.ENSMUSG00000023905.utr_sizes_from_reads = [44, 1, 19]
        self.filtered = self.protocol._filter_utr_sizes(self.mappings)

    def test_number_of_features(self) -> None:
        self.assertEqual(len(self.filtered), 31)

    def test_is_filtered(self):
        dict_true = {
            'utr_name': 'ENSMUSG00000023905',
            'read_start_positions': [23677423, 23677444, 23677448],
            'read_strands': ['-', '-', '-'],
            'read_barcodes': ['S', 'S', 'S'],
            'read_umis': ['CGCCAGGC', 'GACACGAT', 'AAGGGGAC'],
            'read_lengths': [81, 82, 82],
            'utr_sizes_from_reads': [44, 1, 19]
        }
        self.assertDictEqual(
            self.filtered.ENSMUSG00000023905.todict(), dict_true)
