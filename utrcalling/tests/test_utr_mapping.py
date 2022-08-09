import os
from pathlib import Path
import unittest

import pandas as pd

# make sure file paths resolve correctly for the test case
os.chdir(str(Path(__file__).resolve().parents[0]))

from utrcalling.main_module import protocols, utr_mapping, module_log


class TestGenerateUtrIntervals(unittest.TestCase):
    def setUp(self) -> None:
        utr_regions = pd.read_pickle('./test_data/test_UTR_reference.pkl')
        self.intervals = utr_mapping.generate_utr_intervals(
            utr_regions=utr_regions)

    def test_correct_interval(self) -> None:
        """
        Here we test if the intervals are being generated correctly. Please note that 
        since GTF files use 0 indexing and BAM files start indexing on 1, when converted 
        to the HTSeq object the start location will be a smaller number by 1.
        """
        correct = ['ENSMUSG00000051285', 7160891, 7161063, '+']
        name, interval = list(self.intervals.items())[0]
        self.assertListEqual([name, interval.start_d, interval.end_d,
                              interval.strand], correct)


class TestMapReadsToUtr(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        utr_regions = pd.read_pickle('./test_data/test_UTR_reference.pkl')
        utr_regions_5prime = pd.read_pickle(
            './test_data/test_UTR_reference_5prime.pkl'
        )

        # test data for 3'UTR
        cls.intervals = utr_mapping.generate_utr_intervals(
            utr_regions=utr_regions)
        cls.alignment = './test_data/test_UTRoldNSCgenome_bwt2_102_duplicates_sort.bam'

        # test data for 5'UTR
        cls.intervals_5prime = utr_mapping.generate_utr_intervals(
            utr_regions=utr_regions_5prime)
        cls.alignment_cap = './test_data/test_CapPhospho_genome_102_coordsort.bam'


class TestTenxTwo(TestMapReadsToUtr):
    def setUp(self) -> None:
        self.maxDiff = None
        protocol = protocols.protocol(platform='10x', version=2)
        self.mappings = utr_mapping.map_reads_to_utr(self.alignment, self.intervals,
                                                     protocol, read='first')

    def test_details(self):
        dict_true = {
            'read_start_positions': [122748098, 122748087, 122748096],
            'read_strands': ['-', '-', '-'],
            'read_barcodes': ['GATGAAACATTAACCG', 'TGCGCAGAGTCCATAC', 'GCTGGGTGTACGCACC'],
            'read_umis': ['TCAACACAAG', 'AATCGAGAGT', 'CCTGAGTATG'],
            'read_lengths': [51, 37, 44],
            'utr_sizes_from_reads': [1468, 1457, 1466]
        }
        self.assertDictEqual(
            self.mappings.loc["ENSMUSG00000005804"].to_dict(), dict_true)

    def test_length(self):
        self.assertEqual(self.mappings.shape[0], 118)


class TestTenxThree(TestMapReadsToUtr):
    def setUp(self) -> None:
        self.maxDiff = None
        protocol = protocols.protocol(platform='10x', version=3)
        self.mappings = utr_mapping.map_reads_to_utr(self.alignment, self.intervals,
                                                     protocol, read='first')

    def test_details(self):
        dict_true = {
            'read_start_positions': [122748098, 122748087, 122748096],
            'read_strands': ['-', '-', '-'],
            'read_barcodes': ['GATGAAACATTAACCG', 'TGCGCAGAGTCCATAC', 'GCTGGGTGTACGCACC'],
            'read_umis': ['TCAACACAAGTT', 'AATCGAGAGTTT', 'CCTGAGTATGTT'],
            'read_lengths': [51, 37, 44],
            'utr_sizes_from_reads': [1468, 1457, 1466]
        }
        self.assertDictEqual(
            self.mappings.loc["ENSMUSG00000005804"].to_dict(), dict_true)

    def test_length(self):
        self.assertEqual(self.mappings.shape[0], 118)


class TestSmartSeq3(TestMapReadsToUtr):
    def setUp(self) -> None:
        protocol = protocols.protocol(platform='smartseq', version=3)
        self.mappings = utr_mapping.map_reads_to_utr(self.alignment, self.intervals,
                                                     protocol, read='first',
                                                     sample="test_sample")

    def test_details(self):
        dict_true = {
            'read_start_positions': [122748098, 122748087, 122748096],
            'read_strands': ['-', '-', '-'],
            'read_barcodes': ['test_sample', 'test_sample', 'test_sample'],
            'read_umis': ['ACATTAAC', 'GAGTCCAT', 'TGTACGCA'],
            'read_lengths': [51, 37, 44],
            'utr_sizes_from_reads': [1468, 1457, 1466]
        }
        self.assertDictEqual(
            self.mappings.loc["ENSMUSG00000005804"].to_dict(), dict_true)

    def test_length(self):
        self.assertEqual(self.mappings.shape[0], 118)


class TestSmartSeq3Cap(TestMapReadsToUtr):
    def setUp(self) -> None:
        protocol = protocols.protocol(platform='smart-seq_cap', version=3)
        self.mappings = utr_mapping.map_reads_to_utr(self.alignment_cap,
                                                     self.intervals_5prime,
                                                     protocol, read='first',
                                                     sample="test_sample")

    def test_details(self):
        dict_true = {
            'read_start_positions': [23677423, 23677444, 23677448],
            'read_strands': ['-', '-', '-'],
            'read_barcodes': ['test_sample', 'test_sample', 'test_sample'],
            'read_umis': ['CGCCAGGC', 'GACACGAT', 'AAGGGGAC'],
            'read_lengths': [81, 82, 82],
            'utr_sizes_from_reads': [44, 23, 19]
        }
        self.assertDictEqual(
            self.mappings.loc["ENSMUSG00000023905"].to_dict(), dict_true)

    def test_length(self):
        self.assertEqual(self.mappings.shape[0], 31)


class TestProtocolNotSupported(TestMapReadsToUtr):
    def test_platform(self):
        with self.assertRaises(NotImplementedError):
            protocols.protocol(platform='Zehn_x', version=3)

    def test_version_tenx(self):
        with self.assertRaises(NotImplementedError):
            protocols.protocol(platform='10x', version=1)

    def test_version_smartseq(self):
        with self.assertRaises(NotImplementedError):
            protocols.protocol(platform='Smart-seq', version=1)


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2, buffer=True)
    module_log.log_end_run("Finished testing")
