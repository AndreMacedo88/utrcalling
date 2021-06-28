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

from utrcalling import utr_mapping, platforms


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
        print(name, interval)
        self.assertListEqual([name, interval.start_d, interval.end_d,
                              interval.strand], correct)


class TestMapReadsToUtr(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        utr_regions = pd.read_pickle('./test_data/test_UTR_reference.pkl')
        cls.intervals = utr_mapping.generate_utr_intervals(
            utr_regions=utr_regions)
        cls.alignment = './test_data/test_UTRoldNSCgenome_bwt2_102_sort.bam'


class TestTenxTwo(TestMapReadsToUtr):
    def setUp(self) -> None:
        platform = platforms.Platform(platform='10x', version=2)
        self.mappings = utr_mapping.map_reads_to_utr(self.alignment, self.intervals,
                                                     platform, read='first')

    def test_details(self):
        dict_true = {
            'utr_name': 'ENSMUSG00000005804',
            'read_start_positions': [122748098, 122748088, 122748096],
            'read_strands': ['-', '-', '-'],
            'read_barcodes': ['GATGAAACATTAACCG', 'TGCGCAGAGTCCATAC', 'GCTGGGTGTACGCACC'],
            'read_umis': ['CAACACAAG', 'ATCGAGAGT', 'CTGAGTATG'],
            'read_lengths': [51, 37, 44],
            'utr_sizes_from_reads': [1468, 1458, 1466]
        }
        self.assertDictEqual(
            self.mappings.ENSMUSG00000005804.todict(), dict_true)

    def test_length(self):
        self.assertEqual(len(self.mappings), 126)


class TestTenxThree(TestMapReadsToUtr):
    def setUp(self) -> None:
        platform = platforms.Platform(platform='10x', version=3)
        self.mappings = utr_mapping.map_reads_to_utr(self.alignment, self.intervals,
                                                     platform, read='first')

    def test_details(self):
        dict_true = {
            'utr_name': 'ENSMUSG00000005804',
            'read_start_positions': [122748098, 122748088, 122748096],
            'read_strands': ['-', '-', '-'],
            'read_barcodes': ['GATGAAACATTAACCG', 'TGCGCAGAGTCCATAC', 'GCTGGGTGTACGCACC'],
            'read_umis': ['CAACACAAGTT', 'ATCGAGAGTTT', 'CTGAGTATGTT'],
            'read_lengths': [51, 37, 44],
            'utr_sizes_from_reads': [1468, 1458, 1466]
        }
        self.assertDictEqual(
            self.mappings.ENSMUSG00000005804.todict(), dict_true)

    def test_length(self):
        self.assertEqual(len(self.mappings), 126)


class TestPlatformsNotSupported(TestMapReadsToUtr):
    def test_platform(self):
        with self.assertRaises(RuntimeError):
            platforms.Platform(platform='Zehn_x', version=3)

    def test_version_tenx(self):
        with self.assertRaises(RuntimeError):
            platforms.Platform(platform='10x', version=1)

    def test_version_smartseq(self):
        with self.assertRaises(RuntimeError):
            platforms.Platform(platform='Smart-seq', version=1)


if __name__ == '__main__':
    unittest.main()
