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

from utrcalling import utr_mapping


class TestGenerateUtrIntervals(unittest.TestCase):
    def setUp(self) -> None:
        utr_regions = pybedtools.BedTool('./test_data/test_bed_merged.bed')
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
        utr_regions = pybedtools.BedTool('./test_data/test_bed_merged.bed')
        cls.intervals = utr_mapping.generate_utr_intervals(
            utr_regions=utr_regions)
        cls.alignment = './test_data/test_UTRoldNSCgenome_bwt2_102_sort.bam'

    def setUp(self) -> None:
        self.mappings = utr_mapping.map_reads_to_utr(self.alignment, self.intervals,
                                                     read='first', version=2)

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
