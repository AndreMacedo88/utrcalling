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


class TestStorage(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        utr_regions = pd.read_pickle('./test_data/test_utr_reference.pkl')
        intervals = utr_mapping.generate_utr_intervals(utr_regions=utr_regions)
        alignment = './test_data/test_UTRoldNSCgenome_bwt2_102_sort.bam'
        platform = platforms.Platform(platform='10x', version=2)
        cls.mappings = utr_mapping.map_reads_to_utr(alignment, intervals, platform,
                                                    read='first')


class TestUtrMappings(TestStorage):

    def test_indexing_simple(self):
        self.assertEqual(self.mappings[0], 'ENSMUSG00000025917')

    def test_indexing_slice(self):
        self.assertListEqual(self.mappings[0:4:2],
                             ['ENSMUSG00000025917', 'ENSMUSG00000037509'])


class TestMappedReadInformation(TestStorage):

    def test_todict(self):
        dict_true = {
            'utr_name': 'ENSMUSG00000025917',
            'read_start_positions': [10024824, 10024829, 10024829, 10024833],
            'read_strands': ['+', '+', '+', '+'],
            'read_barcodes': ['CGTTGGGCAGTCAGAG', 'CCGTGGAGTCTCGTTC',
                              'GTAGTCAAGCTATGCT', 'TGCGCAGAGGCATGTG'],
            'read_umis': ['TGGCTTATA', 'AACGTAACA', 'ACCTCCGGT', 'ACCCGCACT'],
            'read_lengths': [46, 50, 48, 46],
            'utr_sizes_from_reads': [153, 148, 148, 144]
        }
        self.assertDictEqual(
            self.mappings.ENSMUSG00000025917.todict(), dict_true)


if __name__ == '__main__':
    unittest.main()
