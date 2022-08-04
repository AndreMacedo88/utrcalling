import os
from pathlib import Path
import unittest

import pandas as pd

# make sure file paths resolve correctly for the test case
os.chdir(str(Path(__file__).resolve().parents[0]))

from utrcalling.main_module import protocols, utr_mapping, module_log
from utrcalling.main_module.storage import MappedReadInformation


class TestStorage(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.maxDiff = None
        utr_regions = pd.read_pickle('./test_data/test_UTR_reference.pkl')
        intervals = utr_mapping.generate_utr_intervals(utr_regions=utr_regions)
        alignment = './test_data/test_UTRoldNSCgenome_bwt2_102_sort.bam'
        protocol = protocols.protocol(platform='10x', version=2)
        cls.mappings = utr_mapping.map_reads_to_utr(alignment, intervals, protocol,
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
            'read_start_positions': [10024879, 10024880, 10024882, 10024888],
            'read_strands': ['+', '+', '+', '+'],
            'read_barcodes': ['CGTTGGGCAGTCAGAG', 'CCGTGGAGTCTCGTTC',
                              'GTAGTCAAGCTATGCT', 'TGCGCAGAGGCATGTG'],
            'read_umis': ['CTGGCTTATA', 'CAACGTAACA', 'GACCTCCGGT', 'AACCCGCACT'],
            'read_lengths': [46, 50, 48, 46],
            'utr_sizes_from_reads': [98, 97, 95, 89]
        }
        self.assertDictEqual(
            self.mappings.ENSMUSG00000025917.todict(), dict_true)


class TestGetUniqueUmis(unittest.TestCase):
    def setUp(self) -> None:
        self.read = MappedReadInformation(
            utr_name='ENSMUSG00000025917',
            read_start_positions=[10024879, 10024880, 10024882, 10024888, 10024883] * 4,
            read_strands=['+'] * 20,

            # 2 unique barcodes, corresponding to 2 cells
            read_barcodes=['CGTTGGGCAGTCAGAG', 'CGTTGGGCAGTCAGAG',
                           'CGTTGGGCAGTCAGAG', 'CGTTGGGCAGTCAGAG',
                           'CGTTGGGCAGTCAGAG', 'CGTTGGGCAGTCAGAG',
                           'CGTTGGGCAGTCAGAG', 'CGTTGGGCAGTCAGAG',
                           'CGTTGGGCAGTCAGAG', 'CGTTGGGCAGTCAGAG',
                           'GTAGTCAAGCTATGCT', 'GTAGTCAAGCTATGCT',
                           'GTAGTCAAGCTATGCT', 'GTAGTCAAGCTATGCT',
                           'GTAGTCAAGCTATGCT', 'GTAGTCAAGCTATGCT',
                           'GTAGTCAAGCTATGCT', 'GTAGTCAAGCTATGCT',
                           'GTAGTCAAGCTATGCT', 'GTAGTCAAGCTATGCT'],

            # 2 sets of similar UMIs: 2 are the same, 1 is 1 hamming away, 1 is 2 hamming
            # extra couple of unique UMIs
            # repeated for the other barcode (cell)
            read_umis=['TGGCTTATAC', 'TGGCTTATAC',
                       'TGGCTgATAC', 'TGGtTTAaAC',
                       'ACCTCCGGTA', 'ACCTCCGGTA',
                       'ACCTCCGGTt', 'gCgTCCGGTA',
                       'AACGTAACAG', 'ACCCGCACTT'] * 2,
            read_lengths=[46, 50, 48, 46, 47] * 4,
            utr_sizes_from_reads=[98, 97, 95, 89, 94] * 4
        )

    def test_umis(self):
        filtered = self.read.get_unique_umis()
        self.assertListEqual(
            filtered.read_umis,
            ['TGGCTTATAC', 'TGGtTTAaAC', 'ACCTCCGGTA',
             'gCgTCCGGTA', 'AACGTAACAG', 'ACCCGCACTT',
             'TGGCTTATAC', 'TGGtTTAaAC', 'ACCTCCGGTA',
             'gCgTCCGGTA', 'AACGTAACAG', 'ACCCGCACTT'])

    def test_barcodes(self):
        filtered = self.read.get_unique_umis()
        self.assertEqual(len(set(filtered.read_barcodes)), 2)

    def test_sizes(self):
        filtered = self.read.get_unique_umis()
        results = [len(param) for param in filtered.todict().values()]
        expected = [18] + [12] * 6  # 18 is the length of the utr name
        self.assertListEqual(results, expected)


if __name__ == '__main__':
    unittest.main(exit=False)
    module_log.log_end_run("Finished testing")
