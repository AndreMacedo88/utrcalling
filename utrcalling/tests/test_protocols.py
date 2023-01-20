import os
from pathlib import Path
import unittest
import copy

import pandas as pd
from utrcalling.core import peak_calling, protocols, utr_mapping, module_log

# make sure file paths resolve correctly for the test case
os.chdir(str(Path(__file__).resolve().parents[0]))


class Test10x(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        utr_regions = pd.read_pickle('./test_data/test_UTR_reference.pkl')
        intervals = utr_mapping.generate_utr_intervals(utr_regions=utr_regions)
        alignment_1 = './test_data/test_UTRoldNSCgenome_bwt2_102_sort.bam'
        alignment_2 = './test_data/test_UTRyoungNSCgenome_bwt2_102_sort.bam'
        cls.protocol = protocols.protocol(platform='10x', version=2,
                                          merge_method="intersection")
        cls.mappings = [
            utr_mapping.map_reads_to_utr(alignment, intervals, cls.protocol,
                                         read='first') for alignment in [alignment_1,
                                                                         alignment_2]
        ]


class TestFilterUtrSizes(Test10x):
    def setUp(self) -> None:
        merged = peak_calling.merge_samples_utr_sizes(copy.deepcopy(self.mappings),
                                                      method=self.protocol.merge_method)

        # modify the data to check for successfull filtering.
        # Original was [772, 756, 751, 753, 811, 751].
        merged.at["ENSMUSG00000001036", "utr_sizes_from_reads"] = [1, 798, 1, 798, 1, 798]
        self.filtered = self.protocol.filter_utr_sizes(merged, threshold=1)

    def test_number_of_features(self) -> None:
        self.assertEqual(self.filtered.shape, (63, 6))

    def test_is_filtered(self):
        dict_true = {
            'read_start_positions': [61518618, 61518618, 61518618],
            'read_strands': ['+', '+', '+'],
            'read_barcodes': ['CTAATGGCATTCGACA-1',
                              'CCCAGTTTCGCTTGTC-1',
                              'TATTACCAGGATTCGG-2'],
            'read_umis': ['AACCGAGCGT', 'AATGGTAACG', 'AGCGATAGGA'],
            'read_lengths': [59, 52, 52],
            'utr_sizes_from_reads': [798, 798, 798]
        }
        self.assertDictEqual(
            self.filtered.loc["ENSMUSG00000001036"].to_dict(), dict_true)


class TestSmartSeqCapPhosphoReadname(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        utr_regions = pd.read_pickle('./test_data/test_UTR_reference_5prime.pkl')
        intervals = utr_mapping.generate_utr_intervals(utr_regions=utr_regions)
        alignment = './test_data/test_CapPhospho_genome_102_coordsort.bam'
        cls.protocol = protocols.protocol(platform='smart-seq_cap-readname', version=3)
        cls.mappings = utr_mapping.map_reads_to_utr(
            alignment, intervals, cls.protocol, read='first', batch="test_sample")


class TestFilterUtrSizesSmartSeqCap(TestSmartSeqCapPhosphoReadname):
    def setUp(self) -> None:
        # modify the data to check for successfull filtering. Original was [44, 23, 19].
        self.mappings.loc["ENSMUSG00000023905"].utr_sizes_from_reads = [44, 1, 19]
        self.filtered = self.protocol.filter_utr_sizes(self.mappings)

    def test_number_of_features(self) -> None:
        # For 5'UTRs we don't want filtering by default
        self.assertEqual(len(self.filtered), 31)

    def test_is_filtered(self):
        # For 5'UTRs we don't want filtering by default
        dict_true = {
            'read_start_positions': [23677423, 23677444, 23677448],
            'read_strands': ['-', '-', '-'],
            'read_barcodes': ['test_sample', 'test_sample', 'test_sample'],
            'read_umis': ['CGCCAGGC', 'GACACGAT', 'AAGGGGAC'],
            'read_lengths': [81, 82, 82],
            'utr_sizes_from_reads': [44, 1, 19]
        }
        self.assertDictEqual(
            self.filtered.loc["ENSMUSG00000023905"].to_dict(), dict_true)


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2, buffer=True)
    module_log.log_end_run("Finished testing")
