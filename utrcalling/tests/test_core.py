import os
from pathlib import Path
import unittest

import pandas as pd

# make sure file paths resolve correctly for the test case
os.chdir(str(Path(__file__).resolve().parents[0]))

from utrcalling.core import core, module_log, protocols


class TestMapAlignedReads(unittest.TestCase):
    def setUp(self) -> None:

        utr_regions = pd.read_pickle('./test_data/test_UTR_reference.pkl')
        protocol = protocols.protocol(platform='10x', version=2)

        self.alignments_n = 5
        alignments = [
            './test_data/test_UTRoldNSCgenome_bwt2_102_duplicates_sort.bam',
        ] * self.alignments_n

        self.read_mappings = core.map_aligned_reads(
            alignments, utr_regions, protocol, store=False
        )

        self.true_results_5804 = {
            'read_start_positions': [122748098, 122748087, 122748096],
            'read_strands': ['-', '-', '-'],
            'read_barcodes': ['GATGAAACATTAACCG', 'TGCGCAGAGTCCATAC', 'GCTGGGTGTACGCACC'],
            'read_umis': ['TCAACACAAG', 'AATCGAGAGT', 'CCTGAGTATG'],
            'read_lengths': [51, 37, 44],
            'utr_sizes_from_reads': [1468, 1457, 1466]
        }

    def expect_dict_equal(self, i, first, second, msg=None):
        with self.subTest(i=i):
            self.assertDictEqual(first, second, msg)

    def test_length(self):
        self.assertEqual(len(self.read_mappings), self.alignments_n)

    def test_read_mappings(self):
        for i in range(self.alignments_n):
            self.expect_dict_equal(
                i,
                self.read_mappings[i].loc["ENSMUSG00000005804"].to_dict(),
                self.true_results_5804
            )


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2, buffer=True)
    module_log.log_end_run("Finished testing")
