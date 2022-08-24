from __future__ import annotations
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
            './test_data/test_UTRoldNSCgenome_bwt2_102_sort.bam',
        ] * self.alignments_n

        self.read_mappings = core.map_aligned_reads(
            alignments, utr_regions, protocol, store=False
        )

        self.true_results_5804 = {
            'read_start_positions': [122748098, 122748088, 122748096],
            'read_strands': ['-', '-', '-'],
            'read_barcodes': ['GATGAAACATTAACCG', 'TGCGCAGAGTCCATAC', 'GCTGGGTGTACGCACC'],
            'read_umis': ['TCAACACAAG', 'AATCGAGAGT', 'CCTGAGTATG'],
            'read_lengths': [51, 37, 44],
            'utr_sizes_from_reads': [1468, 1458, 1466]
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


class TestCallPeaksSmartseq3Minimal(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.annotations = "./test_data/smartseq3_anno.csv"
        cls.utr_regions = {"utr_name": ["UTR1", "UTR1-1", "UTR2", "UTR3", "UTR4"],
                            "chrom": [1, 1, 1, 1, 1],
                            "start": [14190, 16400, 184700, 628000, 1200000],
                            "end": [14500, 16600, 184900, 630000, 1200500],
                            "name": ["UTR1", "UTR1", "UTR2", "UTR3", "UTR4"],
                            "score": [0.0] * 5,
                            "strand": ["+", "-", "-", "+", "-"],
                            "transcript_id": ["UTR1_t", "UTR1_t", "UTR2_t", "UTR3_t", "UTR4_t"],
                            "gene_name": ["Gene1", "Gene1", "Gene1", "Gene2", "Gene3"]
                            }
        cls.utr_regions = pd.DataFrame(cls.utr_regions).set_index("utr_name")
        cls.protocol = protocols.protocol(platform='smartseq', version=3,
                                           label_batches=True,
                                           annotations=cls.annotations)
        cls.alignments = ['./test_data/smartseq3_test_sorted.bam']
        cls.mapped = core.map_aligned_reads(
            alignments=cls.alignments, utr_regions=cls.utr_regions,
            protocol=cls.protocol, store=False)


class TestCallPeaksSmartseq3LabelMinimal(TestCallPeaksSmartseq3Minimal):
    def setUp(self) -> None:
        self.results = core.call_peaks(
            utr_sizes_counts=self.mapped,
            utr_reference_regions=self.utr_regions,
            protocol=self.protocol,
            annotations=self.annotations,
            store=False
        )

    def test_sad(self):
        pass


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2, buffer=False)
    module_log.log_end_run("Finished testing")
