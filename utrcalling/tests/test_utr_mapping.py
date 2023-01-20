from __future__ import annotations
import os
from pathlib import Path
import unittest

import pandas as pd

# make sure file paths resolve correctly for the test case
os.chdir(str(Path(__file__).resolve().parents[0]))

from utrcalling.core import protocols, utr_mapping, module_log
from test_data import synthetic_data


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
        cls.alignment = './test_data/test_UTRoldNSCgenome_bwt2_102_sort.bam'

        # test data for 5'UTR
        cls.intervals_5prime = utr_mapping.generate_utr_intervals(
            utr_regions=utr_regions_5prime)
        cls.alignment_cap = './test_data/test_CapPhospho_genome_102_coordsort.bam'


class TestTenxTwo(TestMapReadsToUtr):
    def setUp(self) -> None:
        self.maxDiff = None
        protocol = protocols.protocol(platform='10x', version=2,
                                      annotations=pd.read_csv("./test_data/vivo_anno.csv"))
        self.mappings = utr_mapping.map_reads_to_utr(self.alignment, self.intervals,
                                                     protocol, read='first')

    def test_details(self):
        dict_true = {
            'read_start_positions': [122748098, 122748088, 122748096],
            'read_strands': ['-', '-', '-'],
            'read_barcodes': ['GATGAAACATTAACCG', 'TGCGCAGAGTCCATAC', 'GCTGGGTGTACGCACC'],
            'read_umis': ['TCAACACAAG', 'AATCGAGAGT', 'CCTGAGTATG'],
            'read_lengths': [51, 37, 44],
            'utr_sizes_from_reads': [1468, 1458, 1466]
        }
        self.assertDictEqual(
            self.mappings.loc["ENSMUSG00000005804"].to_dict(), dict_true)

    def test_length(self):
        self.assertEqual(self.mappings.shape[0], 110)


class TestTenxThree(TestMapReadsToUtr):
    def setUp(self) -> None:
        self.maxDiff = None
        protocol = protocols.protocol(platform='10x', version=3,
                                      annotations=pd.read_csv("./test_data/vivo_anno.csv"))
        self.mappings = utr_mapping.map_reads_to_utr(self.alignment, self.intervals,
                                                     protocol, read='first')

    def test_details(self):
        dict_true = {
            'read_start_positions': [122748098, 122748088, 122748096],
            'read_strands': ['-', '-', '-'],
            'read_barcodes': ['GATGAAACATTAACCG', 'TGCGCAGAGTCCATAC', 'GCTGGGTGTACGCACC'],
            'read_umis': ['TCAACACAAGTT', 'AATCGAGAGTTT', 'CCTGAGTATGTT'],
            'read_lengths': [51, 37, 44],
            'utr_sizes_from_reads': [1468, 1458, 1466]
        }
        self.assertDictEqual(
            self.mappings.loc["ENSMUSG00000005804"].to_dict(), dict_true)

    def test_length(self):
        self.assertEqual(self.mappings.shape[0], 110)


class TestSmartSeq3(TestMapReadsToUtr):
    def setUp(self) -> None:
        protocol = protocols.protocol(platform='smartseq', version=3, label_batches=True,
                                      annotations="./test_data/smartseq3_anno.csv")
        self.mappings = utr_mapping.map_reads_to_utr(self.alignment, self.intervals,
                                                     protocol, read='first')

    def test_details(self):
        dict_true = {
            'read_start_positions': [122748098, 122748088, 122748096],
            'read_strands': ['-', '-', '-'],
            'read_barcodes': ['GATGAA-2', 'TGCGCA-2', 'GCTGGG-2'],
            'read_umis': ['ACATTAAC', 'GAGTCCAT', 'TGTACGCA'],
            'read_lengths': [51, 37, 44],
            'utr_sizes_from_reads': [1468, 1458, 1466]
        }
        self.assertDictEqual(
            self.mappings.loc["ENSMUSG00000005804"].to_dict(), dict_true)

    def test_length(self):
        self.assertEqual(self.mappings.shape[0], 110)


class TestSmartSeq3Cap(TestMapReadsToUtr):
    def setUp(self) -> None:
        protocol = protocols.protocol(platform='smart-seq_cap', version=3)
        self.mappings = utr_mapping.map_reads_to_utr(self.alignment_cap,
                                                     self.intervals_5prime,
                                                     protocol, read='first',
                                                     batch="test_sample")

    def test_details(self):
        dict_true = {
            'read_start_positions': [23677423, 23677444, 23677448],
            'read_strands': ['-', '-', '-'],
            'read_barcodes': ['test_sample', 'test_sample', 'test_sample'],
            'read_umis': ['CGCCAGGC', 'GACACGAT', 'AAGGGGAC'],
            'read_lengths': [81, 82, 82],
            'utr_sizes_from_reads': [4, 25, 29]
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


class TestMapReadsToUtrMinimal(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        utr_regions = {"utr_name": ["UTR1", "UTR1-1", "UTR2", "UTR3", "UTR4"],
                       "chrom": [1, 1, 1, 1, 1],
                       "start": [14190, 16400, 184700, 628000, 1200000],
                       "end": [14500, 16600, 184900, 630000, 1200500],
                       "name": ["UTR1", "UTR1", "UTR2", "UTR3", "UTR4"],
                       "score": [0.0] * 5,
                       "strand": ["+", "-", "-", "+", "-"],
                       "transcript_id": ["UTR1_t", "UTR1_t", "UTR2_t", "UTR3_t", "UTR4_t"],
                       "gene_name": ["Gene1", "Gene1", "Gene1", "Gene2", "Gene3"]
                       }
        utr_regions = pd.DataFrame(utr_regions).set_index("utr_name")
        cls.intervals = utr_mapping.generate_utr_intervals(
            utr_regions=utr_regions)
        cls.alignment = './test_data/smartseq3_test_sorted.bam'


class TestSmartSeq3Minimal(TestMapReadsToUtrMinimal):
    def setUp(self) -> None:
        protocol = protocols.protocol(platform='smartseq', version=3)
        self.mappings = utr_mapping.map_reads_to_utr(self.alignment, self.intervals,
                                                     protocol, read='first')

    def test_details(self):
        dict_true = {
            'read_start_positions': [[14412], [16441]],
            'read_strands': [['-'], ['+']],
            'read_barcodes': [['CACTGT'], ['CACTGT']],
            'read_umis': [['GTGATATG'], ['GGCCACCA']],
            'read_lengths': [[97], [93]],
            'utr_sizes_from_reads': [[223], [158]]
        }
        self.assertDictEqual(
            self.mappings.to_dict(orient="list"), dict_true)

    def test_length(self):
        self.assertEqual(self.mappings.shape[0], 2)


class TestMapReadsToUtrMinimalCap(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        utr_regions = {"utr_name": ["UTR1", "UTR2", "UTR2-1", "UTR3", "UTR4"],
                       "chrom": [1, 1, 1, 1, 1],
                       "start": [3102600, 3152900, 3154500, 4580000, 4846300],
                       "end": [3102750, 3153050, 3154600, 4760000, 4846300],
                       "name": ["UTR1", "UTR2", "UTR2", "UTR3", "UTR4"],
                       "score": [0.0] * 5,
                       "strand": ["+", "-", "+", "+", "+"],
                       "transcript_id": ["UTR1_t", "UTR2_t", "UTR2_t", "UTR3_t", "UTR4_t"],
                       "gene_name": ["Gene1", "Gene1", "Gene1", "Gene2", "Gene3"]
                       }
        utr_regions = pd.DataFrame(utr_regions).set_index("utr_name")
        cls.intervals = utr_mapping.generate_utr_intervals(
            utr_regions=utr_regions)
        cls.alignment = './test_data/capseq_test_sorted.bam'


class TestSmartSeq3CapReadnameMinimal(TestMapReadsToUtrMinimalCap):
    def setUp(self) -> None:
        self.maxDiff = None
        protocol = protocols.protocol(platform='smart-seq_cap-readname', version=3)
        self.mappings = utr_mapping.map_reads_to_utr(self.alignment, self.intervals,
                                                     protocol, read='first',
                                                     batch="Test")

    def test_details(self):
        dict_true = {
            'read_start_positions': [[3102682], [3153041]],
            'read_strands': [['+'], ['-']],
            'read_barcodes': [['Test'], ['Test']],
            'read_umis': [['TTGCCACG'], ['TTCAGACG']],
            'read_lengths': [[80], [81]],
            'utr_sizes_from_reads': [[68], [143]]
        }
        self.assertDictEqual(
            self.mappings.to_dict(orient="list"), dict_true)

    def test_length(self):
        self.assertEqual(self.mappings.shape[0], 2)


class TestSmartSeq3LabelSamplesMinimal(TestMapReadsToUtrMinimal):
    def setUp(self) -> None:
        protocol = protocols.protocol(platform='smartseq', version=3, label_batches=True,
                                      annotations="./test_data/smartseq3_anno.csv")
        self.mappings = utr_mapping.map_reads_to_utr(self.alignment, self.intervals,
                                                     protocol, read='first')

    def test_details(self):
        dict_true = {
            'read_start_positions': [[14412], [16441]],
            'read_strands': [['-'], ['+']],
            'read_barcodes': [['CACTGT-3'], ['CACTGT-3']],
            'read_umis': [['GTGATATG'], ['GGCCACCA']],
            'read_lengths': [[97], [93]],
            'utr_sizes_from_reads': [[223], [158]]
        }
        self.assertDictEqual(
            self.mappings.to_dict(orient="list"), dict_true)

    def test_length(self):
        self.assertEqual(self.mappings.shape[0], 2)


class TestMapReadsToUtrSyntheticCap(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        utr_regions = {"utr_name": ["UTR1", "UTR2", "UTR2-1", "UTR3", "UTR4"],
                       "chrom": [1, 1, 1, 1, 1],
                       "start": [3102600, 3340260, 3154500, 3536770, 4846300],
                       "end": [3102750, 3340670, 3154600, 3536840, 4846300],
                       "name": ["UTR1", "UTR2", "UTR2", "UTR3", "UTR4"],
                       "score": [0.0] * 5,
                       "strand": ["+", "-", "+", "+", "+"],
                       "transcript_id": ["UTR1_t", "UTR2_t", "UTR2_t", "UTR3_t", "UTR4_t"],
                       "gene_name": ["Gene1", "Gene1", "Gene1", "Gene2", "Gene3"]
                       }
        utr_regions = pd.DataFrame(utr_regions).set_index("utr_name")
        cls.intervals = utr_mapping.generate_utr_intervals(
            utr_regions=utr_regions)
        cls.data = synthetic_data.SmartSeqCapSeqBam()


class TestSmartSeqCapPhosphoSynthetic(TestMapReadsToUtrSyntheticCap):
    def setUp(self) -> None:
        self.maxDiff = None
        protocol = protocols.protocol(platform='smart-seq_cap', version=3)
        with self.data:
            self.mappings = utr_mapping.map_reads_to_utr(self.data.alignment, self.intervals,
                                                         protocol, read='first',
                                                         batch="Test")
            print(self.mappings)

    def test_details(self):
        dict_true = {
            'read_start_positions': [[3340662], [3536771]],
            'read_strands': [['-'], ['+']],
            'read_barcodes': [['Test'], ['Test']],
            'read_umis': [['TCTGCGTG'], ['TTGGCAGG']],
            'read_lengths': [[82], [82]],
            'utr_sizes_from_reads': [[404], [69]]
        }
        self.assertDictEqual(
            self.mappings.to_dict(orient="list"), dict_true)

    def test_length(self):
        self.assertEqual(self.mappings.shape[0], 2)


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2, buffer=True)
    module_log.log_end_run("Finished testing")
