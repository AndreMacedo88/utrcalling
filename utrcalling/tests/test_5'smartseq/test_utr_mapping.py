import os
from pathlib import Path
import unittest
import pandas as pd
from utrcalling.tests.test_data import synthetic_data_5smartseq
from utrcalling.core import protocol, utr_mapping, module_log

# we change working dir to the base test dir so that relative paths are sane
os.chdir(str(Path(__file__).resolve().parents[1]))


class TestMapReadsToUtr(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.utr_regions = pd.read_parquet(
            "./test_data/test_UTR_reference_5prime.parquet"
        )

        # test data for 5'smartseq
        cls.alignment = "./test_data/test_CapPhospho_genome_102_coordsort.bam"


class TestSmartSeq3(TestMapReadsToUtr):
    def setUp(self) -> None:
        self.intervals_5prime = utr_mapping.generate_utr_intervals(
            utr_regions=self.utr_regions
        )
        protocol_ = protocol.create(
            platform="smart-seq_cap-readname",
            version=3,
            alignments=self.alignment,
            annotations="./test_data/smartseq3_anno.csv",
        )
        self.mappings = utr_mapping.map_reads_to_utr(
            self.alignment,
            self.intervals_5prime,
            protocol_,
            read="first",
            batch="test_sample",
        )

    def test_details(self):
        dict_true = {
            "read_barcodes": [
                self.alignment,
                self.alignment,
                self.alignment,
            ],
            "utr_sizes_from_reads": [4, 25, 29],
        }
        self.assertDictEqual(
            self.mappings.loc["ENSMUSG00000023905"].to_dict(), dict_true
        )

    def test_length(self):
        self.assertEqual(self.mappings.shape[0], 31)


class TestMapReadsToUtrMinimal(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        utr_regions = {
            "utr_name": ["UTR1", "UTR2", "UTR2-1", "UTR3", "UTR4"],
            "chrom": [1, 1, 1, 1, 1],
            "start": [3102600, 3152900, 3154500, 4580000, 4846300],
            "end": [3102750, 3153050, 3154600, 4760000, 4846300],
            "name": ["UTR1", "UTR2", "UTR2", "UTR3", "UTR4"],
            "score": [0.0] * 5,
            "strand": ["+", "-", "+", "+", "+"],
            "transcript_id": ["UTR1_t", "UTR2_t", "UTR2_t", "UTR3_t", "UTR4_t"],
            "gene_name": ["Gene1", "Gene1", "Gene1", "Gene2", "Gene3"],
        }
        utr_regions = pd.DataFrame(utr_regions).set_index("utr_name")
        cls.intervals = utr_mapping.generate_utr_intervals(utr_regions=utr_regions)
        cls.alignment = "./test_data/capseq_test_sorted.bam"


class TestSmartSeq3ReadnameMinimal(TestMapReadsToUtrMinimal):
    def setUp(self) -> None:
        self.maxDiff = None
        protocol_ = protocol.create(
            platform="smart-seq_cap-readname",
            version=3,
            alignments=self.alignment,
            annotations="./test_data/smartseq3_anno.csv",
        )
        self.mappings = utr_mapping.map_reads_to_utr(
            self.alignment, self.intervals, protocol_, read="first", batch="Test"
        )

    def test_details(self):
        dict_true = {
            "read_barcodes": [[self.alignment], [self.alignment]],
            "utr_sizes_from_reads": [[68], [143]],
        }
        self.assertDictEqual(self.mappings.to_dict(orient="list"), dict_true)

    def test_length(self):
        self.assertEqual(self.mappings.shape[0], 2)


class TestMapReadsToUtrSynthetic(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        utr_regions = {
            "utr_name": ["UTR1", "UTR2", "UTR2-1", "UTR3", "UTR4"],
            "chrom": [1, 1, 1, 1, 1],
            "start": [3102600, 3340260, 3154500, 3536770, 4846300],
            "end": [3102750, 3340670, 3154600, 3536840, 4846300],
            "name": ["UTR1", "UTR2", "UTR2", "UTR3", "UTR4"],
            "score": [0.0] * 5,
            "strand": ["+", "-", "+", "+", "+"],
            "transcript_id": ["UTR1_t", "UTR2_t", "UTR2_t", "UTR3_t", "UTR4_t"],
            "gene_name": ["Gene1", "Gene1", "Gene1", "Gene2", "Gene3"],
        }
        utr_regions = pd.DataFrame(utr_regions).set_index("utr_name")
        cls.intervals = utr_mapping.generate_utr_intervals(utr_regions=utr_regions)
        cls.data = synthetic_data_5smartseq.Bam()


class TestSmartSeqPhosphoSynthetic(TestMapReadsToUtrSynthetic):
    def setUp(self) -> None:
        self.maxDiff = None
        protocol_ = protocol.create(
            platform="smart-seq_cap",
            version=3,
            alignments=self.data.alignment,
            annotations="./test_data/smartseq3_anno.csv",
        )
        with self.data:
            self.mappings = utr_mapping.map_reads_to_utr(
                self.data.alignment,
                self.intervals,
                protocol_,
                read="first",
                batch="Test",
            )

    def test_details(self):
        dict_true = {
            "read_barcodes": [[self.data.alignment], [self.data.alignment]],
            "utr_sizes_from_reads": [[404], [69]],
        }
        self.assertDictEqual(self.mappings.to_dict(orient="list"), dict_true)

    def test_length(self):
        self.assertEqual(self.mappings.shape[0], 2)


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2, buffer=True)
    module_log.log_end_run("Finished testing")
