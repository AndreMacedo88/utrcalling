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
        )

    def test_details(self):
        utr_data = self.mappings[
            self.mappings["utr_name"] == "ENSMUSG00000023905"
        ]
        self.assertListEqual(
            utr_data["observation_id"].tolist(),
            [self.alignment, self.alignment, self.alignment],
        )
        self.assertListEqual(
            utr_data["utr_size"].tolist(),
            [4, 25, 29],
        )

    def test_length(self):
        self.assertEqual(len(self.mappings["utr_name"].unique()), 31)


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
            self.alignment, self.intervals, protocol_, read="first"
        )

    def test_details(self):
        utrs = sorted(self.mappings["utr_name"].unique().tolist())
        obs_per_utr = []
        sizes_per_utr = []
        for utr in utrs:
            utr_data = self.mappings[self.mappings["utr_name"] == utr]
            obs_per_utr.append(utr_data["observation_id"].tolist())
            sizes_per_utr.append(utr_data["utr_size"].tolist())
        self.assertListEqual(obs_per_utr, [[self.alignment], [self.alignment]])
        self.assertListEqual(sizes_per_utr, [[68], [143]])

    def test_length(self):
        self.assertEqual(len(self.mappings["utr_name"].unique()), 2)


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
            )

    def test_details(self):
        utrs = sorted(self.mappings["utr_name"].unique().tolist())
        obs_per_utr = []
        sizes_per_utr = []
        for utr in utrs:
            utr_data = self.mappings[self.mappings["utr_name"] == utr]
            obs_per_utr.append(utr_data["observation_id"].tolist())
            sizes_per_utr.append(utr_data["utr_size"].tolist())
        self.assertListEqual(
            obs_per_utr, [[self.data.alignment], [self.data.alignment]]
        )
        self.assertListEqual(sizes_per_utr, [[404], [69]])

    def test_length(self):
        self.assertEqual(len(self.mappings["utr_name"].unique()), 2)


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2, buffer=True)
    module_log.log_end_run("Finished testing")
