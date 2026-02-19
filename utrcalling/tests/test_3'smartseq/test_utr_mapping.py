import os
from pathlib import Path
import unittest
import pandas as pd
from utrcalling.core import protocol, utr_mapping, module_log

# we change working dir to the base test dir so that relative paths are sane
os.chdir(str(Path(__file__).resolve().parents[1]))


class TestMapReadsToUtr(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.utr_regions = pd.read_parquet(
            "./test_data/test_UTR_reference_smartseq.parquet"
        )

        # test data for SmartSeq3
        cls.alignment = "./test_data/smartseq3_test_sorted_AS-740066-LR-59119.bam"


class TestSmartSeq3(TestMapReadsToUtr):
    def setUp(self) -> None:
        # test data for SmartSeq3
        self.intervals = utr_mapping.generate_utr_intervals(
            utr_regions=self.utr_regions
        )
        protocol_ = protocol.create(
            platform="smartseq",
            version=3,
            label_batches=True,
            alignments=self.alignment,
            annotations="./test_data/smartseq3_anno.csv",
        )
        self.mappings = utr_mapping.map_reads_to_utr(
            self.alignment,
            self.intervals,
            protocol_,
            read="first",
            batch_idx=protocol_.sample_idx[self.alignment],
        )

    def test_details(self):
        # Check that all observations and sizes are present (order-independent per UTR)
        all_obs = sorted(self.mappings["observation_id"].tolist())
        all_sizes = sorted(self.mappings["utr_size"].tolist())
        self.assertListEqual(all_obs, ["CACTGT-2", "CACTGT-2"])
        self.assertListEqual(all_sizes, [30, 300])

    def test_details_one(self):
        utr_data = self.mappings[
            self.mappings["utr_name"] == "ENSG00000188976"
        ]
        self.assertListEqual(utr_data["observation_id"].tolist(), ["CACTGT-2"])
        self.assertListEqual(utr_data["utr_size"].tolist(), [300])

    def test_length(self):
        self.assertEqual(len(self.mappings["utr_name"].unique()), 2)


class TestSmartSeq3NoBarcodes(TestMapReadsToUtr):
    def setUp(self) -> None:
        # test data for SmartSeq3
        self.intervals = utr_mapping.generate_utr_intervals(
            utr_regions=self.utr_regions
        )
        annotations = pd.read_csv("./test_data/smartseq3_anno.csv")
        annotations = annotations.drop("Sample_Barcode", axis=1)
        protocol_ = protocol.create(
            platform="smartseq",
            version=3,
            label_batches=True,
            alignments=self.alignment,
            annotations=annotations,
        )
        self.mappings = utr_mapping.map_reads_to_utr(
            self.alignment,
            self.intervals,
            protocol_,
            read="first",
            batch_idx=protocol_.sample_idx[self.alignment],
        )

    def test_details(self):
        all_obs = sorted(self.mappings["observation_id"].tolist())
        all_sizes = sorted(self.mappings["utr_size"].tolist())
        self.assertListEqual(all_obs, ["CACTGT-2", "CACTGT-2"])
        self.assertListEqual(all_sizes, [30, 300])

    def test_details_one(self):
        utr_data = self.mappings[
            self.mappings["utr_name"] == "ENSG00000188976"
        ]
        self.assertListEqual(utr_data["observation_id"].tolist(), ["CACTGT-2"])
        self.assertListEqual(utr_data["utr_size"].tolist(), [300])

    def test_length(self):
        self.assertEqual(len(self.mappings["utr_name"].unique()), 2)


class TestMapReadsToUtrMinimal(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        utr_regions = {
            "utr_name": ["UTR1", "UTR1-1", "UTR2", "UTR3", "UTR4"],
            "chrom": [1, 1, 1, 1, 1],
            "start": [14190, 16400, 184700, 628000, 1200000],
            "end": [14500, 16600, 184900, 630000, 1200500],
            "name": ["UTR1", "UTR1", "UTR2", "UTR3", "UTR4"],
            "score": [0.0] * 5,
            "strand": ["+", "-", "-", "+", "-"],
            "transcript_id": ["UTR1_t", "UTR1_t", "UTR2_t", "UTR3_t", "UTR4_t"],
            "gene_name": ["Gene1", "Gene1", "Gene1", "Gene2", "Gene3"],
        }
        utr_regions = pd.DataFrame(utr_regions).set_index("utr_name")
        cls.intervals = utr_mapping.generate_utr_intervals(utr_regions=utr_regions)
        cls.alignment = "./test_data/smartseq3_test_sorted_AS-740066-LR-59119.bam"


class TestSmartSeq3Minimal(TestMapReadsToUtrMinimal):
    def setUp(self) -> None:
        protocol_ = protocol.create(
            platform="smartseq",
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
        self.assertListEqual(obs_per_utr, [["CACTGT"], ["CACTGT"]])
        self.assertListEqual(sizes_per_utr, [[223], [158]])

    def test_length(self):
        self.assertEqual(len(self.mappings["utr_name"].unique()), 2)


class TestSmartSeq3LabelSamplesMinimal(TestMapReadsToUtrMinimal):
    def setUp(self) -> None:
        protocol_ = protocol.create(
            platform="smartseq",
            version=3,
            label_batches=True,
            alignments=self.alignment,
            annotations="./test_data/smartseq3_anno.csv",
        )
        self.mappings = utr_mapping.map_reads_to_utr(
            self.alignment, self.intervals, protocol_, read="first",
            batch_idx=protocol_.sample_idx[self.alignment],
        )

    def test_details(self):
        utrs = sorted(self.mappings["utr_name"].unique().tolist())
        obs_per_utr = []
        sizes_per_utr = []
        for utr in utrs:
            utr_data = self.mappings[self.mappings["utr_name"] == utr]
            obs_per_utr.append(utr_data["observation_id"].tolist())
            sizes_per_utr.append(utr_data["utr_size"].tolist())
        self.assertListEqual(obs_per_utr, [["CACTGT-2"], ["CACTGT-2"]])
        self.assertListEqual(sizes_per_utr, [[223], [158]])

    def test_length(self):
        self.assertEqual(len(self.mappings["utr_name"].unique()), 2)


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2, buffer=True)
    module_log.log_end_run("Finished testing")
