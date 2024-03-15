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
        )

    def test_details(self):
        dict_true = {
            "read_barcodes": [["CACTGT-2"], ["CACTGT-2"]],
            "utr_sizes_from_reads": [[300], [30]],
        }
        self.assertDictEqual(self.mappings.to_dict(orient="list"), dict_true)

    def test_details_one(self):
        dict_true = {"read_barcodes": ["CACTGT-2"], "utr_sizes_from_reads": [300]}
        self.assertDictEqual(self.mappings.loc["ENSG00000188976"].to_dict(), dict_true)

    def test_length(self):
        self.assertEqual(self.mappings.shape[0], 2)


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
        )

    def test_details(self):
        dict_true = {
            "read_barcodes": [["CACTGT-2"], ["CACTGT-2"]],
            "utr_sizes_from_reads": [[300], [30]],
        }
        self.assertDictEqual(self.mappings.to_dict(orient="list"), dict_true)

    def test_details_one(self):
        dict_true = {"read_barcodes": ["CACTGT-2"], "utr_sizes_from_reads": [300]}
        self.assertDictEqual(self.mappings.loc["ENSG00000188976"].to_dict(), dict_true)

    def test_length(self):
        self.assertEqual(self.mappings.shape[0], 2)


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
        dict_true = {
            "read_barcodes": [["CACTGT"], ["CACTGT"]],
            "utr_sizes_from_reads": [[223], [158]],
        }
        self.assertDictEqual(self.mappings.to_dict(orient="list"), dict_true)

    def test_length(self):
        self.assertEqual(self.mappings.shape[0], 2)


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
            self.alignment, self.intervals, protocol_, read="first"
        )

    def test_details(self):
        dict_true = {
            "read_barcodes": [["CACTGT-2"], ["CACTGT-2"]],
            "utr_sizes_from_reads": [[223], [158]],
        }
        self.assertDictEqual(self.mappings.to_dict(orient="list"), dict_true)

    def test_length(self):
        self.assertEqual(self.mappings.shape[0], 2)


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2, buffer=True)
    module_log.log_end_run("Finished testing")
