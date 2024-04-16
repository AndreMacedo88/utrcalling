import os
from pathlib import Path
import unittest
import pandas as pd
from utrcalling.tests.test_data import synthetic_data_3tenx
from utrcalling.core import protocol, utr_mapping, module_log

# we change working dir to the base test dir so that relative paths are sane
os.chdir(str(Path(__file__).resolve().parents[1]))


class TestMapReadsToUtr(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.utr_regions = pd.read_parquet("./test_data/test_UTR_reference.parquet")

        # test data for 3'10X
        cls.alignment = "./test_data/test_UTRoldNSCgenome_bwt2_102_sort.bam"


class TestTenxTwo(TestMapReadsToUtr):
    def setUp(self) -> None:
        self.maxDiff = None
        self.intervals = utr_mapping.generate_utr_intervals(
            utr_regions=self.utr_regions
        )
        protocol_ = protocol.create(
            platform="10x",
            version=2,
            annotations=pd.read_csv("./test_data/vivo_anno.csv"),
        )
        self.mappings = utr_mapping.map_reads_to_utr(
            self.alignment, self.intervals, protocol_, read="first"
        )

    def test_details(self):
        dict_true = {
            "read_barcodes": [
                "GATGAAACATTAACCG",
                "TGCGCAGAGTCCATAC",
                "GCTGGGTGTACGCACC",
            ],
            "utr_sizes_from_reads": [1468, 1458, 1466],
        }
        self.assertDictEqual(
            self.mappings.loc["ENSMUSG00000005804"].to_dict(), dict_true
        )

    def test_length(self):
        self.assertEqual(self.mappings.shape[0], 110)


class TestTenxThree(TestMapReadsToUtr):
    def setUp(self) -> None:
        self.maxDiff = None
        self.intervals = utr_mapping.generate_utr_intervals(
            utr_regions=self.utr_regions
        )
        protocol_ = protocol.create(
            platform="10x",
            version=3,
            annotations=pd.read_csv("./test_data/vivo_anno.csv"),
        )
        self.mappings = utr_mapping.map_reads_to_utr(
            self.alignment, self.intervals, protocol_, read="first"
        )

    def test_details(self):
        dict_true = {
            "read_barcodes": [
                "GATGAAACATTAACCG",
                "TGCGCAGAGTCCATAC",
                "GCTGGGTGTACGCACC",
            ],
            "utr_sizes_from_reads": [1468, 1458, 1466],
        }
        self.assertDictEqual(
            self.mappings.loc["ENSMUSG00000005804"].to_dict(), dict_true
        )

    def test_length(self):
        self.assertEqual(self.mappings.shape[0], 110)


class TestTenxThreeNoBarcodes(TestMapReadsToUtr):
    def setUp(self) -> None:
        self.maxDiff = None
        self.intervals = utr_mapping.generate_utr_intervals(
            utr_regions=self.utr_regions
        )
        annotations = pd.read_csv("./test_data/vivo_anno.csv")
        annotations = annotations.drop("CellBarcode", axis=1)

        protocol_ = protocol.create(platform="10x", version=3, annotations=annotations)
        self.mappings = utr_mapping.map_reads_to_utr(
            self.alignment, self.intervals, protocol_, read="first"
        )

    def test_details(self):
        dict_true = {
            "read_barcodes": [
                "GATGAAACATTAACCG",
                "TGCGCAGAGTCCATAC",
                "GCTGGGTGTACGCACC",
            ],
            "utr_sizes_from_reads": [1468, 1458, 1466],
        }
        self.assertDictEqual(
            self.mappings.loc["ENSMUSG00000005804"].to_dict(), dict_true
        )

    def test_length(self):
        self.assertEqual(self.mappings.shape[0], 110)


class TestMapReadsToUtrSynthetic(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        utr_regions = {
            "utr_name": ["UTR1", "UTR2", "UTR3"],  # UTR that has no reads mapping to it
            "chrom": [
                1,
                1,
                16,  # to test alternative chr
            ],
            "start": [
                3102600,
                172196728,  # Read 1 is inside the UTR by 0bp
                91270659,  # Read 2 is inside the UTR
            ],
            "end": [
                3102750,
                172196828,  # Read 2 outside the UTR
                91271933,  # Read 1 is inside the UTR by 13bp
            ],
            "name": ["UTR1", "UTR2", "UTR3"],
            "strand": [
                "+",
                "-",
                "+",
            ],
            "transcript_id": ["UTR1_t", "UTR2_t", "UTR3_t"],
            "gene_name": ["Gene1", "Gene2", "Gene3"],
            "score": [0.0] * 3,
        }
        utr_regions = pd.DataFrame(utr_regions).set_index("utr_name")
        cls.intervals = utr_mapping.generate_utr_intervals(utr_regions=utr_regions)


class TestTenxThreeSynthetic(TestMapReadsToUtrSynthetic):
    def setUp(self) -> None:
        self.maxDiff = None
        self.data = synthetic_data_3tenx.Bam()
        protocol_ = protocol.create(
            platform="10x",
            version=3,
            alignments=self.data.alignment,
            annotations="./test_data/vivo_anno.csv",  # Placeholder, not needed
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
        size_true_sense = 172196828 - 172196728 - 1
        size_true_anti = (91271840 + 80) - 91270659
        dict_true = {
            "read_barcodes": [
                ["GACTCTCCACAATGAA", "GACTCTCCACAATGAA"],
                ["TTCTTGACAAGTCATC", "TTCTTGACAAGTCATC"],
            ],
            "utr_sizes_from_reads": [
                [size_true_sense, size_true_sense],
                [size_true_anti, size_true_anti],
            ],
        }
        self.assertDictEqual(self.mappings.to_dict(orient="list"), dict_true)

    def test_length(self):
        self.assertEqual(self.mappings.shape[0], 2)


class TestTenxTwoSynthetic(TestMapReadsToUtrSynthetic):
    def setUp(self) -> None:
        self.maxDiff = None
        self.data = synthetic_data_3tenx.Bam()
        protocol_ = protocol.create(
            platform="10x",
            version=2,
            alignments=self.data.alignment,
            annotations="./test_data/vivo_anno.csv",  # Placeholder, not needed
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
        size_true_sense = 172196828 - 172196728 - 1
        size_true_anti = (91271840 + 80) - 91270659
        dict_true = {
            "read_barcodes": [
                ["GACTCTCCACAATGAA", "GACTCTCCACAATGAA"],
                ["TTCTTGACAAGTCATC", "TTCTTGACAAGTCATC"],
            ],
            "utr_sizes_from_reads": [
                [size_true_sense, size_true_sense],
                [size_true_anti, size_true_anti],
            ],
        }
        self.assertDictEqual(self.mappings.to_dict(orient="list"), dict_true)

    def test_length(self):
        self.assertEqual(self.mappings.shape[0], 2)


# TODO: do similar class as TestTenxSynthetic, but for smartseq to test UMI deduplication


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2, buffer=True)
    module_log.log_end_run("Finished testing")
