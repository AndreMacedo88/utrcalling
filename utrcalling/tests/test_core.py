from utrcalling.tests.test_data import synthetic_annotations
from utrcalling.core import core, module_log, protocol
import os
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import anndata as ad

# we change working dir to the base test dir so that relative paths are sane
os.chdir(str(Path(__file__).resolve().parents[0]))


class TestPrepareUtrReferenceRegionsSynthetic(unittest.TestCase):
    def setUp(self) -> None:
        gtf = synthetic_annotations.GtfEnsembl()
        with (
            gtf
        ):  # context manager so that the GtfMouse class cleans the file at the end
            self.regions = core.prepare_utr_reference_regions(
                gtf=gtf.file_path,
                feature="three_prime_utr",
                store=False,
            )

    def test_resulting_df(self):
        expected = pd.DataFrame(
            columns=[
                "chrom",
                "start",
                "end",
                "name",
                "score",
                "strand",
                "transcript_id",
                "gene_name",
            ],
            index=[
                "ENSMUSG4",
                "ENSMUSG5",
                "ENSMUSG1",
                "ENSMUSG1,ENSMUSG4",
                "ENSMUSG5_2",
                "ENSMUSG2",
                "ENSMUSG2_2",
                "ENSMUSG3",
            ],
            data=np.array(
                [
                    ["1", 500, 600, "ENSMUSG4", ".", "+", "ENSMUST6", "Test4"],
                    ["1", 500, 650, "ENSMUSG5", ".", "-", "ENSMUST7", "Test5"],
                    [
                        "1",
                        1000,
                        1100,
                        "ENSMUSG1",
                        ".",
                        "+",
                        "ENSMUST1,ENSMUST2",
                        "Test1",
                    ],
                    [
                        "1",
                        1150,
                        1350,
                        "ENSMUSG1,ENSMUSG4",
                        ".",
                        "+",
                        "ENSMUST1,ENSMUST2,ENSMUST6",
                        "Test1,Test4",
                    ],
                    ["1", 1200, 1300, "ENSMUSG5", ".", "-", "ENSMUST7", "Test5"],
                    [
                        "1",
                        2000,
                        2200,
                        "ENSMUSG2",
                        ".",
                        "-",
                        "ENSMUST3,ENSMUST4",
                        "Test2",
                    ],
                    ["1", 2400, 2600, "ENSMUSG2", ".", "-", "ENSMUST3", "Test2"],
                    ["1", 3000, 3200, "ENSMUSG3", ".", "+", "ENSMUST5", "Test3"],
                ]
            ),
        )
        expected.index.name = "utr_name"
        for i in range(1, 3):
            expected.isetitem(i, expected.iloc[:, i].astype("int64"))

        pd.testing.assert_frame_equal(
            self.regions,
            expected,
        )


class TestMapAlignedReads(unittest.TestCase):
    def setUp(self) -> None:
        utr_regions = pd.read_parquet("./test_data/test_UTR_reference.parquet")
        protocol_ = protocol.create(
            platform="10x", version=2, annotations="./test_data/vivo_anno.csv"
        )

        self.alignments_n = 5
        alignments = [
            "./test_data/test_UTRoldNSCgenome_bwt2_102_sort.bam",
        ] * self.alignments_n

        self.read_mappings = core.map_aligned_reads(
            alignments, utr_regions, protocol_, store=False
        )

        self.true_results_5804 = {
            "read_barcodes": [
                "GATGAAACATTAACCG",
                "TGCGCAGAGTCCATAC",
                "GCTGGGTGTACGCACC",
            ],
            "utr_sizes_from_reads": [1468, 1458, 1466],
        }

    def test_length(self):
        self.assertEqual(len(self.read_mappings), self.alignments_n)

    def test_read_mappings(self):
        for i in range(self.alignments_n):
            with self.subTest(i=i):
                utr_data = self.read_mappings[i][
                    self.read_mappings[i]["utr_name"] == "ENSMUSG00000005804"
                ]
                result = {
                    "observation_id": utr_data["observation_id"].tolist(),
                    "utr_size": utr_data["utr_size"].tolist(),
                }
                # Each BAM gets a batch suffix from get_observation_id()
                expected_barcodes = [
                    f"{bc}-{i + 1}" for bc in self.true_results_5804["read_barcodes"]
                ]
                expected = {
                    "observation_id": expected_barcodes,
                    "utr_size": self.true_results_5804["utr_sizes_from_reads"],
                }
                self.assertDictEqual(result, expected)


class TestCallPeaksSmartseq3Minimal(unittest.TestCase):
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
        cls.utr_regions = pd.DataFrame(utr_regions).set_index("utr_name")
        cls.alignments = ["./test_data/smartseq3_test_sorted_AS-740066-LR-59119.bam"]
        annotations = pd.read_csv("./test_data/smartseq3_anno.csv")
        cls.annotations = annotations.query("Sample_ID=='AS-740066-LR-59119'")
        cls.annotations = cls.annotations.drop_duplicates(subset=["Sample_Barcode"])

        # objects for testing without label_batches
        cls.protocol_ = protocol.create(
            platform="smartseq",
            version=3,
            label_batches=False,
            annotations=cls.annotations,
            alignments=cls.alignments,
        )
        cls.mapped = core.map_aligned_reads(
            alignments=cls.alignments,
            utr_regions=cls.utr_regions,
            protocol=cls.protocol_,
            store=False,
        )
        cls.results_nolabel = core.call_peaks(
            utr_sizes_counts=cls.mapped,
            utr_reference_regions=cls.utr_regions,
            protocol=cls.protocol_,
            store=False,
        )
        # objects for testing with label_batches
        cls.protocol_label = protocol.create(
            platform="smartseq",
            version=3,
            label_batches=True,
            annotations=cls.annotations,
            alignments=cls.alignments,
        )
        cls.mapped_label = core.map_aligned_reads(
            alignments=cls.alignments,
            utr_regions=cls.utr_regions,
            protocol=cls.protocol_label,
            store=False,
        )
        cls.results_label = core.call_peaks(
            utr_sizes_counts=cls.mapped_label,
            utr_reference_regions=cls.utr_regions,
            protocol=cls.protocol_label,
            store=False,
        )


class TestCallPeaksSmartseq3LabelMinimal(TestCallPeaksSmartseq3Minimal):

    def test_is_anndata(self):
        self.assertIsInstance(self.results_label, ad.AnnData)

    def test_shape_anndata(self):
        self.assertTupleEqual(self.results_label.shape, (1, 2))

    def test_utr_sizes(self):
        self.assertListEqual(self.results_label.var.utr_size.to_list(), [223, 158])

    def test_barcode(self):
        self.assertEqual(self.results_label.obs_names[0], "CACTGT-1")

    def test_utr_names(self):
        self.assertListEqual(
            self.results_label.var_names.to_list(), ["UTR1-223", "UTR1-1-158"]
        )


class TestCallPeaksSmartseq3NoLabelMinimal(TestCallPeaksSmartseq3Minimal):

    def test_is_anndata(self):
        self.assertIsInstance(self.results_nolabel, ad.AnnData)

    def test_shape_anndata(self):
        self.assertTupleEqual(self.results_nolabel.shape, (1, 2))

    def test_utr_sizes(self):
        self.assertListEqual(self.results_nolabel.var.utr_size.to_list(), [223, 158])

    def test_barcode(self):
        self.assertEqual(self.results_nolabel.obs_names[0], "CACTGT")

    def test_utr_names(self):
        self.assertListEqual(
            self.results_nolabel.var_names.to_list(), ["UTR1-223", "UTR1-1-158"]
        )


class TestCallPeaksSmartseq3NoBarcodeMinimal(TestCallPeaksSmartseq3Minimal):
    def setUp(self) -> None:
        self.annotations = self.annotations.drop("Sample_Barcode", axis=1)
        self.protocol_ = protocol.create(
            platform="smartseq",
            version=3,
            label_batches=False,
            annotations=self.annotations,
            alignments=self.alignments,
        )
        self.mapped = core.map_aligned_reads(
            alignments=self.alignments,
            utr_regions=self.utr_regions,
            protocol=self.protocol_,
            store=False,
        )
        self.results = core.call_peaks(
            utr_sizes_counts=self.mapped,
            utr_reference_regions=self.utr_regions,
            protocol=self.protocol_,
            store=False,
        )

    def test_is_anndata(self):
        self.assertIsInstance(self.results, ad.AnnData)

    def test_shape_anndata(self):
        self.assertTupleEqual(self.results.shape, (1, 2))

    def test_utr_sizes(self):
        self.assertListEqual(self.results.var.utr_size.to_list(), [223, 158])

    def test_utr_names(self):
        self.assertListEqual(
            self.results.var_names.to_list(), ["UTR1-223", "UTR1-1-158"]
        )


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2, buffer=True)
    module_log.log_end_run("Finished testing")
