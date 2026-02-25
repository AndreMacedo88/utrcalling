from utrcalling.tests.test_data import (
    synthetic_annotations,
    synthetic_data_3tenx,
    synthetic_data_5tenx,
    synthetic_data_5smartseq,
)
from utrcalling.core import core, module_log, protocol
import os
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import scipy.sparse
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
        cls.filtered_nolabel, cls.ref_subset_nolabel = core.filter_mapped_reads(
            utr_sizes_counts=cls.mapped,
            utr_reference_regions=cls.utr_regions,
            protocol=cls.protocol_,
        )
        cls.results_nolabel = core.call_peaks(
            filtered_reads=cls.filtered_nolabel,
            utr_reference_subset=cls.ref_subset_nolabel,
            protocol=cls.protocol_,
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
        cls.filtered_label, cls.ref_subset_label = core.filter_mapped_reads(
            utr_sizes_counts=cls.mapped_label,
            utr_reference_regions=cls.utr_regions,
            protocol=cls.protocol_label,
        )
        cls.results_label = core.call_peaks(
            filtered_reads=cls.filtered_label,
            utr_reference_subset=cls.ref_subset_label,
            protocol=cls.protocol_label,
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
        self.filtered, self.ref_subset = core.filter_mapped_reads(
            utr_sizes_counts=self.mapped,
            utr_reference_regions=self.utr_regions,
            protocol=self.protocol_,
        )
        self.results = core.call_peaks(
            filtered_reads=self.filtered,
            utr_reference_subset=self.ref_subset,
            protocol=self.protocol_,
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


# ---------------------------------------------------------------------------
# 4.2: End-to-end pipeline tests for additional protocols
# ---------------------------------------------------------------------------

# Shared UTR regions used across synthetic pipeline tests
_SYNTHETIC_UTR_REGIONS_3PRIME = {
    "utr_name": ["UTR1", "UTR2", "UTR3"],
    "chrom": [1, 1, 16],
    "start": [3102600, 172196728, 91270659],
    "end": [3102750, 172196828, 91271933],
    "name": ["UTR1", "UTR2", "UTR3"],
    "strand": ["+", "-", "+"],
    "transcript_id": ["UTR1_t", "UTR2_t", "UTR3_t"],
    "gene_name": ["Gene1", "Gene2", "Gene3"],
    "score": [0.0] * 3,
}

_SYNTHETIC_UTR_REGIONS_5TENX = {
    "utr_name": ["UTR1", "UTR2", "UTR3"],
    "chrom": [1, 1, 16],
    "start": [3102600, 172196719, 91270659],
    "end": [3102750, 172196828, 91271933],
    "name": ["UTR1", "UTR2", "UTR3"],
    "strand": ["-", "+", "-"],
    "transcript_id": ["UTR1_t", "UTR2_t", "UTR3_t"],
    "gene_name": ["Gene1", "Gene2", "Gene3"],
    "score": [0.0] * 3,
}

_SYNTHETIC_UTR_REGIONS_5SMARTSEQ = {
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

# Minimal annotations without barcode columns (no barcode filtering)
_ANNOTATIONS_NO_BARCODES = pd.DataFrame({"sample": ["test"]})


class TestPipeline3TenxV3Synthetic(unittest.TestCase):
    """End-to-end pipeline test: 3' 10x v3 with synthetic BAM data."""

    @classmethod
    def setUpClass(cls):
        cls.utr_regions = pd.DataFrame(_SYNTHETIC_UTR_REGIONS_3PRIME).set_index(
            "utr_name"
        )
        cls._data = synthetic_data_3tenx.Bam()
        cls._data.enter()

        protocol_ = protocol.create(
            platform="10x",
            version=3,
            alignments=cls._data.alignment,
            annotations=_ANNOTATIONS_NO_BARCODES,
        )
        mapped = core.map_aligned_reads(
            alignments=cls._data.alignment,
            utr_regions=cls.utr_regions,
            protocol=protocol_,
            store=False,
        )
        cls.filtered, cls.ref_subset = core.filter_mapped_reads(
            utr_sizes_counts=mapped,
            utr_reference_regions=cls.utr_regions,
            protocol=protocol_,
        )
        cls.results = core.call_peaks(
            filtered_reads=cls.filtered,
            utr_reference_subset=cls.ref_subset,
            protocol=protocol_,
        )

    @classmethod
    def tearDownClass(cls):
        cls._data.__exit__(None, None, None)

    def test_is_anndata(self):
        self.assertIsInstance(self.results, ad.AnnData)

    def test_nonempty(self):
        self.assertGreater(self.results.shape[0], 0)
        self.assertGreater(self.results.shape[1], 0)

    def test_sparse_matrix(self):
        self.assertTrue(scipy.sparse.issparse(self.results.X))

    def test_var_has_utr_size(self):
        self.assertIn("utr_size", self.results.var.columns)

    def test_shape(self):
        self.assertTupleEqual(self.results.shape, (2, 2))


class TestPipeline3TenxV2Synthetic(unittest.TestCase):
    """End-to-end pipeline test: 3' 10x v2 with synthetic BAM data."""

    @classmethod
    def setUpClass(cls):
        cls.utr_regions = pd.DataFrame(_SYNTHETIC_UTR_REGIONS_3PRIME).set_index(
            "utr_name"
        )
        cls._data = synthetic_data_3tenx.Bam()
        cls._data.enter()

        protocol_ = protocol.create(
            platform="10x",
            version=2,
            alignments=cls._data.alignment,
            annotations=_ANNOTATIONS_NO_BARCODES,
        )
        mapped = core.map_aligned_reads(
            alignments=cls._data.alignment,
            utr_regions=cls.utr_regions,
            protocol=protocol_,
            store=False,
        )
        cls.filtered, cls.ref_subset = core.filter_mapped_reads(
            utr_sizes_counts=mapped,
            utr_reference_regions=cls.utr_regions,
            protocol=protocol_,
        )
        cls.results = core.call_peaks(
            filtered_reads=cls.filtered,
            utr_reference_subset=cls.ref_subset,
            protocol=protocol_,
        )

    @classmethod
    def tearDownClass(cls):
        cls._data.__exit__(None, None, None)

    def test_is_anndata(self):
        self.assertIsInstance(self.results, ad.AnnData)

    def test_nonempty(self):
        self.assertGreater(self.results.shape[0], 0)
        self.assertGreater(self.results.shape[1], 0)

    def test_sparse_matrix(self):
        self.assertTrue(scipy.sparse.issparse(self.results.X))

    def test_shape(self):
        self.assertTupleEqual(self.results.shape, (2, 2))


class TestPipeline5TenxSynthetic(unittest.TestCase):
    """End-to-end pipeline test: 5' 10x v1 with synthetic BAM data."""

    @classmethod
    def setUpClass(cls):
        cls.utr_regions = pd.DataFrame(_SYNTHETIC_UTR_REGIONS_5TENX).set_index(
            "utr_name"
        )
        cls._data = synthetic_data_5tenx.Bam()
        cls._data.enter()

        protocol_ = protocol.create(
            platform="5prime10x",
            version=1,
            alignments=cls._data.alignment,
            annotations=_ANNOTATIONS_NO_BARCODES,
        )
        mapped = core.map_aligned_reads(
            alignments=cls._data.alignment,
            utr_regions=cls.utr_regions,
            protocol=protocol_,
            store=False,
        )
        cls.filtered, cls.ref_subset = core.filter_mapped_reads(
            utr_sizes_counts=mapped,
            utr_reference_regions=cls.utr_regions,
            protocol=protocol_,
        )
        cls.results = core.call_peaks(
            filtered_reads=cls.filtered,
            utr_reference_subset=cls.ref_subset,
            protocol=protocol_,
        )

    @classmethod
    def tearDownClass(cls):
        cls._data.__exit__(None, None, None)

    def test_is_anndata(self):
        self.assertIsInstance(self.results, ad.AnnData)

    def test_nonempty(self):
        self.assertGreater(self.results.shape[0], 0)
        self.assertGreater(self.results.shape[1], 0)

    def test_sparse_matrix(self):
        self.assertTrue(scipy.sparse.issparse(self.results.X))


class TestPipeline5CapseqSynthetic(unittest.TestCase):
    """End-to-end pipeline test: 5' CapSeq SmartSeq3 with synthetic BAM data."""

    @classmethod
    def setUpClass(cls):
        cls.utr_regions = pd.DataFrame(_SYNTHETIC_UTR_REGIONS_5SMARTSEQ).set_index(
            "utr_name"
        )
        cls._data = synthetic_data_5smartseq.Bam()
        cls._data.enter()

        # CapSeq uses BAM path as observation ID, so annotations must not have
        # a barcode column (barcode filtering would fail on path vs barcode length)
        annotations = pd.read_csv("./test_data/smartseq3_anno.csv")
        annotations = annotations.drop("Sample_Barcode", axis=1)
        protocol_ = protocol.create(
            platform="smart-seq_cap",
            version=3,
            alignments=cls._data.alignment,
            annotations=annotations,
        )
        mapped = core.map_aligned_reads(
            alignments=cls._data.alignment,
            utr_regions=cls.utr_regions,
            protocol=protocol_,
            store=False,
        )
        cls.filtered, cls.ref_subset = core.filter_mapped_reads(
            utr_sizes_counts=mapped,
            utr_reference_regions=cls.utr_regions,
            protocol=protocol_,
        )
        cls.results = core.call_peaks(
            filtered_reads=cls.filtered,
            utr_reference_subset=cls.ref_subset,
            protocol=protocol_,
        )

    @classmethod
    def tearDownClass(cls):
        cls._data.__exit__(None, None, None)

    def test_is_anndata(self):
        self.assertIsInstance(self.results, ad.AnnData)

    def test_nonempty(self):
        self.assertGreater(self.results.shape[0], 0)
        self.assertGreater(self.results.shape[1], 0)

    def test_sparse_matrix(self):
        self.assertTrue(scipy.sparse.issparse(self.results.X))


# ---------------------------------------------------------------------------
# 4.4: dtype preservation tests
# ---------------------------------------------------------------------------

class TestDtypePreservationSmartseq3(TestCallPeaksSmartseq3Minimal):
    """Verify that intermediate DataFrames use canonical dtypes."""

    def test_mapped_utr_size_dtype(self):
        self.assertEqual(self.mapped.dtypes["utr_size"], np.int32)

    def test_mapped_utr_name_is_categorical(self):
        self.assertIsInstance(self.mapped.dtypes["utr_name"], pd.CategoricalDtype)

    def test_mapped_observation_id_is_categorical(self):
        self.assertIsInstance(
            self.mapped.dtypes["observation_id"], pd.CategoricalDtype
        )

    def test_filtered_utr_size_dtype(self):
        self.assertEqual(self.filtered_nolabel.dtypes["utr_size"], np.int32)

    def test_filtered_utr_name_is_categorical(self):
        self.assertIsInstance(
            self.filtered_nolabel.dtypes["utr_name"], pd.CategoricalDtype
        )

    def test_anndata_x_is_numeric(self):
        # anndata may convert int32 to float32; verify it's a valid numeric type
        self.assertTrue(np.issubdtype(self.results_nolabel.X.dtype, np.number))


# ---------------------------------------------------------------------------
# 4.5: Determinism tests
# ---------------------------------------------------------------------------

class TestPipelineDeterminism(unittest.TestCase):
    """Verify that running the pipeline twice produces identical results."""

    @classmethod
    def setUpClass(cls):
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
        alignments = [
            "./test_data/smartseq3_test_sorted_AS-740066-LR-59119.bam"
        ]
        annotations = pd.read_csv("./test_data/smartseq3_anno.csv")
        annotations = annotations.query("Sample_ID=='AS-740066-LR-59119'")
        annotations = annotations.drop_duplicates(subset=["Sample_Barcode"])

        # Run the full pipeline twice
        runs = []
        for _ in range(2):
            protocol_ = protocol.create(
                platform="smartseq",
                version=3,
                label_batches=False,
                annotations=annotations,
                alignments=alignments,
            )
            mapped = core.map_aligned_reads(
                alignments=alignments,
                utr_regions=utr_regions,
                protocol=protocol_,
                store=False,
            )
            filtered, ref_subset = core.filter_mapped_reads(
                utr_sizes_counts=mapped,
                utr_reference_regions=utr_regions,
                protocol=protocol_,
            )
            result = core.call_peaks(
                filtered_reads=filtered,
                utr_reference_subset=ref_subset,
                protocol=protocol_,
            )
            runs.append(result)

        cls.run1 = runs[0]
        cls.run2 = runs[1]

    def test_same_shape(self):
        self.assertTupleEqual(self.run1.shape, self.run2.shape)

    def test_same_var_names(self):
        self.assertListEqual(
            self.run1.var_names.tolist(), self.run2.var_names.tolist()
        )

    def test_same_obs_names(self):
        self.assertListEqual(
            self.run1.obs_names.tolist(), self.run2.obs_names.tolist()
        )

    def test_same_count_matrix(self):
        diff = self.run1.X - self.run2.X
        self.assertEqual(diff.nnz, 0)

    def test_same_var_utr_sizes(self):
        self.assertListEqual(
            self.run1.var.utr_size.to_list(), self.run2.var.utr_size.to_list()
        )


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2, buffer=True)
    module_log.log_end_run("Finished testing")
