from test_data import synthetic_data
from utrcalling.core import (
    barcode_filtering,
    process_mapped_reads,
    protocol,
    utr_mapping,
    utr_reference,
    module_log,
    helpers,
)

import os
from pathlib import Path
from unittest import mock
import unittest

import pandas as pd
import numpy as np
import copy

# make sure file paths resolve correctly for the test case
os.chdir(str(Path(__file__).resolve().parents[0]))


class TestPeakCalling(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.utr_regions = pd.read_parquet("./test_data/test_UTR_reference.parquet")
        intervals = utr_mapping.generate_utr_intervals(utr_regions=cls.utr_regions)
        alignment_1 = "./test_data/test_UTRoldNSCgenome_bwt2_102_sort.bam"
        alignment_2 = "./test_data/test_UTRyoungNSCgenome_bwt2_102_sort.bam"
        cls.annotations = pd.read_csv("./test_data/vivo_anno.csv")
        cls.protocol_ = protocol.create(
            platform="10x",
            version=2,
            annotations=cls.annotations,
            merge_method="intersection",
        )
        cls.protocol_union = protocol.create(
            platform="10x",
            version=2,
            annotations=cls.annotations,
            merge_method="union",
        )
        cls.mappings = [
            utr_mapping.map_reads_to_utr(
                bam=alignment, intervals=intervals, protocol=cls.protocol_, read="first",
                batch_idx=str(i + 1)
            )
            for i, alignment in enumerate([alignment_1, alignment_2])
        ]


class TestMergeSamplesUtrSizesIntersection(TestPeakCalling):
    def setUp(self) -> None:
        self.maxDiff = None
        self.merged = process_mapped_reads.merge_samples_utr_sizes(
            utr_sizes=self.mappings, method=self.protocol_.merge_method
        )

    def test_details(self):
        utr_data = self.merged[
            self.merged["utr_name"] == "ENSMUSG00000025917"
        ]
        obs = utr_data["observation_id"].tolist()
        sizes = utr_data["utr_size"].tolist()
        expected_obs = [
            "CGTTGGGCAGTCAGAG-1",
            "CCGTGGAGTCTCGTTC-1",
            "GTAGTCAAGCTATGCT-1",
            "TGCGCAGAGGCATGTG-1",
            "CGACCTTTCTGCTGCT-2",
            "TGTTCCGAGGGTCGAT-2",
            "GTACGTAAGCGTTTAC-2",
            "GTGCGGTAGGACAGCT-2",
            "GCTGCAGCACTGTGTA-2",
            "GTCCTCAGTAAACGCG-2",
        ]
        expected_sizes = [153, 148, 148, 144, 375, 151, 148, 148, 142, 138]
        self.assertListEqual(obs, expected_obs)
        self.assertListEqual(sizes, expected_sizes)

    def test_length(self):
        self.assertEqual(len(self.merged["utr_name"].unique()), 63)


class TestMergeSamplesUtrSizesUnion(TestPeakCalling):
    def setUp(self) -> None:
        self.maxDiff = None
        self.merged = process_mapped_reads.merge_samples_utr_sizes(
            utr_sizes=copy.deepcopy(self.mappings),
            method=self.protocol_union.merge_method,
        )

    def test_identical_index(self):
        utr_data = self.merged[
            self.merged["utr_name"] == "ENSMUSG00000025917"
        ]
        obs = utr_data["observation_id"].tolist()
        sizes = utr_data["utr_size"].tolist()
        expected_obs = [
            "CGTTGGGCAGTCAGAG-1",
            "CCGTGGAGTCTCGTTC-1",
            "GTAGTCAAGCTATGCT-1",
            "TGCGCAGAGGCATGTG-1",
            "CGACCTTTCTGCTGCT-2",
            "TGTTCCGAGGGTCGAT-2",
            "GTACGTAAGCGTTTAC-2",
            "GTGCGGTAGGACAGCT-2",
            "GCTGCAGCACTGTGTA-2",
            "GTCCTCAGTAAACGCG-2",
        ]
        expected_sizes = [153, 148, 148, 144, 375, 151, 148, 148, 142, 138]
        self.assertListEqual(obs, expected_obs)
        self.assertListEqual(sizes, expected_sizes)

    def test_index_on_first_df(self):
        utr_data = self.merged[
            self.merged["utr_name"] == "ENSMUSG00000115423"
        ]
        obs = utr_data["observation_id"].tolist()
        sizes = utr_data["utr_size"].tolist()
        self.assertListEqual(obs, ["CGAGAAGGTGCAACGA-1", "CAGTCCTAGAAGGACA-1"])
        self.assertListEqual(sizes, [53, 28])

    def test_index_on_second_df(self):
        utr_data = self.merged[
            self.merged["utr_name"] == "ENSMUSG00000079334"
        ]
        obs = utr_data["observation_id"].tolist()
        sizes = utr_data["utr_size"].tolist()
        self.assertListEqual(obs, ["ACAGCTATCCTTAATC-2"])
        self.assertListEqual(sizes, [192])

    def test_length(self):
        n_unique_utrs = len(self.merged["utr_name"].unique())
        expected = len(
            self.mappings[0]["utr_name"].unique().tolist()
            + [u for u in self.mappings[1]["utr_name"].unique()
               if u not in self.mappings[0]["utr_name"].unique().tolist()]
        )
        self.assertEqual(n_unique_utrs, expected)


class TestSubsetReferenceData(TestPeakCalling):
    def setUp(self) -> None:
        merged = process_mapped_reads.merge_samples_utr_sizes(
            utr_sizes=copy.deepcopy(self.mappings)
        )
        self.subsetted = utr_reference.subset_reference_data(
            utr_sizes=merged, utr_reference_regions=self.utr_regions
        )

    def test_number_of_features(self) -> None:
        self.assertEqual(self.subsetted.shape[0], 63)

    def test_number_of_fields(self) -> None:
        self.assertEqual(self.subsetted.shape[1], 8)

    def test_values_are_proper(self) -> None:
        proper_values = [
            "1",
            10024601,
            10024978,
            "ENSMUSG00000025917",
            0.0,
            "-",
            "ENSMUST00000027050",
            "Cops5",
        ]
        self.assertListEqual(
            self.subsetted.loc["ENSMUSG00000025917"].values.tolist(), proper_values
        )

    def test_strand_is_correct(self) -> None:
        self.assertEqual(self.subsetted.iloc[0].strand, "-")


class TestMergeUtrsPerGene(unittest.TestCase):
    def setUp(self) -> None:
        mapped_data = synthetic_data.MappedData()
        utr_regions = mapped_data.utr_regions
        mapped_reads = mapped_data.mapped_reads
        self.utr_regions, self.merged_mapped_reads = (
            process_mapped_reads.merge_utrs_per_gene(
                utr_regions=utr_regions, read_mappings=mapped_reads
            )
        )

    def test_index(self):
        self.assertListEqual(
            sorted(self.merged_mapped_reads["utr_name"].unique().tolist()),
            [
                "ENSMUSG1",
                "ENSMUSG1,ENSMUSG4",
                "ENSMUSG2",
                "ENSMUSG3",
                "ENSMUSG4",
                "ENSMUSG5",
            ],
        )

    def test_positive_strand_sizes(self):
        pos_strand_genes = self.utr_regions[
            self.utr_regions["strand"] == "+"
        ].index.tolist()
        pos_data = self.merged_mapped_reads[
            self.merged_mapped_reads["utr_name"].isin(pos_strand_genes)
        ]
        # Group by utr_name and get sizes as lists, preserving gene order
        result = {}
        for gene in pos_strand_genes:
            gene_data = pos_data[pos_data["utr_name"] == gene]
            result[gene] = gene_data["utr_size"].tolist()

        expected = {
            "ENSMUSG1": [60],
            "ENSMUSG1,ENSMUSG4": [60, 65],
            "ENSMUSG3": [90],
            "ENSMUSG4": [50, 45],
        }
        self.assertDictEqual(result, expected)

    def test_negative_strand_sizes(self):
        neg_strand_genes = self.utr_regions[
            self.utr_regions["strand"] == "-"
        ].index.tolist()
        neg_data = self.merged_mapped_reads[
            self.merged_mapped_reads["utr_name"].isin(neg_strand_genes)
        ]
        result = {}
        for gene in neg_strand_genes:
            gene_data = neg_data[neg_data["utr_name"] == gene]
            result[gene] = gene_data["utr_size"].tolist()

        expected = {
            "ENSMUSG2": [50, 230, 235],
            "ENSMUSG5": [20, 25, 80, 240, 350],
        }
        self.assertDictEqual(result, expected)

    def test_utr_regions_dataframe(self):
        expected = {
            "chrom": {
                "ENSMUSG1": "1",
                "ENSMUSG1,ENSMUSG4": "1",
                "ENSMUSG2": "1",
                "ENSMUSG3": "1",
                "ENSMUSG4": "1",
                "ENSMUSG5": "1",
            },
            "start": {
                "ENSMUSG1": "1000",
                "ENSMUSG1,ENSMUSG4": "1150",
                "ENSMUSG2": "2000|2400",
                "ENSMUSG3": "3000",
                "ENSMUSG4": "500",
                "ENSMUSG5": "500|800|1100|1200",
            },
            "end": {
                "ENSMUSG1": "1100",
                "ENSMUSG1,ENSMUSG4": "1350",
                "ENSMUSG2": "2200|2600",
                "ENSMUSG3": "3200",
                "ENSMUSG4": "600",
                "ENSMUSG5": "650|900|1250|1300",
            },
            "strand": {
                "ENSMUSG1": "+",
                "ENSMUSG1,ENSMUSG4": "+",
                "ENSMUSG2": "-",
                "ENSMUSG3": "+",
                "ENSMUSG4": "+",
                "ENSMUSG5": "-",
            },
            "transcript_id": {
                "ENSMUSG1": "ENSMUST1|ENSMUST2",
                "ENSMUSG1,ENSMUSG4": "ENSMUST1|ENSMUST2|ENSMUST6",
                "ENSMUSG2": "ENSMUST3|ENSMUST4",
                "ENSMUSG3": "ENSMUST5",
                "ENSMUSG4": "ENSMUST6",
                "ENSMUSG5": "ENSMUST7|ENSMUST8|ENSMUST9",
            },
            "gene_name": {
                "ENSMUSG1": "Test1",
                "ENSMUSG1,ENSMUSG4": "Test1|Test4",
                "ENSMUSG2": "Test2",
                "ENSMUSG3": "Test3",
                "ENSMUSG4": "Test4",
                "ENSMUSG5": "Test5",
            },
            "name": {
                "ENSMUSG1": "ENSMUSG1",
                "ENSMUSG1,ENSMUSG4": "ENSMUSG1,ENSMUSG4",
                "ENSMUSG2": "ENSMUSG2",
                "ENSMUSG3": "ENSMUSG3",
                "ENSMUSG4": "ENSMUSG4",
                "ENSMUSG5": "ENSMUSG5",
            },
        }
        self.assertDictEqual(expected, self.utr_regions.to_dict())

    def test_read_mappings_dataframe(self):
        # Check that observation_id and utr_size are correctly paired per gene.
        # In the flat format, each row correctly associates an observation with its
        # offset-adjusted size. For negative strand genes the sort order differs from
        # the old list-in-cells code, but the pairings are now correct.
        expected = {
            "ENSMUSG1": [("GAAGACGT", 60)],
            "ENSMUSG1,ENSMUSG4": [("GAAGACGA", 60), ("ATGATGA", 65)],
            "ENSMUSG2": [("AAAGACGG", 50), ("AAAGACGA", 230), ("GTGATGA", 235)],
            "ENSMUSG3": [("CAAGACGG", 90)],
            "ENSMUSG4": [("CTAGACGT", 50), ("CTGATGA", 45)],
            "ENSMUSG5": [
                ("TAAGACGA", 20), ("TTGATGA", 25),
                ("ATAGCCGT", 80), ("GTAGCCGT", 240), ("GTAGACGT", 350),
            ],
        }
        for gene, expected_pairs in expected.items():
            gene_data = self.merged_mapped_reads[
                self.merged_mapped_reads["utr_name"] == gene
            ]
            actual_pairs = list(zip(
                gene_data["observation_id"].tolist(),
                gene_data["utr_size"].tolist(),
            ))
            self.assertListEqual(
                actual_pairs, expected_pairs,
                msg=f"mismatch for {gene}"
            )


class TestMergeUtrsPerGeneCategoricalOffsetMap(unittest.TestCase):
    """Regression test for pandas 2.1+ Categorical .map(dict).fillna(0) behavior.

    When both read_mappings["utr_name"] and utr_regions' index carry matching
    Categorical dtype (as can happen in production flows where dtypes propagate
    through joins), the inner merge preserves the Categorical column on
    `combined["utr_name"]`. Subsequently, `.map(offset_map)` with an injective
    non-zero-valued mapping preserves the Categorical dtype (new categories =
    the offset values), and the downstream `.fillna(0)` then fails because 0
    is not among the Categorical's categories.
    """

    _offset_counter = [0]

    @classmethod
    def _nonzero_unique_offsets(cls, group: pd.DataFrame) -> dict[str, int]:
        # Deterministic injective non-zero mapping across all calls, so every
        # UTR gets a distinct non-zero offset. This is what triggers the pandas
        # Categorical preservation behavior that the bug relied on.
        out = {}
        for name in group.index.tolist():
            cls._offset_counter[0] += 1
            out[name] = cls._offset_counter[0] * 100
        return out

    def setUp(self) -> None:
        # Reset counter so repeated runs stay deterministic.
        type(self)._offset_counter[0] = 0

    def test_does_not_crash_when_offsets_are_all_nonzero_distinct(self):
        utr_regions = pd.DataFrame(
            {
                "chrom": ["1", "1"],
                "start": [1000, 2000],
                "end": [1100, 2100],
                "name": ["GENE1", "GENE2"],
                "score": [".", "."],
                "strand": ["+", "+"],
                "transcript_id": ["TX1", "TX2"],
                "gene_name": ["G1", "G2"],
            },
            index=pd.CategoricalIndex(["UTR_A", "UTR_B"], name="utr_name"),
        )
        read_mappings = pd.DataFrame(
            {
                "utr_name": pd.Categorical(["UTR_A", "UTR_B"]),
                "observation_id": pd.Categorical(["obs1", "obs2"]),
                "utr_size": np.array([10, 20], dtype=np.int32),
            }
        )

        with mock.patch.object(
            process_mapped_reads,
            "_compute_gene_offsets",
            side_effect=self._nonzero_unique_offsets,
        ):
            regions_out, reads_out = process_mapped_reads.merge_utrs_per_gene(
                utr_regions=utr_regions, read_mappings=read_mappings
            )

        # Offsets: first-processed UTR gets 100, second gets 200 (distinct,
        # non-zero). Applied to utr_size (10 and 20), result should be {110, 220}.
        self.assertEqual(reads_out["utr_size"].dtype, np.int32)
        self.assertEqual(len(reads_out), 2)
        self.assertListEqual(
            sorted(reads_out["utr_size"].tolist()), [110, 220]
        )


class TestFilterBarcodes(TestPeakCalling):
    def setUp(self) -> None:
        self.merged = process_mapped_reads.merge_samples_utr_sizes(
            utr_sizes=copy.deepcopy(self.mappings)
        )
        self.annotations = "./test_data/vivo_anno.csv"

    @staticmethod
    def are_all_barcodes_in_anno(data, annotations):
        barcodes_real = set(annotations.CellBarcode)
        unique_utrs = data["utr_name"].unique()
        counter = 0
        control = 0
        for utr in unique_utrs:
            control += 1
            utr_obs = set(data[data["utr_name"] == utr]["observation_id"].unique())
            if utr_obs.issubset(barcodes_real):
                counter += 1
        return counter, control

    def test_before_filter(self):
        annotations = pd.read_csv(self.annotations)
        barcodes_in_data, barcodes_in_annotations = self.are_all_barcodes_in_anno(
            self.merged, annotations
        )
        self.assertNotEqual(barcodes_in_data, barcodes_in_annotations)

    def test_filter(self):
        filtered = barcode_filtering.filter_barcodes(
            utr_sizes=self.merged, whitelist=self.protocol_.barcode_whitelist
        )
        annotations = pd.read_csv(self.annotations)
        barcodes_in_data, barcodes_in_annotations = self.are_all_barcodes_in_anno(
            filtered, annotations
        )
        self.assertEqual(barcodes_in_data, barcodes_in_annotations)

    def test_specific_utr(self):
        filtered = barcode_filtering.filter_barcodes(
            utr_sizes=self.merged, whitelist=self.protocol_.barcode_whitelist
        )
        merged_count = len(
            self.merged[self.merged["utr_name"] == "ENSMUSG00000025917"]
        )
        filtered_count = len(
            filtered[filtered["utr_name"] == "ENSMUSG00000025917"]
        )
        self.assertNotEqual(merged_count, filtered_count)

    def test_specific_utr_2(self):
        barcodes_real = [
            "CGTTGGGCAGTCAGAG-1",
            "CCGTGGAGTCTCGTTC-1",
            "TGCGCAGAGGCATGTG-1",
            "CGACCTTTCTGCTGCT-2",
            "TGTTCCGAGGGTCGAT-2",
            "GTACGTAAGCGTTTAC-2",
            "GTGCGGTAGGACAGCT-2",
            "GCTGCAGCACTGTGTA-2",
            "GTCCTCAGTAAACGCG-2",
        ]
        filtered = barcode_filtering.filter_barcodes(
            utr_sizes=self.merged, whitelist=self.protocol_.barcode_whitelist
        )
        utr_data = filtered[filtered["utr_name"] == "ENSMUSG00000025917"]
        self.assertListEqual(
            utr_data["observation_id"].tolist(), barcodes_real
        )

    def test_using_annotations_as_dataframe(self):
        annotations = pd.read_csv(self.annotations)
        filtered = barcode_filtering.filter_barcodes(
            utr_sizes=self.merged, whitelist=self.protocol_.barcode_whitelist
        )
        barcodes_in_data, barcodes_in_annotations = self.are_all_barcodes_in_anno(
            filtered, annotations
        )
        self.assertEqual(barcodes_in_data, barcodes_in_annotations)


class TestFilterBarcodesSynthetic(unittest.TestCase):
    def setUp(self) -> None:
        data_synthetic = synthetic_data.MappedDataThreePrimeTenX()
        self.data = data_synthetic.data_synthetic
        self.anno = data_synthetic.anno
        self.filtered = barcode_filtering.filter_barcodes(
            utr_sizes=self.data, whitelist=self.anno["barcodes"].unique()
        )
        self.barcodes_filtered = self.filtered["observation_id"].tolist()

    def test_barcodes_in_anno(self):
        self.assertFalse(
            [
                barcode
                for barcode in self.barcodes_filtered
                if barcode not in self.anno.barcodes.tolist()
            ]
        )

    def test_filtered_len(self):
        self.assertEqual(len(self.barcodes_filtered), 10)

    def test_filtered_id(self):
        correct = [
            "AAAAAAAAAAAAAAAA-1",
            "AAAAAAAAAAAAAAAA-1",
            "GAAAAAAAAAAAAAAT-2",
            "GAAAAAAAAAAAAAAT-2",
            "AAAAAAAAAAAAAAGG-2",
            "AAAAAAAAAAAAAAAG-1",
            "AAAAAAAAAAAAAAAG-1",
            "AAAAAAAAAAAAAAGG-2",
            "GAAAAAAAAAAAAAAT-2",
            "AAAAAAAAAAAAATTT-1",
        ]
        self.assertListEqual(correct, self.barcodes_filtered)


class TestMergeSamplesEdgeCases(unittest.TestCase):
    """Edge case tests for merge_samples_utr_sizes()."""

    def _make_df(self, utr_names, obs_ids, sizes):
        return pd.DataFrame({
            "utr_name": pd.Categorical(utr_names),
            "observation_id": pd.Categorical(obs_ids),
            "utr_size": np.array(sizes, dtype=np.int32),
        })

    def test_single_sample_union(self):
        """Single-sample merge should return the same data."""
        df = self._make_df(["A", "B"], ["obs1", "obs2"], [10, 20])
        result = process_mapped_reads.merge_samples_utr_sizes([df], method="union")
        self.assertEqual(len(result), 2)
        self.assertListEqual(sorted(result["utr_name"].unique().tolist()), ["A", "B"])

    def test_single_sample_intersection(self):
        """Single-sample intersection should return the same data."""
        df = self._make_df(["A", "B"], ["obs1", "obs2"], [10, 20])
        result = process_mapped_reads.merge_samples_utr_sizes([df], method="intersection")
        self.assertEqual(len(result), 2)

    def test_no_common_utrs_intersection(self):
        """Intersection with no common UTRs should return empty DataFrame."""
        df1 = self._make_df(["A"], ["obs1"], [10])
        df2 = self._make_df(["B"], ["obs2"], [20])
        result = process_mapped_reads.merge_samples_utr_sizes(
            [df1, df2], method="intersection"
        )
        self.assertEqual(len(result), 0)

    def test_invalid_method_raises(self):
        """Invalid merge method should raise AttributeError."""
        df = self._make_df(["A"], ["obs1"], [10])
        with self.assertRaises(AttributeError):
            process_mapped_reads.merge_samples_utr_sizes([df], method="invalid")


class TestSubsetReferenceDataEdgeCases(unittest.TestCase):
    """Edge case tests for subset_reference_data()."""

    def test_no_matching_utrs(self):
        """When no UTR names match the reference, result should be empty."""
        utr_sizes = pd.DataFrame({
            "utr_name": pd.Categorical(["NONEXISTENT"]),
            "observation_id": pd.Categorical(["obs1"]),
            "utr_size": np.array([10], dtype=np.int32),
        })
        utr_ref = pd.DataFrame(
            {"chrom": ["1"], "start": [100], "end": [200]},
            index=pd.Index(["SOME_UTR"], name="utr_name"),
        )
        result = utr_reference.subset_reference_data(utr_sizes, utr_ref)
        self.assertEqual(len(result), 0)

    def test_all_matching_utrs(self):
        """When all UTRs match, result should have the same number of rows."""
        utr_sizes = pd.DataFrame({
            "utr_name": pd.Categorical(["UTR1", "UTR2"]),
            "observation_id": pd.Categorical(["obs1", "obs2"]),
            "utr_size": np.array([10, 20], dtype=np.int32),
        })
        utr_ref = pd.DataFrame(
            {"chrom": ["1", "1"], "start": [100, 200], "end": [150, 250]},
            index=pd.Index(["UTR1", "UTR2"], name="utr_name"),
        )
        result = utr_reference.subset_reference_data(utr_sizes, utr_ref)
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2, buffer=True)
    module_log.log_end_run("Finished testing")
