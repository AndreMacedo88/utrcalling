import os
from pathlib import Path
import unittest
import copy

import pandas as pd

from utrcalling.core import protocol, utr_mapping, process_mapped_reads, module_log

# make sure file paths resolve correctly for the test case
os.chdir(str(Path(__file__).resolve().parents[1]))


class Test10x(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        utr_regions = pd.read_parquet("./test_data/test_UTR_reference.parquet")
        intervals = utr_mapping.generate_utr_intervals(utr_regions=utr_regions)
        alignment_1 = "./test_data/test_UTRoldNSCgenome_bwt2_102_sort.bam"
        alignment_2 = "./test_data/test_UTRyoungNSCgenome_bwt2_102_sort.bam"
        cls.protocol = protocol.create(
            platform="10x",
            version=2,
            alignments=[alignment_1, alignment_2],
            annotations="./test_data/smartseq3_anno.csv",
            merge_method="intersection",
        )
        cls.mappings = [
            utr_mapping.map_reads_to_utr(
                alignment, intervals, cls.protocol, read="first",
                batch_idx=str(i + 1)
            )
            for i, alignment in enumerate([alignment_1, alignment_2])
        ]


class TestFilterUtrSizes(Test10x):
    def setUp(self) -> None:
        merged = process_mapped_reads.merge_samples_utr_sizes(
            copy.deepcopy(self.mappings), method=self.protocol.merge_method
        )

        # modify the data to check for successful filtering.
        # Replace utr_size values for ENSMUSG00000001036 with alternating 1 and 798
        mask = merged["utr_name"] == "ENSMUSG00000001036"
        utr_rows = merged[mask].index
        new_sizes = [1, 798, 1, 798, 1, 798]
        for i, idx in enumerate(utr_rows):
            merged.at[idx, "utr_size"] = new_sizes[i]

        self.filtered = self.protocol.filter_utr_sizes(merged, threshold=1)

    def test_number_of_features(self) -> None:
        n_unique_utrs = len(self.filtered["utr_name"].unique())
        self.assertEqual(n_unique_utrs, 63)

    def test_is_filtered(self):
        utr_data = self.filtered[
            self.filtered["utr_name"] == "ENSMUSG00000001036"
        ]
        self.assertListEqual(
            utr_data["observation_id"].tolist(),
            [
                "CTAATGGCATTCGACA-1",
                "CCCAGTTTCGCTTGTC-1",
                "TATTACCAGGATTCGG-2",
            ],
        )
        self.assertListEqual(
            utr_data["utr_size"].tolist(),
            [798, 798, 798],
        )


class TestSmartSeqCapPhosphoReadname(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        utr_regions = pd.read_parquet("./test_data/test_UTR_reference_5prime.parquet")
        intervals = utr_mapping.generate_utr_intervals(utr_regions=utr_regions)
        cls.alignment = "./test_data/test_CapPhospho_genome_102_coordsort.bam"
        cls.protocol = protocol.create(
            platform="smart-seq_cap-readname",
            version=3,
            alignments=cls.alignment,
            annotations="./test_data/smartseq3_anno.csv",
        )
        cls.mappings = utr_mapping.map_reads_to_utr(
            cls.alignment, intervals, cls.protocol, read="first"
        )


class TestFilterUtrSizesSmartSeqCap(TestSmartSeqCapPhosphoReadname):
    def setUp(self) -> None:
        mappings = self.mappings.copy()
        # modify the data to check for successful filtering. Original was [44, 23, 19].
        # Replace the second utr_size with 1 for ENSMUSG00000023905
        mask = mappings["utr_name"] == "ENSMUSG00000023905"
        utr_rows = mappings[mask].index
        new_sizes = [44, 1, 19]
        for i, idx in enumerate(utr_rows):
            mappings.at[idx, "utr_size"] = new_sizes[i]

        self.filtered = self.protocol.filter_utr_sizes(mappings)

    def test_number_of_features(self) -> None:
        # For 5'UTRs we don't want filtering by default — all rows stay
        total_rows = len(self.filtered)
        self.assertEqual(total_rows, len(self.mappings))

    def test_is_filtered(self):
        # For 5'UTRs we don't want filtering by default — all rows stay including size=1
        utr_data = self.filtered[
            self.filtered["utr_name"] == "ENSMUSG00000023905"
        ]
        self.assertListEqual(
            utr_data["observation_id"].tolist(),
            [self.alignment] * 3,
        )
        self.assertListEqual(
            utr_data["utr_size"].tolist(),
            [44, 1, 19],
        )


class TestStandardTenxCreation(unittest.TestCase):
    """Test creation and properties of standard 10x protocols."""

    def test_create_standard_10x_v3(self):
        p = protocol.create(
            "standard_10x", 3,
            annotations=pd.DataFrame({"sample": ["test"]}),
        )
        self.assertIsInstance(p, protocol.ProtocolStandardTenx3)

    def test_create_standard_10x_v2(self):
        p = protocol.create(
            "standard_10x", 2,
            annotations=pd.DataFrame({"sample": ["test"]}),
        )
        self.assertIsInstance(p, protocol.ProtocolStandardTenx2)

    def test_standard_10x_properties(self):
        p = protocol.create(
            "standard_10x", 3,
            annotations=pd.DataFrame({"sample": ["test"]}),
        )
        self.assertTrue(p.has_umi)
        self.assertEqual(p.dedup_strategy, "keep_max")
        self.assertEqual(p.read_in_pair, "any")

    def test_standard_10x_aliases(self):
        """Alternative platform name spellings."""
        for name in ("standard_10x", "standard_10_x", "standard_tenx"):
            p = protocol.create(
                name, 3,
                annotations=pd.DataFrame({"sample": ["test"]}),
            )
            self.assertIsInstance(p, protocol.ProtocolStandardTenx3)

    def test_standard_10x_strand_convention_is_sense(self):
        """Cell Ranger 10x 3' v3 R2 aligns sense to the gene, so standard_10x
        overrides the ProtocolThreePrime default to require matching strands."""
        p = protocol.create(
            "standard_10x", 3,
            annotations=pd.DataFrame({"sample": ["test"]}),
        )
        # + strand UTR → + strand read passes; - strand read is wrong-strand
        self.assertTrue(p.check_strand_matches("+", "+"))
        self.assertFalse(p.check_strand_matches("-", "+"))
        # - strand UTR → - strand read passes; + strand read is wrong-strand
        self.assertTrue(p.check_strand_matches("-", "-"))
        self.assertFalse(p.check_strand_matches("+", "-"))

    def test_standard_10x_unsupported_version(self):
        with self.assertRaises(NotImplementedError):
            protocol.create(
                "standard_10x", 1,
                annotations=pd.DataFrame({"sample": ["test"]}),
            )


class TestStandardSmartseqCreation(unittest.TestCase):
    """Test creation and properties of standard SmartSeq3 protocol."""

    def test_create_standard_smartseq_v3(self):
        p = protocol.create(
            "standard_smartseq", 3,
            annotations=pd.DataFrame({
                "sample_id": ["test"],
            }),
        )
        self.assertIsInstance(p, protocol.ProtocolStandardSmartseq3)

    def test_standard_smartseq_properties(self):
        p = protocol.create(
            "standard_smartseq", 3,
            annotations=pd.DataFrame({
                "sample_id": ["test"],
            }),
        )
        self.assertFalse(p.has_umi)
        self.assertEqual(p.dedup_strategy, "keep_first")
        self.assertEqual(p.read_in_pair, "any")

    def test_standard_smartseq_both_strands(self):
        p = protocol.create(
            "standard_smartseq", 3,
            annotations=pd.DataFrame({
                "sample_id": ["test"],
            }),
        )
        self.assertTrue(p.check_strand_matches("+", "+"))
        self.assertTrue(p.check_strand_matches("-", "+"))
        self.assertTrue(p.check_strand_matches("+", "-"))
        self.assertTrue(p.check_strand_matches("-", "-"))

    def test_standard_smartseq_aliases(self):
        for name in ("standard_smartseq", "standard_smart-seq"):
            p = protocol.create(
                name, 3,
                annotations=pd.DataFrame({
                    "sample_id": ["test"],
                }),
            )
            self.assertIsInstance(p, protocol.ProtocolStandardSmartseq3)

    def test_standard_smartseq_unsupported_version(self):
        with self.assertRaises(NotImplementedError):
            protocol.create(
                "standard_smartseq", 2,
                annotations=pd.DataFrame({"sample_id": ["test"]}),
            )


class TestTruseqCreation(unittest.TestCase):
    """Test creation and properties of the TruSeq stranded bulk protocol."""

    @staticmethod
    def _make(version=1, **kwargs):
        return protocol.create(
            "truseq", version,
            annotations=pd.DataFrame({"sample_id": ["test"]}),
            **kwargs,
        )

    def test_create_truseq(self):
        self.assertIsInstance(self._make(), protocol.ProtocolTruseq)

    def test_truseq_aliases(self):
        for name in ("truseq", "truseq_stranded", "standard_truseq", "standard_bulk"):
            p = protocol.create(
                name, 1, annotations=pd.DataFrame({"sample_id": ["test"]})
            )
            self.assertIsInstance(p, protocol.ProtocolTruseq)

    def test_truseq_version_agnostic(self):
        # Bulk RNA-seq has no protocol version; any value resolves the same.
        for version in (1, 2, 99):
            self.assertIsInstance(self._make(version=version), protocol.ProtocolTruseq)

    def test_truseq_properties(self):
        p = self._make()
        self.assertFalse(p.has_umi)
        self.assertEqual(p.dedup_strategy, "keep_first")
        self.assertEqual(p.read_in_pair, "any")
        self.assertFalse(p._reads_have_batch_suffix)

    def test_truseq_default_strandedness_is_reverse(self):
        p = self._make()
        self.assertIsNone(p.strandedness)
        # Reverse: keep antisense (read strand differs from UTR), reject sense.
        self.assertTrue(p.check_strand_matches("-", "+"))
        self.assertTrue(p.check_strand_matches("+", "-"))
        self.assertFalse(p.check_strand_matches("+", "+"))
        self.assertFalse(p.check_strand_matches("-", "-"))

    def test_truseq_forward_strandedness(self):
        p = self._make(strandedness="forward")
        self.assertEqual(p.strandedness, "forward")
        self.assertTrue(p.check_strand_matches("+", "+"))
        self.assertFalse(p.check_strand_matches("-", "+"))

    def test_truseq_unstranded(self):
        p = self._make(strandedness="unstranded")
        self.assertEqual(p.strandedness, "unstranded")
        self.assertTrue(p.check_strand_matches("+", "+"))
        self.assertTrue(p.check_strand_matches("-", "+"))

    def test_truseq_strandedness_case_insensitive(self):
        self.assertEqual(self._make(strandedness="Reverse").strandedness, "reverse")

    def test_truseq_invalid_strandedness_raises(self):
        with self.assertRaises(ValueError):
            self._make(strandedness="nonsense")

    def test_other_protocols_accept_strandedness_kwarg(self):
        # strandedness is a universal kwarg; non-stranded protocols ignore it.
        p = protocol.create(
            "10x", 3,
            annotations=pd.DataFrame({"barcode": ["AAAA"], "batch": [1]}),
            strandedness=None,
        )
        self.assertIsNone(p.strandedness)


class TestExistingProtocolDefaults(unittest.TestCase):
    """Existing protocols should return the original defaults."""

    def test_tenx3_defaults(self):
        p = protocol.create(
            "10x", 3,
            annotations=pd.DataFrame({"sample": ["test"]}),
        )
        self.assertTrue(p.has_umi)
        self.assertEqual(p.dedup_strategy, "keep_first")
        self.assertEqual(p.read_in_pair, "first")

    def test_smartseq3_defaults(self):
        p = protocol.create(
            "smartseq", 3,
            annotations=pd.DataFrame({"sample_id": ["test"]}),
        )
        self.assertTrue(p.has_umi)
        self.assertEqual(p.dedup_strategy, "keep_first")
        self.assertEqual(p.read_in_pair, "first")


class TestReadsBatchSuffixProperty(unittest.TestCase):
    """_reads_have_batch_suffix distinguishes CB-tag from sequence-based protocols."""

    def test_tenx3_no_suffix(self):
        p = protocol.create(
            "10x", 3,
            annotations=pd.DataFrame({"sample": ["test"]}),
        )
        self.assertFalse(p._reads_have_batch_suffix)

    def test_tenx2_no_suffix(self):
        p = protocol.create(
            "10x", 2,
            annotations=pd.DataFrame({"sample": ["test"]}),
        )
        self.assertFalse(p._reads_have_batch_suffix)

    def test_five_tenx_has_suffix(self):
        p = protocol.create(
            "5'10x", 1,
            annotations=pd.DataFrame({"sample": ["test"]}),
        )
        self.assertTrue(p._reads_have_batch_suffix)

    def test_standard_tenx3_has_suffix(self):
        p = protocol.create(
            "standard_10x", 3,
            annotations=pd.DataFrame({"sample": ["test"]}),
        )
        self.assertTrue(p._reads_have_batch_suffix)

    def test_standard_tenx2_has_suffix(self):
        p = protocol.create(
            "standard_10x", 2,
            annotations=pd.DataFrame({"sample": ["test"]}),
        )
        self.assertTrue(p._reads_have_batch_suffix)

    def test_smartseq3_no_suffix(self):
        p = protocol.create(
            "smartseq", 3,
            annotations=pd.DataFrame({"sample_id": ["test"]}),
        )
        self.assertFalse(p._reads_have_batch_suffix)

    def test_standard_smartseq3_no_suffix(self):
        p = protocol.create(
            "standard_smartseq", 3,
            annotations=pd.DataFrame({"sample_id": ["test"]}),
        )
        self.assertFalse(p._reads_have_batch_suffix)


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2, buffer=True)
    module_log.log_end_run("Finished testing")
