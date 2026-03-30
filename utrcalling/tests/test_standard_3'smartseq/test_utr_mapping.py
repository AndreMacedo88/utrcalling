"""Tests for standard SmartSeq3 protocol UTR mapping.

Validates both-strand acceptance, distal position extraction,
no deduplication, and observation_id from BAM path.
"""

import os
import unittest

import pandas as pd

from utrcalling.core import protocol, utr_mapping
from utrcalling.tests.test_data import synthetic_data_standard_3smartseq


class TestMapReadsToUtrSynthetic(unittest.TestCase):
    """Base class providing shared UTR region setup."""

    @classmethod
    def setUpClass(cls):
        os.chdir(os.path.join(os.path.dirname(__file__), ".."))
        utr_regions = {
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
        utr_regions = pd.DataFrame(utr_regions).set_index("utr_name")
        cls.intervals = utr_mapping.generate_utr_intervals(utr_regions=utr_regions)


class TestStandardSmartseq3(TestMapReadsToUtrSynthetic):
    """Test standard SmartSeq3 protocol mapping with synthetic BAM."""

    def setUp(self) -> None:
        self.maxDiff = None
        self.data = synthetic_data_standard_3smartseq.Bam()
        self.protocol = protocol.create(
            platform="standard_smartseq",
            version=3,
            alignments=self.data.alignment,
            annotations=pd.DataFrame({
                "sample_id": ["test_sample"],
            }),
        )
        with self.data:
            self.bam_path = self.data.alignment
            self.mappings = utr_mapping.map_reads_to_utr(
                self.data.alignment,
                self.intervals,
                self.protocol,
                read=self.protocol.read_in_pair,
            )

    def test_protocol_properties(self):
        self.assertFalse(self.protocol.has_umi)
        self.assertEqual(self.protocol.read_in_pair, "any")
        self.assertEqual(self.protocol.dedup_strategy, "keep_first")

    def test_both_strands_accepted(self):
        """check_strand_matches should always return True."""
        self.assertTrue(self.protocol.check_strand_matches("+", "+"))
        self.assertTrue(self.protocol.check_strand_matches("-", "+"))
        self.assertTrue(self.protocol.check_strand_matches("+", "-"))
        self.assertTrue(self.protocol.check_strand_matches("-", "-"))

    def test_utrs_found(self):
        """Should find reads mapping to UTR2 and UTR3."""
        utrs = sorted(self.mappings["utr_name"].unique().tolist())
        self.assertEqual(len(utrs), 2)
        self.assertIn("UTR2", utrs)
        self.assertIn("UTR3", utrs)

    def test_no_dedup_all_reads_kept(self):
        """Without UMIs, all reads should be kept (no deduplication).
        UTR3 has 3 reads, UTR2 has 2 reads."""
        utr3 = self.mappings[self.mappings["utr_name"] == "UTR3"]
        utr2 = self.mappings[self.mappings["utr_name"] == "UTR2"]
        self.assertEqual(len(utr3), 3)
        self.assertEqual(len(utr2), 2)

    def test_utr3_sizes_distal_position(self):
        """UTR3 (+ strand): distal position = reference_end - 1.
        Expected sizes: 441, 641, 941"""
        utr3 = self.mappings[self.mappings["utr_name"] == "UTR3"]
        sizes = sorted(utr3["utr_size"].tolist())
        self.assertEqual(sizes, [441, 641, 941])

    def test_utr2_sizes_distal_position(self):
        """UTR2 (- strand): distal position = reference_start.
        Expected sizes: 87, 97"""
        utr2 = self.mappings[self.mappings["utr_name"] == "UTR2"]
        sizes = sorted(utr2["utr_size"].tolist())
        self.assertEqual(sizes, [87, 97])

    def test_observation_id_is_bam_path(self):
        """Observation IDs should all be the BAM file path."""
        obs_ids = self.mappings["observation_id"].unique().tolist()
        self.assertEqual(len(obs_ids), 1)


class TestStandardSmartseq3DistalPosition(unittest.TestCase):
    """Direct unit tests for get_read_position method."""

    def test_plus_strand_uses_rightmost(self):
        """For + strand UTR, should always return ref_end_minus1."""
        from utrcalling.core.protocol import ProtocolStandardSmartseq3
        # Forward read
        pos = ProtocolStandardSmartseq3.get_read_position(100, 199, False, "+")
        self.assertEqual(pos, 199)
        # Reverse read
        pos = ProtocolStandardSmartseq3.get_read_position(100, 199, True, "+")
        self.assertEqual(pos, 199)

    def test_minus_strand_uses_leftmost(self):
        """For - strand UTR, should always return ref_start."""
        from utrcalling.core.protocol import ProtocolStandardSmartseq3
        # Forward read
        pos = ProtocolStandardSmartseq3.get_read_position(100, 199, False, "-")
        self.assertEqual(pos, 100)
        # Reverse read
        pos = ProtocolStandardSmartseq3.get_read_position(100, 199, True, "-")
        self.assertEqual(pos, 100)


class TestDefaultGetReadPosition(unittest.TestCase):
    """Verify default get_read_position gives directional start."""

    def test_forward_read(self):
        from utrcalling.core.protocol import Protocol
        pos = Protocol.get_read_position(100, 199, False, "+")
        self.assertEqual(pos, 100)

    def test_reverse_read(self):
        from utrcalling.core.protocol import Protocol
        pos = Protocol.get_read_position(100, 199, True, "+")
        self.assertEqual(pos, 199)


if __name__ == "__main__":
    unittest.main()
