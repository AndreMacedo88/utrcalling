"""Tests for standard 3' 10x protocol UTR mapping.

Validates Read 2 selection, CB/UB tag extraction, keep-max deduplication,
and correct UTR size calculation from Cell Ranger-style BAM files.
"""

import os
import unittest

import pandas as pd

from utrcalling.core import protocol, utr_mapping
from utrcalling.tests.test_data import synthetic_data_standard_3tenx


class TestMapReadsToUtrSynthetic(unittest.TestCase):
    """Base class providing shared UTR region setup."""

    @classmethod
    def setUpClass(cls):
        os.chdir(os.path.join(os.path.dirname(__file__), ".."))
        # Same UTR layout as the modified 10x tests
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


class TestStandardTenxV3(TestMapReadsToUtrSynthetic):
    """Test standard 10x v3 protocol mapping with synthetic BAM."""

    def setUp(self) -> None:
        self.maxDiff = None
        self.data = synthetic_data_standard_3tenx.Bam()
        protocol_ = protocol.create(
            platform="standard_10x",
            version=3,
            alignments=self.data.alignment,
            annotations=pd.DataFrame({"sample": ["test"]}),
        )
        with self.data:
            self.mappings = utr_mapping.map_reads_to_utr(
                self.data.alignment,
                self.intervals,
                protocol_,
                read=protocol_.read_in_pair,
            )

    def test_read_2_selected(self):
        """Protocol should select Read 2 (flag 0x80)."""
        p = protocol.create(
            "standard_10x", 3,
            annotations=pd.DataFrame({"sample": ["test"]}),
        )
        self.assertEqual(p.read_in_pair, "second")

    def test_protocol_properties(self):
        p = protocol.create(
            "standard_10x", 3,
            annotations=pd.DataFrame({"sample": ["test"]}),
        )
        self.assertTrue(p.has_umi)
        self.assertEqual(p.dedup_strategy, "keep_max")
        self.assertEqual(p.read_in_pair, "second")

    def test_utrs_found(self):
        """Should find reads mapping to UTR2 and UTR3."""
        utrs = sorted(self.mappings["utr_name"].unique().tolist())
        self.assertEqual(len(utrs), 2)
        self.assertIn("UTR2", utrs)
        self.assertIn("UTR3", utrs)

    def test_keep_max_dedup_utr3(self):
        """Cell A has two fragments with same UMI but different positions.
        Keep-max should retain the one with the larger utr_size (1261).
        Cell A also has a second molecule with a different UMI (size 1131).
        So UTR3 should have 2 entries for cell A after dedup."""
        utr3 = self.mappings[self.mappings["utr_name"] == "UTR3"]
        # After keep-max dedup: 2 entries (one per UMI)
        self.assertEqual(len(utr3), 2)

        sizes = sorted(utr3["utr_size"].tolist())
        # UMI1 (AACGATTTCCGA): max of 1231 and 1261 = 1261
        # UMI2 (TTGCAATTGGCC): 1131
        self.assertEqual(sizes, [1131, 1261])

    def test_utr2_observation(self):
        """UTR2 should have 1 read from cell B."""
        utr2 = self.mappings[self.mappings["utr_name"] == "UTR2"]
        self.assertEqual(len(utr2), 1)
        self.assertEqual(utr2["observation_id"].iloc[0], "TTTGGGTTCAACGTTT-1")
        self.assertEqual(utr2["utr_size"].iloc[0], 87)

    def test_observation_ids_are_cb_tags(self):
        """Observation IDs should come from CB BAM tags."""
        obs_ids = self.mappings["observation_id"].unique().tolist()
        self.assertIn("AAACCCAAGAAACACT-1", obs_ids)
        self.assertIn("TTTGGGTTCAACGTTT-1", obs_ids)


class TestStandardTenxV2(TestMapReadsToUtrSynthetic):
    """Standard 10x v2 should behave identically to v3 for Cell Ranger BAMs."""

    def setUp(self) -> None:
        self.data = synthetic_data_standard_3tenx.Bam()
        protocol_ = protocol.create(
            platform="standard_10x",
            version=2,
            alignments=self.data.alignment,
            annotations=pd.DataFrame({"sample": ["test"]}),
        )
        with self.data:
            self.mappings = utr_mapping.map_reads_to_utr(
                self.data.alignment,
                self.intervals,
                protocol_,
                read=protocol_.read_in_pair,
            )

    def test_identical_to_v3(self):
        """v2 and v3 produce identical results for Cell Ranger BAMs."""
        utrs = sorted(self.mappings["utr_name"].unique().tolist())
        self.assertEqual(len(utrs), 2)


class TestStandardTenxMissingTags(TestMapReadsToUtrSynthetic):
    """Test that reads missing CB/UB tags are skipped gracefully."""

    def test_missing_tags_skipped(self):
        """Reads without CB or UB tags should be silently skipped."""
        import os
        from tempfile import mkstemp
        import pysam

        fd, bam_path = mkstemp(suffix=".bam", dir="./test_data/")
        os.close(fd)
        header = {
            "HD": {"VN": "1.0"},
            "SQ": [{"LN": 195471971, "SN": "1"}, {"LN": 98207768, "SN": "16"}],
        }
        with pysam.AlignmentFile(bam_path, "wb", header=header) as outf:
            # Read 2 WITHOUT CB/UB tags
            a = pysam.AlignedSegment()
            a.query_name = "no_tags"
            a.query_sequence = "A" * 90
            a.flag = 0x93  # paired, proper pair, reverse, second in pair
            a.reference_id = 1
            a.reference_start = 91271800
            a.mapping_quality = 255
            a.cigar = [(0, 90)]
            a.next_reference_id = 1
            a.next_reference_start = 91271800
            a.template_length = 0
            a.query_qualities = pysam.qualitystring_to_array("C" * 90)
            # No CB/UB tags!
            outf.write(a)

        pysam.index(bam_path)

        try:
            protocol_ = protocol.create(
                "standard_10x", 3,
                annotations=pd.DataFrame({"sample": ["test"]}),
            )
            mappings = utr_mapping.map_reads_to_utr(
                bam_path, self.intervals, protocol_,
                read=protocol_.read_in_pair,
            )
            self.assertEqual(len(mappings), 0)
        finally:
            os.remove(bam_path)
            os.remove(f"{bam_path}.bai")


if __name__ == "__main__":
    unittest.main()
