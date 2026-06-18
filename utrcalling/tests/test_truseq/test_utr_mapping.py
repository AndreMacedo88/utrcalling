"""Tests for the TruSeq stranded bulk RNA-seq protocol UTR mapping.

Validates strand-aware acceptance (reverse / forward / unstranded), distal
position extraction, no deduplication, single- and paired-end handling, and
observation_id resolution to the annotation sample_id.
"""

import os
import unittest
from pathlib import Path

import pandas as pd

from utrcalling.core import protocol, utr_mapping
from utrcalling.tests.test_data import synthetic_data_truseq


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


class _TruseqMappingBase(TestMapReadsToUtrSynthetic):
    """Maps the single-end synthetic BAM under a configurable strandedness."""

    STRANDEDNESS = None  # None -> reverse (the protocol default)

    def setUp(self) -> None:
        self.maxDiff = None
        self.data = synthetic_data_truseq.Bam()
        # sample_id matches the temp BAM stem so it resolves to a friendly name.
        self.sample_id = Path(self.data.alignment).stem
        self.protocol = protocol.create(
            platform="truseq",
            version=1,
            alignments=self.data.alignment,
            annotations=pd.DataFrame({"sample_id": [self.sample_id]}),
            strandedness=self.STRANDEDNESS,
        )
        with self.data:
            self.mappings = utr_mapping.map_reads_to_utr(
                self.data.alignment,
                self.intervals,
                self.protocol,
                read=self.protocol.read_in_pair,
            )

    def _sizes(self, utr_name):
        return sorted(
            self.mappings[self.mappings["utr_name"] == utr_name]["utr_size"].tolist()
        )


class TestTruseqProtocolProperties(_TruseqMappingBase):
    def test_properties(self):
        self.assertFalse(self.protocol.has_umi)
        self.assertEqual(self.protocol.read_in_pair, "any")
        self.assertEqual(self.protocol.dedup_strategy, "keep_first")
        self.assertFalse(self.protocol._reads_have_batch_suffix)

    def test_default_strandedness_is_reverse(self):
        # strandedness is stored as None but interpreted as reverse.
        self.assertIsNone(self.protocol.strandedness)
        self.assertTrue(self.protocol.check_strand_matches("-", "+"))
        self.assertFalse(self.protocol.check_strand_matches("+", "+"))

    def test_observation_id_resolves_to_sample_id(self):
        obs_ids = self.mappings["observation_id"].unique().tolist()
        self.assertEqual(obs_ids, [self.sample_id])


class TestTruseqReverse(_TruseqMappingBase):
    """Default reverse strandedness keeps antisense reads only."""

    STRANDEDNESS = "reverse"

    def test_utrs_found(self):
        self.assertEqual(
            sorted(self.mappings["utr_name"].unique().tolist()), ["UTR2", "UTR3"]
        )

    def test_utr3_sizes(self):
        # + strand UTR, antisense (reverse) reads kept.
        self.assertEqual(self._sizes("UTR3"), [441, 641, 941])

    def test_utr2_sizes(self):
        # - strand UTR, antisense (forward) reads kept.
        self.assertEqual(self._sizes("UTR2"), [77, 87])

    def test_no_dedup_all_antisense_reads_kept(self):
        self.assertEqual(len(self.mappings), 5)

    def test_sense_reads_rejected(self):
        # Sense-read would-be sizes (741 on UTR3, 67 on UTR2) must be absent.
        self.assertNotIn(741, self.mappings["utr_size"].tolist())
        self.assertNotIn(67, self.mappings["utr_size"].tolist())


class TestTruseqForward(_TruseqMappingBase):
    """Forward strandedness keeps sense reads only (mirror image)."""

    STRANDEDNESS = "forward"

    def test_only_sense_reads_kept(self):
        self.assertEqual(len(self.mappings), 2)
        self.assertEqual(self._sizes("UTR3"), [741])
        self.assertEqual(self._sizes("UTR2"), [67])


class TestTruseqUnstranded(_TruseqMappingBase):
    """Unstranded keeps reads from both strands."""

    STRANDEDNESS = "unstranded"

    def test_all_reads_kept(self):
        self.assertEqual(len(self.mappings), 7)
        self.assertEqual(self._sizes("UTR3"), [441, 641, 741, 941])
        self.assertEqual(self._sizes("UTR2"), [67, 77, 87])


class TestTruseqPairedEnd(TestMapReadsToUtrSynthetic):
    """Paired-end input: the sense mate (R2) is dropped, fragment counted once."""

    def setUp(self) -> None:
        self.data = synthetic_data_truseq.BamPaired()
        self.protocol = protocol.create(
            platform="truseq",
            version=1,
            alignments=self.data.alignment,
            annotations=pd.DataFrame(
                {"sample_id": [Path(self.data.alignment).stem]}
            ),
        )
        with self.data:
            self.mappings = utr_mapping.map_reads_to_utr(
                self.data.alignment,
                self.intervals,
                self.protocol,
                read=self.protocol.read_in_pair,
            )

    def test_only_antisense_mate_counted(self):
        # Two alignments overlap UTR3, but only the antisense R1 is kept.
        self.assertEqual(len(self.mappings), 1)
        self.assertEqual(self.mappings["utr_size"].tolist(), [441])


class TestTruseqGetReadPosition(unittest.TestCase):
    """Direct unit tests for the distal-by-UTR-strand position."""

    def test_plus_strand_uses_rightmost(self):
        from utrcalling.core.protocol import ProtocolTruseq

        self.assertEqual(ProtocolTruseq.get_read_position(100, 199, False, "+"), 199)
        self.assertEqual(ProtocolTruseq.get_read_position(100, 199, True, "+"), 199)

    def test_minus_strand_uses_leftmost(self):
        from utrcalling.core.protocol import ProtocolTruseq

        self.assertEqual(ProtocolTruseq.get_read_position(100, 199, False, "-"), 100)
        self.assertEqual(ProtocolTruseq.get_read_position(100, 199, True, "-"), 100)


if __name__ == "__main__":
    unittest.main()
