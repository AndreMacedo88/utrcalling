"""Which alignments the read loop counts: SAM flags, UMIs, contigs and clamping."""

import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import pysam

from utrcalling.core import protocol, utr_mapping
from utrcalling.core.utr_mapping import _IntervalData

BARCODE = "AAACCCAAGAAACACT-1"
ANNOTATIONS = pd.DataFrame({"sample": ["test"]})


def _alignment(name, start, flag=0, umi="UMI1", length=100, mate_start=None, tlen=0):
    """A mapped read on contig "1" with CB/UB tags (no UB when *umi* is None)."""
    a = pysam.AlignedSegment()
    a.query_name = name
    a.query_sequence = "A" * length
    a.flag = flag
    a.reference_id = 0
    a.reference_start = start
    a.mapping_quality = 255
    a.cigar = [(0, length)]
    a.query_qualities = pysam.qualitystring_to_array("C" * length)
    a.tags = [("CB", BARCODE)] + ([("UB", umi)] if umi else [])
    if mate_start is not None:
        a.next_reference_id = 0
        a.next_reference_start = mate_start
        a.template_length = tlen
    return a


class _MappingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)

    def _map(self, alignments, protocol_, intervals):
        """Map *alignments* to *intervals*: (utr names, UTR sizes, mapping stats)."""
        path = str(self.tmp / "reads.bam")
        header = {"HD": {"VN": "1.0", "SO": "coordinate"}, "SQ": [{"SN": "1", "LN": 10_000}]}
        with pysam.AlignmentFile(path, "wb", header=header) as out:
            for a in sorted(alignments, key=lambda a: a.reference_start):
                out.write(a)
        pysam.index(path)
        names, _, sizes, stats = utr_mapping._process_interval_batch(
            path, list(intervals.items()), protocol_, protocol_.read_in_pair, batch_idx=None
        )
        return names, sizes, stats


class TestAnyReadProtocols(_MappingTest):
    """Protocols that take any read (here Cell Ranger BAMs) skip duplicates and the like."""

    def setUp(self):
        super().setUp()
        self.protocol = protocol.create("standard_10x", 3, annotations=ANNOTATIONS)
        self.utr = {"UTR1": _IntervalData("1", 1000, 1500, "+")}
        # A molecule sits at the 3' end of its read: 1100 + 100 - 1, 199 nt into the UTR.
        self.size = 199

    def test_duplicates_secondary_and_unmapped_reads_are_skipped(self):
        alignments = [
            _alignment("kept", 1100),
            _alignment("duplicate", 1100, flag=0x400, umi="UMI2"),
            _alignment("secondary", 1100, flag=0x100, umi="UMI3"),
            _alignment("unmapped", 1100, flag=0x4, umi="UMI4"),  # placed, like a mate's
        ]
        _, sizes, stats = self._map(alignments, self.protocol, self.utr)
        self.assertListEqual(sizes, [self.size])
        self.assertEqual(stats["PCR duplicates"], 1)
        self.assertEqual(stats["Alignments not primary alignment"], 1)

    def test_a_read_without_a_umi_is_skipped(self):
        alignments = [_alignment("kept", 1100), _alignment("no_umi", 1200, umi=None)]
        _, sizes, _ = self._map(alignments, self.protocol, self.utr)
        self.assertListEqual(sizes, [self.size])

    def test_a_utr_on_a_contig_missing_from_the_bam_is_skipped(self):
        # pysam cannot fetch an unknown contig: that UTR maps nothing, the rest go on.
        intervals = {"ELSEWHERE": _IntervalData("2", 1000, 1500, "+"), **self.utr}
        names, sizes, _ = self._map([_alignment("kept", 1100)], self.protocol, intervals)
        self.assertListEqual(names, ["UTR1"])
        self.assertListEqual(sizes, [self.size])


class TestFirstReadProtocols(_MappingTest):
    """Protocols that take read 1 of proper pairs (here 5' 10x) skip all other reads."""

    def test_only_primary_unduplicated_first_mates_of_proper_pairs_count(self):
        protocol_ = protocol.create("5prime10x", 1, annotations=ANNOTATIONS)
        first, second, paired, proper = 0x40, 0x80, 0x1, 0x2
        alignments = [
            _alignment("kept", 1100, flag=paired | proper | first),
            _alignment("second_mate", 1100, flag=paired | proper | second, umi="UMI2"),
            _alignment("not_proper", 1100, flag=paired | first, umi="UMI3"),
            _alignment("duplicate", 1100, flag=paired | proper | first | 0x400, umi="UMI4"),
            _alignment("secondary", 1100, flag=paired | proper | first | 0x100, umi="UMI5"),
        ]
        _, sizes, stats = self._map(
            alignments, protocol_, {"UTR1": _IntervalData("1", 1000, 1500, "+")}
        )
        self.assertEqual(len(sizes), 1)
        self.assertEqual(stats["Reads not 'first'"], 1)
        self.assertEqual(stats["Alignments not proper pair"], 1)
        self.assertEqual(stats["PCR duplicates"], 1)
        self.assertEqual(stats["Alignments not primary alignment"], 1)


class TestFragmentClampingOnTheMinusStrand(_MappingTest):
    """bulk_pe places fragments running past a - strand UTR's 3' end at that end."""

    def test_fragment_past_the_3_prime_end_is_clamped(self):
        protocol_ = protocol.create("bulk_pe", 1, annotations=ANNOTATIONS)
        # UTR on 1001-1500 (1-based), - strand: its 3' end is genome position 1001.
        # The fragment spans 951-1200, so its distal base lies 50 nt beyond it.
        fragment = _alignment(
            "fragment", 950, flag=0x1 | 0x2 | 0x40 | 0x20, umi=None, mate_start=1100, tlen=250
        )
        _, sizes, stats = self._map(
            [fragment], protocol_, {"UTR1": _IntervalData("1", 1000, 1500, "-")}
        )
        self.assertListEqual(sizes, [500])  # the whole UTR: 1500 - 1000
        self.assertEqual(stats["Positions clamped to the UTR end"], 1)


if __name__ == "__main__":
    unittest.main()
