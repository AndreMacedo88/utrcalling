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
        self.assertEqual(stats["duplicate_flag"], 1)
        self.assertEqual(stats["secondary"], 1)

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
            _alignment("unmapped", 1100, flag=paired | first | 0x4 | 0x8, umi="UMI6"),
        ]
        _, sizes, stats = self._map(
            alignments, protocol_, {"UTR1": _IntervalData("1", 1000, 1500, "+")}
        )
        self.assertEqual(len(sizes), 1)
        self.assertEqual(stats["unmapped"], 1)
        self.assertEqual(stats["other_mate"], 1)
        self.assertEqual(stats["not_proper_pair"], 1)
        self.assertEqual(stats["duplicate_flag"], 1)
        self.assertEqual(stats["secondary"], 1)


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
        self.assertEqual(stats["clamped_to_utr_end"], 1)


class TestMappingSummary(_MappingTest):
    """The mapping summary counts each alignment once: where it was skipped, or kept."""

    def test_every_alignment_lands_in_one_step(self):
        protocol_ = protocol.create("standard_10x", 3, annotations=ANNOTATIONS)
        no_bases = _alignment("no_bases", 1100, umi="UMI6")
        no_bases.cigar = None
        no_barcode = _alignment("no_barcode", 1100, umi="UMI7")
        no_barcode.tags = [("UB", "UMI7")]
        alignments = [
            _alignment("kept", 1100),
            _alignment("same_molecule", 1100),  # same barcode and UMI as "kept"
            _alignment("unmapped", 1100, flag=0x4, umi="UMI2"),
            _alignment("duplicate", 1100, flag=0x400, umi="UMI3"),
            _alignment("secondary", 1100, flag=0x100, umi="UMI4"),
            _alignment("antisense", 1100, flag=0x10, umi="UMI5"),  # this protocol reads sense
            no_bases,
            no_barcode,
            _alignment("no_umi", 1100, umi=None),
            _alignment("past_the_end", 1450, umi="UMI8"),  # ends at 1549, after the UTR
        ]
        _, sizes, stats = self._map(
            alignments, protocol_, {"UTR1": _IntervalData("1", 1000, 1500, "+")}
        )
        self.assertListEqual(list(stats), list(utr_mapping.MAPPING_STEPS))
        expected = {
            "alignments": 10,
            "unmapped": 1,
            "other_mate": 0,
            "not_proper_pair": 0,
            "duplicate_flag": 1,
            "secondary": 1,
            "wrong_strand": 1,
            "no_aligned_bases": 1,
            "outside_utr": 1,
            "no_barcode": 1,
            "no_umi": 1,
            "reads": 2,
            "clamped_to_utr_end": 0,
            "molecules": 1,
        }
        self.assertDictEqual(stats, expected)
        self.assertListEqual(sizes, [199])

    def test_minus_strand_utrs_count_reads_outside_them(self):
        # The UTR on 1001-1500, - strand, ends at 1001: a sense read's position
        # is its first base. The read starting at 951 lies past the UTR's end.
        protocol_ = protocol.create("standard_10x", 3, annotations=ANNOTATIONS)
        alignments = [
            _alignment("inside", 1100, flag=0x10),
            _alignment("past_the_end", 950, flag=0x10, umi="UMI2"),
        ]
        _, sizes, stats = self._map(
            alignments, protocol_, {"UTR1": _IntervalData("1", 1000, 1500, "-")}
        )
        self.assertEqual(stats["outside_utr"], 1)
        self.assertEqual(stats["molecules"], 1)
        self.assertEqual(len(sizes), 1)

    def test_map_reads_to_utr_attaches_the_summary_of_all_workers(self):
        # More than 50 UTRs map in parallel: the workers' counts add up.
        protocol_ = protocol.create("standard_10x", 3, annotations=ANNOTATIONS)
        path = self.tmp / "reads.bam"
        header = {"HD": {"VN": "1.0", "SO": "coordinate"}, "SQ": [{"SN": "1", "LN": 100_000}]}
        intervals = {}
        with pysam.AlignmentFile(str(path), "wb", header=header) as out:
            for utr in range(60):
                start = 1000 * (utr + 1)
                intervals[f"UTR{utr}"] = _IntervalData("1", start, start + 500, "+")
                out.write(_alignment(f"read{utr}", start + 100, umi=f"UMI{utr}"))
        pysam.index(str(path))
        mappings = utr_mapping.map_reads_to_utr(
            str(path), intervals, protocol_, read=protocol_.read_in_pair, n_jobs=2
        )
        summary = mappings.attrs["mapping_summary"]
        self.assertListEqual(list(summary), list(utr_mapping.MAPPING_STEPS))
        self.assertEqual(summary["alignments"], 60)
        self.assertEqual(summary["reads"], 60)
        self.assertEqual(summary["molecules"], len(mappings))


if __name__ == "__main__":
    unittest.main()
