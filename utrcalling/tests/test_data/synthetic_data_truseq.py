"""Synthetic BAM data for the TruSeq stranded bulk RNA-seq protocol.

TruSeq Stranded mRNA is reverse-stranded (dUTP): the read maps **antisense**
to the transcript. There are no cell barcodes and no UMIs. Each BAM file is one
sample; the observation ID is derived from the BAM path (resolved to the
annotation ``sample_id``). Reads are randomly fragmented, so the position
furthest toward the 3' end of the UTR is used as a per-read size estimate.

Uses the same UTR layout as the other synthetic generators so the same
``_SYNTHETIC_UTR_REGIONS_3PRIME`` can be reused in pipeline tests.

Strand handling under each mode (``check_strand_matches``):
- reverse  -> keep reads whose strand differs from the UTR (antisense)
- forward  -> keep reads whose strand equals the UTR (sense)
- unstranded -> keep both

Expected results (sizes via distal-by-UTR-strand position, then
``abs(read_pos - interval.start_d)``):

============  =================================  ================
mode          UTR3 (+, chr16) sizes              UTR2 (-, chr1) sizes
============  =================================  ================
reverse        441, 641, 941 (antisense reads)    77, 87 (antisense reads)
forward        741 (sense read)                   67 (sense read)
unstranded     441, 641, 741, 941                 67, 77, 87
============  =================================  ================
"""

import os
from tempfile import mkstemp

import pysam

# Shared header: reference_id 0 -> chr1, reference_id 1 -> chr16
_HEADER = {
    "HD": {"VN": "1.0"},
    "SQ": [{"LN": 195471971, "SN": "1"}, {"LN": 98207768, "SN": "16"}],
}


def _aln(name, reference_id, reference_start, flag, length=100):
    """Build a simple fully-matched (``length``M) AlignedSegment."""
    a = pysam.AlignedSegment()
    a.query_name = name
    a.query_sequence = "A" * length
    a.flag = flag
    a.reference_id = reference_id
    a.reference_start = reference_start
    a.mapping_quality = 255
    a.cigar = [(0, length)]
    a.query_qualities = pysam.qualitystring_to_array("F" * length)
    return a


class Bam:
    """Generate a single-end TruSeq BAM (one sample, no UMI/barcode).

    UTR layout (matches _SYNTHETIC_UTR_REGIONS_3PRIME):
        UTR1: chr1, 3102600-3102750, + strand  (no reads)
        UTR2: chr1, 172196728-172196828, - strand
        UTR3: chr16, 91270659-91271933, + strand
    """

    def __init__(self) -> None:
        fd, self.alignment = mkstemp(suffix=".bam", dir="./test_data/")
        os.close(fd)

    def enter(self):
        with pysam.AlignmentFile(self.alignment, "wb", header=_HEADER) as outf:
            # -------------------------------------------------------------
            # UTR3 (chr16, + strand). start_d = 91270658.
            # read_pos = reference_end - 1 (distal for + UTR).
            # antisense reads map to the - strand -> reverse (flag 0x10).
            # -------------------------------------------------------------
            # A: reverse, [91271000..91271100) -> read_pos 91271099 -> size 441
            a = _aln("utr3_antisense_1", 1, 91271000, 0x10)
            # B: reverse, [91271200..91271300) -> read_pos 91271299 -> size 641
            b = _aln("utr3_antisense_2", 1, 91271200, 0x10)
            # SENSE3: forward (sense) -> rejected in reverse mode.
            #   [91271300..91271400) -> read_pos 91271399 -> size 741 (fwd/unstr)
            sense3 = _aln("utr3_sense", 1, 91271300, 0x0)
            # C: reverse, [91271500..91271600) -> read_pos 91271599 -> size 941
            c = _aln("utr3_antisense_3", 1, 91271500, 0x10)

            # -------------------------------------------------------------
            # UTR2 (chr1, - strand). start_d = 172196828 - 1 = 172196827.
            # read_pos = reference_start (distal for - UTR).
            # antisense reads map to the + strand -> forward (flag 0).
            # -------------------------------------------------------------
            # D: forward, reference_start 172196740 -> size |.-172196827| = 87
            d = _aln("utr2_antisense_1", 0, 172196740, 0x0)
            # E: forward, reference_start 172196750 -> size 77
            e = _aln("utr2_antisense_2", 0, 172196750, 0x0)
            # SENSE2: reverse (sense) -> rejected in reverse mode.
            #   reference_start 172196760 -> size 67 (fwd/unstranded)
            sense2 = _aln("utr2_sense", 0, 172196760, 0x10)

            # Write position-sorted: chr1 (ref 0) then chr16 (ref 1)
            for align in [d, e, sense2, a, b, sense3, c]:
                outf.write(align)

        pysam.index(self.alignment)
        self.index = f"{self.alignment}.bai"

    __enter__ = enter

    def __exit__(self, exc_type, exc_value, traceback):
        os.remove(self.alignment)
        os.remove(self.index)


class BamPaired:
    """Generate a paired-end TruSeq BAM with a single fragment on UTR3.

    Demonstrates that with reverse strandedness and ``read_in_pair='any'`` the
    sense mate (R2) is dropped by the strand filter, so the fragment is counted
    exactly once (via the antisense R1).

        R1 (antisense, reverse, first-in-pair)  -> kept,  size 441
        R2 (sense, forward, second-in-pair)     -> rejected (sense on + UTR)
    """

    def __init__(self) -> None:
        fd, self.alignment = mkstemp(suffix=".bam", dir="./test_data/")
        os.close(fd)

    def enter(self):
        with pysam.AlignmentFile(self.alignment, "wb", header=_HEADER) as outf:
            # R1: paired + proper + first + reverse = 0x1|0x2|0x40|0x10 = 83
            r1 = _aln("frag1", 1, 91271000, 0x1 | 0x2 | 0x40 | 0x10)
            r1.next_reference_id = 1
            r1.next_reference_start = 91271010
            r1.template_length = -110
            # R2: paired + proper + second + forward = 0x1|0x2|0x80 = 131
            r2 = _aln("frag1", 1, 91271010, 0x1 | 0x2 | 0x80)
            r2.next_reference_id = 1
            r2.next_reference_start = 91271000
            r2.template_length = 110

            for align in [r1, r2]:
                outf.write(align)

        pysam.index(self.alignment)
        self.index = f"{self.alignment}.bai"

    __enter__ = enter

    def __exit__(self, exc_type, exc_value, traceback):
        os.remove(self.alignment)
        os.remove(self.index)
