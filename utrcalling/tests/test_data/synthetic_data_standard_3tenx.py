"""
Synthetic BAM data mimicking Cell Ranger output for standard 3' 10x sequencing.

In standard 10x 3', Read 1 is ~28bp (barcode+UMI only) and Read 2 maps to
the 3'UTR. Cell Ranger writes the corrected barcode into the CB tag and the
corrected UMI into the UB tag. Read 2 is the positional read.

The BAM uses the same chromosome/UTR layout as ``synthetic_data_3tenx.py``
so the same ``_SYNTHETIC_UTR_REGIONS_3PRIME`` can be reused in pipeline tests.

Key test scenarios:
- Two cells (different CB values)
- Same UMI with different fragment positions (tests keep-max dedup)
- Reads on opposite strand from UTR (standard for 3' protocols)
- Both + and - strand UTRs
"""

import os
from tempfile import mkstemp

import pysam


class Bam:
    """Generate a Cell Ranger-style BAM for standard 3' 10x v3.

    UTR layout (matches _SYNTHETIC_UTR_REGIONS_3PRIME):
        UTR1: chr1, 3102600-3102750, + strand  (no reads target this UTR)
        UTR2: chr1, 172196728-172196828, - strand
        UTR3: chr16, 91270659-91271933, + strand
    """

    def __init__(self) -> None:
        fd, self.alignment = mkstemp(suffix=".bam", dir="./test_data/")
        os.close(fd)

    def enter(self):
        header = {
            "HD": {"VN": "1.0"},
            "SQ": [{"LN": 195471971, "SN": "1"}, {"LN": 98207768, "SN": "16"}],
        }

        qual_r1 = pysam.qualitystring_to_array("CCCCCCCCCCCCCCCCCCCCCCCCCCCC")  # 28bp
        qual_r2 = "C" * 90

        with pysam.AlignmentFile(self.alignment, "wb", header=header) as outf:
            # ---------------------------------------------------------------
            # UTR3 (chr16, 91270659-91271933, + strand)
            # For + strand 3' UTR, reads map on the - strand.
            # read_start_d = reference_end - 1  (the rightmost base)
            # utr_size = abs(read_start_d - interval.start_d)
            #          = abs(read_start_d - (91270659 - 1))   [note: -1 from generate_utr_intervals]
            #          = abs(read_start_d - 91270658)
            # ---------------------------------------------------------------

            # Cell A, molecule 1 — two fragments with SAME UMI but different
            # positions.  Keep-max dedup should keep the one with the larger
            # utr_size (= fragment extending further toward the polyA / right).

            # Fragment 1: R2 covers [91271800..91271889], reversed
            # read_start_d = 91271889, utr_size = |91271889 - 91270658| = 1231
            a2 = pysam.AlignedSegment()
            a2.query_name = "read_A1_frag1"
            a2.query_sequence = "A" * 90
            a2.flag = 0x93  # paired, proper pair, reverse, second in pair
            a2.reference_id = 1  # chr16
            a2.reference_start = 91271800
            a2.mapping_quality = 255
            a2.cigar = [(0, 90)]
            a2.next_reference_id = 1
            a2.next_reference_start = 91271800
            a2.template_length = 0
            a2.query_qualities = pysam.qualitystring_to_array(qual_r2)
            a2.tags = [("CB", "AAACCCAAGAAACACT-1"), ("UB", "AACGATTTCCGA")]

            # R1 (unmapped mate, just barcode+UMI — not used for mapping)
            a1 = pysam.AlignedSegment()
            a1.query_name = "read_A1_frag1"
            a1.query_sequence = "AAACCCAAGAAACACT" + "AACGATTTCCGA"  # BC + UMI
            a1.flag = 0x43  # paired, proper pair, first in pair
            a1.reference_id = 1
            a1.reference_start = 91271800
            a1.mapping_quality = 0
            a1.cigar = [(4, 28)]  # all soft-clipped
            a1.next_reference_id = 1
            a1.next_reference_start = 91271800
            a1.template_length = 0
            a1.query_qualities = qual_r1
            a1.tags = [("CB", "AAACCCAAGAAACACT-1"), ("UB", "AACGATTTCCGA")]

            # Fragment 2 (same UMI, different position): R2 covers [91271850..91271919],
            # reversed.
            # read_start_d = 91271919, utr_size = |91271919 - 91270658| = 1261
            # This is LARGER → should be kept by keep-max dedup.
            b2 = pysam.AlignedSegment()
            b2.query_name = "read_A1_frag2"
            b2.query_sequence = "A" * 70
            b2.flag = 0x93  # paired, proper pair, reverse, second in pair
            b2.reference_id = 1
            b2.reference_start = 91271850
            b2.mapping_quality = 255
            b2.cigar = [(0, 70)]
            b2.next_reference_id = 1
            b2.next_reference_start = 91271850
            b2.template_length = 0
            b2.query_qualities = pysam.qualitystring_to_array("C" * 70)
            b2.tags = [("CB", "AAACCCAAGAAACACT-1"), ("UB", "AACGATTTCCGA")]

            b1 = pysam.AlignedSegment()
            b1.query_name = "read_A1_frag2"
            b1.query_sequence = "AAACCCAAGAAACACT" + "AACGATTTCCGA"
            b1.flag = 0x43
            b1.reference_id = 1
            b1.reference_start = 91271850
            b1.mapping_quality = 0
            b1.cigar = [(4, 28)]
            b1.next_reference_id = 1
            b1.next_reference_start = 91271850
            b1.template_length = 0
            b1.query_qualities = qual_r1
            b1.tags = [("CB", "AAACCCAAGAAACACT-1"), ("UB", "AACGATTTCCGA")]

            # Cell A, molecule 2 — different UMI, one fragment
            # R2 covers [91271700..91271789], reversed
            # read_start_d = 91271789, utr_size = |91271789 - 91270658| = 1131
            c2 = pysam.AlignedSegment()
            c2.query_name = "read_A2"
            c2.query_sequence = "A" * 90
            c2.flag = 0x93  # paired, proper pair, reverse, second in pair
            c2.reference_id = 1
            c2.reference_start = 91271700
            c2.mapping_quality = 255
            c2.cigar = [(0, 90)]
            c2.next_reference_id = 1
            c2.next_reference_start = 91271700
            c2.template_length = 0
            c2.query_qualities = pysam.qualitystring_to_array(qual_r2)
            c2.tags = [("CB", "AAACCCAAGAAACACT-1"), ("UB", "TTGCAATTGGCC")]

            c1 = pysam.AlignedSegment()
            c1.query_name = "read_A2"
            c1.query_sequence = "AAACCCAAGAAACACT" + "TTGCAATTGGCC"
            c1.flag = 0x43
            c1.reference_id = 1
            c1.reference_start = 91271700
            c1.mapping_quality = 0
            c1.cigar = [(4, 28)]
            c1.next_reference_id = 1
            c1.next_reference_start = 91271700
            c1.template_length = 0
            c1.query_qualities = qual_r1
            c1.tags = [("CB", "AAACCCAAGAAACACT-1"), ("UB", "TTGCAATTGGCC")]

            # ---------------------------------------------------------------
            # UTR2 (chr1, 172196728-172196828, - strand)
            # For - strand 3' UTR, reads map on the + strand.
            # Interval after generate_utr_intervals: start_d = 172196828-1 = 172196827
            # read_start_d = reference_start (forward read)
            # utr_size = abs(read_start_d - interval.start_d)
            #          = abs(read_start_d - 172196827)
            # ---------------------------------------------------------------

            # Cell B, molecule 1 — forward read on + strand
            # R2 covers [172196740..172196829], forward
            # read_start_d = 172196740, utr_size = |172196740 - 172196827| = 87
            d2 = pysam.AlignedSegment()
            d2.query_name = "read_B1"
            d2.query_sequence = "G" * 90
            d2.flag = 0xA3  # paired, proper pair, mate reverse, second in pair (forward)
            d2.reference_id = 0  # chr1
            d2.reference_start = 172196740
            d2.mapping_quality = 255
            d2.cigar = [(0, 90)]
            d2.next_reference_id = 0
            d2.next_reference_start = 172196740
            d2.template_length = 0
            d2.query_qualities = pysam.qualitystring_to_array(qual_r2)
            d2.tags = [("CB", "TTTGGGTTCAACGTTT-1"), ("UB", "GCATGCATGCAT")]

            d1 = pysam.AlignedSegment()
            d1.query_name = "read_B1"
            d1.query_sequence = "TTTGGGTTCAACGTTT" + "GCATGCATGCAT"
            d1.flag = 0x43
            d1.reference_id = 0
            d1.reference_start = 172196740
            d1.mapping_quality = 0
            d1.cigar = [(4, 28)]
            d1.next_reference_id = 0
            d1.next_reference_start = 172196740
            d1.template_length = 0
            d1.query_qualities = qual_r1
            d1.tags = [("CB", "TTTGGGTTCAACGTTT-1"), ("UB", "GCATGCATGCAT")]

            # Write position-sorted: chr1 first, then chr16 by coordinate
            for align in [d1, d2, c1, c2, a1, a2, b1, b2]:
                outf.write(align)

        pysam.index(self.alignment)
        self.index = f"{self.alignment}.bai"

    __enter__ = enter

    def __exit__(self, exc_type, exc_value, traceback):
        os.remove(self.alignment)
        os.remove(self.index)
