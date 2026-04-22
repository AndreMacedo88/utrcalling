"""
Synthetic BAM data mimicking Cell Ranger output for standard 3' 10x sequencing.

Cell Ranger's ``possorted_genome_bam.bam`` for 10x Chromium 3' v3 contains
only R2 alignments written as single-end (flag=0 forward / flag=16 reverse).
R1 (cell barcode + UMI) is never aligned and is not present in the BAM; the
corrected values are attached as the ``CB`` and ``UB`` tags on the R2 record.

R2 aligns in the **sense** direction relative to the gene (empirically:
>=98% sense at GAPDH/B2M in real data), so + strand UTRs receive forward
(flag=0) R2s and - strand UTRs receive reverse (flag=16) R2s.

The BAM uses the same chromosome/UTR layout as ``synthetic_data_3tenx.py``
so the same ``_SYNTHETIC_UTR_REGIONS_3PRIME`` can be reused in pipeline tests.

Key test scenarios:
- Two cells (different CB values)
- Same UMI with different fragment positions (tests keep-max dedup)
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

        qual_r2 = "C" * 90

        with pysam.AlignmentFile(self.alignment, "wb", header=header) as outf:
            # ---------------------------------------------------------------
            # UTR3 (chr16, 91270659-91271933, + strand)
            # R2 is sense to + strand gene → forward alignment (flag=0).
            # read_pos = ref_end - 1 (polyA-distal end of R2 = rightmost base)
            # utr_size = abs(read_pos - interval.start_d)
            #          = abs(read_pos - 91270658)    [start_d for + = start-1]
            # ---------------------------------------------------------------

            # Cell A, molecule 1 — two fragments with SAME UMI but different
            # positions. Keep-max dedup should keep the one with the larger
            # utr_size (= fragment extending further toward the polyA / right).

            # Fragment 1: R2 covers [91271800..91271889], forward
            # read_pos = 91271889, utr_size = |91271889 - 91270658| = 1231
            a2 = pysam.AlignedSegment()
            a2.query_name = "read_A1_frag1"
            a2.query_sequence = "A" * 90
            a2.flag = 0x0  # single-end forward (sense to + strand gene)
            a2.reference_id = 1  # chr16
            a2.reference_start = 91271800
            a2.mapping_quality = 255
            a2.cigar = [(0, 90)]
            a2.query_qualities = pysam.qualitystring_to_array(qual_r2)
            a2.tags = [("CB", "AAACCCAAGAAACACT-1"), ("UB", "AACGATTTCCGA")]

            # Fragment 2 (same UMI, different position): R2 covers [91271850..91271919]
            # read_pos = 91271919, utr_size = |91271919 - 91270658| = 1261
            # This is LARGER → should be kept by keep-max dedup.
            b2 = pysam.AlignedSegment()
            b2.query_name = "read_A1_frag2"
            b2.query_sequence = "A" * 70
            b2.flag = 0x0  # single-end forward
            b2.reference_id = 1
            b2.reference_start = 91271850
            b2.mapping_quality = 255
            b2.cigar = [(0, 70)]
            b2.query_qualities = pysam.qualitystring_to_array("C" * 70)
            b2.tags = [("CB", "AAACCCAAGAAACACT-1"), ("UB", "AACGATTTCCGA")]

            # Cell A, molecule 2 — different UMI, one fragment
            # R2 covers [91271700..91271789], forward
            # read_pos = 91271789, utr_size = |91271789 - 91270658| = 1131
            c2 = pysam.AlignedSegment()
            c2.query_name = "read_A2"
            c2.query_sequence = "A" * 90
            c2.flag = 0x0  # single-end forward
            c2.reference_id = 1
            c2.reference_start = 91271700
            c2.mapping_quality = 255
            c2.cigar = [(0, 90)]
            c2.query_qualities = pysam.qualitystring_to_array(qual_r2)
            c2.tags = [("CB", "AAACCCAAGAAACACT-1"), ("UB", "TTGCAATTGGCC")]

            # ---------------------------------------------------------------
            # UTR2 (chr1, 172196728-172196828, - strand)
            # R2 is sense to - strand gene → reverse alignment (flag=16).
            # After generate_utr_intervals: start_d = 172196828 - 1 = 172196827
            # read_pos = ref_start (polyA-distal end of R2 on - strand = leftmost)
            # utr_size = abs(read_pos - 172196827)
            # ---------------------------------------------------------------

            # Cell B, molecule 1 — reverse read on - strand
            # R2 covers [172196740..172196829], reverse
            # read_pos = 172196740, utr_size = |172196740 - 172196827| = 87
            d2 = pysam.AlignedSegment()
            d2.query_name = "read_B1"
            d2.query_sequence = "G" * 90
            d2.flag = 0x10  # single-end reverse (sense to - strand gene)
            d2.reference_id = 0  # chr1
            d2.reference_start = 172196740
            d2.mapping_quality = 255
            d2.cigar = [(0, 90)]
            d2.query_qualities = pysam.qualitystring_to_array(qual_r2)
            d2.tags = [("CB", "TTTGGGTTCAACGTTT-1"), ("UB", "GCATGCATGCAT")]

            # Write position-sorted: chr1 first, then chr16 by coordinate
            for align in [d2, c2, a2, b2]:
                outf.write(align)

        pysam.index(self.alignment)
        self.index = f"{self.alignment}.bai"

    __enter__ = enter

    def __exit__(self, exc_type, exc_value, traceback):
        os.remove(self.alignment)
        os.remove(self.index)
