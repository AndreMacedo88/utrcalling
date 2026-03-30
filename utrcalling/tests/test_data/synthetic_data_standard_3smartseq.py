"""
Synthetic BAM data for standard SmartSeq3 protocol — internal reads, no UMI.

In standard SmartSeq3, internal reads from random priming can map to either
strand. There are no barcodes or UMIs on these reads. Each BAM file represents
a single cell/sample. The observation ID is derived from the BAM file path.

Uses the same UTR layout as the other synthetic generators so the same
``_SYNTHETIC_UTR_REGIONS_3PRIME`` can be reused in pipeline tests.

Key test scenarios:
- Reads on BOTH strands mapping within the same UTR
- No UMI tags or barcodes
- Distal position extraction (furthest toward 3' end)
- Both + and - strand UTRs
- Multiple reads per UTR (no deduplication)
"""

import os
from tempfile import mkstemp

import pysam


class Bam:
    """Generate a standard SmartSeq3 paired-end BAM (one cell, no UMI).

    UTR layout (matches _SYNTHETIC_UTR_REGIONS_3PRIME):
        UTR1: chr1, 3102600-3102750, + strand  (no reads)
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

        with pysam.AlignmentFile(self.alignment, "wb", header=header) as outf:
            # ---------------------------------------------------------------
            # UTR3 (chr16, 91270659-91271933, + strand)
            # Interval after generate_utr_intervals: start=91270658, end=91271933
            #   start_d=91270658, end_d=91271933
            # Distal position = reference_end - 1 (rightmost) for + strand
            # utr_size = abs(distal_pos - start_d) = abs(distal_pos - 91270658)
            # ---------------------------------------------------------------

            # Read 1: forward read covering [91271000..91271099], + strand
            # distal_pos = reference_end - 1 = 91271099
            # utr_size = |91271099 - 91270658| = 441
            a = pysam.AlignedSegment()
            a.query_name = "internal_read_1"
            a.query_sequence = "A" * 100
            a.flag = 0  # single-end mapped, forward strand
            a.reference_id = 1  # chr16
            a.reference_start = 91271000
            a.mapping_quality = 255
            a.cigar = [(0, 100)]
            a.query_qualities = pysam.qualitystring_to_array("C" * 100)

            # Read 2: reverse read covering [91271200..91271299], - strand
            # distal_pos = reference_end - 1 = 91271299 (still rightmost for + strand UTR)
            # utr_size = |91271299 - 91270658| = 641
            b = pysam.AlignedSegment()
            b.query_name = "internal_read_2"
            b.query_sequence = "T" * 100
            b.flag = 0x10  # reverse strand
            b.reference_id = 1
            b.reference_start = 91271200
            b.mapping_quality = 255
            b.cigar = [(0, 100)]
            b.query_qualities = pysam.qualitystring_to_array("C" * 100)

            # Read 3: another forward read, different position
            # covers [91271500..91271599]
            # distal_pos = 91271599, utr_size = |91271599 - 91270658| = 941
            c = pysam.AlignedSegment()
            c.query_name = "internal_read_3"
            c.query_sequence = "G" * 100
            c.flag = 0
            c.reference_id = 1
            c.reference_start = 91271500
            c.mapping_quality = 255
            c.cigar = [(0, 100)]
            c.query_qualities = pysam.qualitystring_to_array("C" * 100)

            # ---------------------------------------------------------------
            # UTR2 (chr1, 172196728-172196828, - strand)
            # Interval after generate_utr_intervals: start=172196727, end=172196828
            #   start_d=172196828-1=172196827, end_d=172196727-1=172196726
            # Distal position = reference_start (leftmost) for - strand
            # utr_size = abs(distal_pos - start_d) = abs(distal_pos - 172196827)
            # ---------------------------------------------------------------

            # Read 4: forward read covering [172196740..172196839], + strand
            # distal_pos = reference_start = 172196740
            # utr_size = |172196740 - 172196827| = 87
            d = pysam.AlignedSegment()
            d.query_name = "internal_read_4"
            d.query_sequence = "C" * 100
            d.flag = 0
            d.reference_id = 0  # chr1
            d.reference_start = 172196740
            d.mapping_quality = 255
            d.cigar = [(0, 100)]
            d.query_qualities = pysam.qualitystring_to_array("C" * 100)

            # Read 5: reverse read covering [172196730..172196829], - strand
            # distal_pos = reference_start = 172196730
            # utr_size = |172196730 - 172196827| = 97
            e = pysam.AlignedSegment()
            e.query_name = "internal_read_5"
            e.query_sequence = "T" * 100
            e.flag = 0x10  # reverse strand
            e.reference_id = 0
            e.reference_start = 172196730
            e.mapping_quality = 255
            e.cigar = [(0, 100)]
            e.query_qualities = pysam.qualitystring_to_array("C" * 100)

            # Write position-sorted: chr1 by coordinate, then chr16
            for align in [e, d, a, b, c]:
                outf.write(align)

        pysam.index(self.alignment)
        self.index = f"{self.alignment}.bai"

    __enter__ = enter

    def __exit__(self, exc_type, exc_value, traceback):
        os.remove(self.alignment)
        os.remove(self.index)
