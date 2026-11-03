"""Synthetic BAM data for the unstranded paired-end bulk RNA-seq protocol.

``bulk_pe`` differs from ``truseq`` in three testable ways, and this fixture is
built so each one has a dedicated fragment:

``frag_inside``
    A fragment lying entirely within UTR3, both mates overlapping the interval.
    Under ``bulk_pe`` it must yield **one** observation, sized from the
    *fragment's* distal base. Under ``truseq --strandedness unstranded`` the same
    BAM yields **two** observations with different sizes — the double-counting
    the new protocol exists to remove.

``frag_proximal``
    A fragment whose R1 lies entirely *before* UTR3's start, so only R2 is
    returned by a fetch over the interval. It must still be counted: this is why
    ``read_in_pair`` stays ``'any'`` rather than being narrowed to ``'first'``.

``frag_overrun``
    A fragment whose distal end runs past UTR3's annotated 3' end. It must be
    clamped onto the UTR end rather than dropped, and the clamp must be counted.

``read_unpaired``
    A read with no pair flags at all, exercising the fallback to single-read
    positioning (identical to ``ProtocolTruseq``).

``frag_minus``
    A fragment on the ``-`` strand UTR2, so the strand-dependent branch of the
    fragment arithmetic is covered too.

UTR layout (shared with the other synthetic generators)::

    UTR1: chr1,  3102600-3102750,   + strand   (no reads)
    UTR2: chr1,  172196728-172196828, - strand
    UTR3: chr16, 91270659-91271933, + strand

``generate_utr_intervals`` shifts start by -1, so the directional bounds are
UTR3: ``start_d = 91270658``, ``end_d = 91271933``;
UTR2: ``start_d = 172196827``, ``end_d = 172196726``.

Expected ``bulk_pe`` sizes::

    UTR3: 551 (frag_inside), 141 (frag_proximal), 1275 (frag_overrun, clamped),
          841 (read_unpaired)
    UTR2: 77 (frag_minus)

Expected ``truseq --strandedness unstranded`` sizes on the same BAM (one per
*mate*, not per fragment)::

    UTR3: 441 + 551 (frag_inside), 141 (frag_proximal only R2 is fetched),
          1241 + 1291 (frag_overrun), 841 (read_unpaired)
    UTR2: 77 + 37 (frag_minus)
"""

import os
from tempfile import mkstemp

import pysam

# Shared header: reference_id 0 -> chr1, reference_id 1 -> chr16
_HEADER = {
    "HD": {"VN": "1.0"},
    "SQ": [{"LN": 195471971, "SN": "1"}, {"LN": 98207768, "SN": "16"}],
}

# Flag components
PAIRED = 0x1
PROPER = 0x2
REVERSE = 0x10
FIRST = 0x40
SECOND = 0x80


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


def _pair(name, reference_id, left_start, right_start, length=100):
    """Build an FR proper pair; R1 is the leftmost mate, R2 the rightmost.

    ``TLEN`` follows the SAM definition: rightmost end - leftmost start, signed
    positive on the leftmost mate.
    """
    tlen = (right_start + length) - left_start
    r1 = _aln(name, reference_id, left_start, PAIRED | PROPER | FIRST, length)
    r1.next_reference_id = reference_id
    r1.next_reference_start = right_start
    r1.template_length = tlen
    r2 = _aln(name, reference_id, right_start, PAIRED | PROPER | SECOND | REVERSE, length)
    r2.next_reference_id = reference_id
    r2.next_reference_start = left_start
    r2.template_length = -tlen
    return r1, r2


class Bam:
    """Generate an unstranded paired-end bulk RNA-seq BAM (one sample)."""

    def __init__(self) -> None:
        fd, self.alignment = mkstemp(suffix=".bam", dir="./test_data/")
        os.close(fd)

    def enter(self):
        with pysam.AlignmentFile(self.alignment, "wb", header=_HEADER) as outf:
            # ---- UTR2 (chr1, - strand), start_d = 172196827 -----------------
            # fragment [172196750, 172196820) -> distal (- strand) = 172196750
            # size = |172196750 - 172196827| = 77
            m1, m2 = _pair("frag_minus", 0, 172196750, 172196790, length=30)

            # ---- UTR3 (chr16, + strand), start_d = 91270658, end_d = 91271933
            # fully inside: fragment [91271000, 91271210)
            # distal = 91271000 + 210 - 1 = 91271209 -> size 551
            a1, a2 = _pair("frag_inside", 1, 91271000, 91271110)

            # straddles the proximal boundary: R1 [91270500, 91270600) is before
            # the UTR start, so only R2 [91270700, 91270800) is fetched.
            # distal = 91270500 + 300 - 1 = 91270799 -> size 141
            p1, p2 = _pair("frag_proximal", 1, 91270500, 91270700)

            # overruns the distal end: fragment [91271800, 91271950)
            # distal = 91271949 > end_d 91271933 -> clamped to 91271933
            # size = 91271933 - 91270658 = 1275
            o1, o2 = _pair("frag_overrun", 1, 91271800, 91271850)

            # unpaired read: no pair flags -> single-read fallback
            # distal = 91271400 + 100 - 1 = 91271499 -> size 841
            u = _aln("read_unpaired", 1, 91271400, 0x0)

            # position-sorted: chr1 (ref 0) first, then chr16 (ref 1)
            for align in [m1, m2, p1, p2, a1, a2, u, o1, o2]:
                outf.write(align)

        pysam.index(self.alignment)
        self.index = f"{self.alignment}.bai"

    __enter__ = enter

    def __exit__(self, exc_type, exc_value, traceback):
        os.remove(self.alignment)
        os.remove(self.index)
