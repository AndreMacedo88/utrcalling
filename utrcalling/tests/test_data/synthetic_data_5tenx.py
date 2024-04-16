"""
This file contains objects that are specific to the 10x-based protocols.
The goal is to use these objects to test the output of the tool's methods.
They should represent real objects as closely as possible, while maintaining simplicity
for test speed.
"""

import os
from tempfile import mkstemp
import pysam


class Bam:
    """
    Generates synthetic BAM alignment data in conformity with what a 5'10x would look
    like (note that must have sequenced more than 28 bp for read 1)

    Notes:
    - the reverse strand read 1 will have the real 3'UTR start at `reference_start` +
    number of matches in the CIGAR string. This is because the bam file annotates the
    start and the CIGAR string in relation to the genome, but since this read 1 is in
    the reverse strand the start is actually in the other side of the read
    (minus the soft clipped regions)
    """

    def __init__(self) -> None:
        fd, self.alignment = mkstemp(suffix=".bam", dir="./test_data/")

    def enter(self):
        # set-up the temporary alignment file
        header = {
            "HD": {"VN": "1.0"},
            "SQ": [{"LN": 195471971, "SN": "1"}, {"LN": 98207768, "SN": "16"}],
        }

        ########################
        # Create the alignments
        ########################

        with pysam.AlignmentFile(self.alignment, "wb", header=header) as outf:
            # First alignment - read 1
            a1 = pysam.AlignedSegment()
            a1.query_name = "VH00211:4:AAAHCYCM5:1:1609:71042:30325"
            a1.query_sequence = "CCCCAAGAAGAAAAGAGGGTCAAAAGCGGGCCACCGACACACCGCTCTGAGAACAACCACAGATTGTTCCCCCATATAAGAAA"
            a1.flag = 83  # proper pair, first in pair, reverse strand
            a1.reference_id = 1  # index of the chromosome in the header
            a1.reference_start = 91271850  # Add 70 to get start.
            a1.mapping_quality = 255
            # 13 soft-clipped (Barcode, UMI, etc) + 70 matched
            a1.cigartuples = ((0, 70), (4, 13))
            a1.next_reference_id = 1
            a1.next_reference_start = 91271713
            a1.template_length = -160
            a1.query_qualities = pysam.qualitystring_to_array(
                "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"
            )
            a1.tags = (("CB", "ATAGACCCACACAGAG-1"), ("UB", "TATTCATAAT"))
            # First alignment - read 2
            a2 = pysam.AlignedSegment()
            a2.query_name = "VH00211:4:AAAHCYCM5:1:1609:71042:30325"
            a2.query_sequence = "GACTTTGCAGGTTATATTTATGTATTTAGATACACTGTTGGGAGCCATTAAGGCAACGCTATTGTCCTGATCTCTGAATTGGGCCTCTCCCCCCAAGAAGAAAAGAGGG"
            a2.flag = 163  # proper pair, second in pair, mate in reverse strand
            a2.reference_id = 1  # index of the chromosome in the header
            a2.reference_start = 91271713
            a2.mapping_quality = 255
            a2.cigar = ((0, 109),)
            a2.next_reference_id = 1
            a2.next_reference_start = 91271850
            a2.template_length = 160
            a2.query_qualities = pysam.qualitystring_to_array(
                "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC-CCCCCCCCCCC"
            )
            a2.tags = (("CB", "ATAGACCCACACAGAG-1"), ("UB", "TATTCATAAT"))

            # Second alignment - read 1
            b1 = pysam.AlignedSegment()
            b1.query_name = "VH00211:4:AAAHCYCM5:1:2605:56235:5600"
            b1.query_sequence = "TTTCTTATATGGGAAACCAAACTGGTAGAAGATATTAAAGAGGAAACACAAAAATCCCTTAAAGAATTGCAGGAAAACACAAC"
            b1.flag = 99  # proper pair, first in pair, mate in reverse strand
            b1.reference_id = 0
            b1.reference_start = 172196719
            b1.mapping_quality = 255
            # 14 soft-clipped (Barcode, UMI, etc) + 69 matched
            b1.cigar = ((4, 14), (0, 69))
            b1.next_reference_id = 0
            b1.next_reference_start = 172196993
            b1.template_length = 361
            b1.query_qualities = pysam.qualitystring_to_array(
                ";C-CCCCCCC;CCCC;CCCCCCCCCCCCCCC;CCCCC;CCCCCCCCCCCCC;;CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"
            )
            b1.tags = (("CB", "CCTCAGTGTTAAGATG-1"), ("UB", "AAGTTAGGGG"))
            # Second alignment - read 2
            b2 = pysam.AlignedSegment()
            b2.query_name = "VH00211:4:AAAHCYCM5:1:2605:56235:5600"
            b2.query_sequence = "TTCCATAGAGAACATCGGCACAACAATCAAAGAAAATGGAAAATGCAAAAAGATCCTAACTCAAAACATCCAGGAAATCCAGGACACAATGAGAAGACCGAACCTACGG"
            b2.flag = 147  # proper pair, second in pair, reverse strand
            b2.reference_id = 0
            b2.reference_start = 172196993
            b2.mapping_quality = 255
            b2.cigar = ((0, 109),)
            b2.next_reference_id = 0
            b2.next_reference_start = 172196719
            b2.template_length = -361
            b2.query_qualities = pysam.qualitystring_to_array(
                "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC;CCCC-CCCCCCCCCCCCCCCCCCCC"
            )
            b2.tags = (("CB", "CCTCAGTGTTAAGATG-1"), ("UB", "AAGTTAGGGG"))

            # Duplicate of first alignment - read 1
            c1 = pysam.AlignedSegment()
            c1.query_name = "VH00211:4:AAAHCYCM5:1:2105:54019:11375"
            c1.query_sequence = "CCCCAAGAAGAAAAGAGGGTCAAAAGCGGGCCACCGACACACCGCTCTGAGAACAACCACAGATTGTTCCCCCATATAAGAAA"
            c1.flag = 83  # proper pair, first in pair, reverse strand
            c1.reference_id = 1  # index of the chromosome in the header
            c1.reference_start = 91271850  # Add 70 to get start.
            c1.mapping_quality = 255
            # 13 soft-clipped (Barcode, UMI, etc) + 70 matched
            c1.cigartuples = ((0, 70), (4, 13))
            c1.next_reference_id = 1
            c1.next_reference_start = 91271706
            c1.template_length = -167
            c1.query_qualities = pysam.qualitystring_to_array(
                "C;CCCCCCC-CCCCCC;CCCCCCCC;CCCC;CCC--CCCCCC;CCCCC-C;CCCCC-C-CCCCCCCCCCCCCCCCC;CCCCCC"
            )
            c1.tags = (("CB", "ATAGACCCACACAGAG-1"), ("UB", "TATTCATAAT"))
            # Duplicate of first alignment - read 2
            c2 = pysam.AlignedSegment()
            c2.query_name = "VH00211:4:AAAHCYCM5:1:2105:54019:11375"
            c2.query_sequence = "TAAGACAGACTTTGCAGGTTATATTTATGTATTTAGATACACTGTTGGGAGCCATTAAGGCAACGCTATTGTCCTGATCTCTGAATTGGGCCTCTCCCCCCAAGAAGAA"
            c2.flag = 163  # proper pair, second in pair, mate in reverse strand
            c2.reference_id = 1  # index of the chromosome in the header
            c2.reference_start = 91271706
            c2.mapping_quality = 255
            c2.cigar = ((0, 109),)
            c2.next_reference_id = 1
            c2.next_reference_start = 91271850
            c2.template_length = 167
            c2.query_qualities = pysam.qualitystring_to_array(
                "CCCCCCCCCCCCCC;CCC-CCC-CCCCCC-CC-CCCCC;CCC-;C-CCCC;CCCCCC;-CCCC--CC;;CC;CC-CCCCC;CCCCCCCC;C-C;CC;CCC;CCC;CC;C"
            )
            c2.tags = (("CB", "ATAGACCCACACAGAG-1 "), ("UB", "TATTCATAAT"))

            # Almost duplicate of first alignment - read 1
            d1 = pysam.AlignedSegment()
            d1.query_name = "VH00211:4:AAAHCYCM5:1:2105:54019:11375"
            d1.query_sequence = "CCCCAAGAAGAAAAGAGGGTCAAAAGCGGGCCACCGACACACCGCTCTGAGAACAACCACAGATTGTTCCCCCATATAAGAAA"
            d1.flag = 83  # proper pair, first in pair, reverse strand
            d1.reference_id = 1  # index of the chromosome in the header
            d1.reference_start = 91271850  # Add 70 to get start.
            d1.mapping_quality = 255
            # 13 soft-clipped (Barcode, UMI, etc) + 70 matched
            d1.cigartuples = ((0, 70), (4, 13))
            d1.next_reference_id = 1
            d1.next_reference_start = 91271706
            d1.template_length = -167
            d1.query_qualities = pysam.qualitystring_to_array(
                "C;CCCCCCC-CCCCCC;CCCCCCCC;CCCC;CCC--CCCCCC;CCCCC-C;CCCCC-C-CCCCCCCCCCCCCCCCC;CCCCCC"
            )
            d1.tags = (
                ("CB", "ATAGACCCACACAGAG-1"),
                ("UB", "TATTCATAAC"),
            )  # Changed last T to C in the UMI
            # Duplicate of first alignment - read 2
            d2 = pysam.AlignedSegment()
            d2.query_name = "VH00211:4:AAAHCYCM5:1:2105:54019:11375"
            d2.query_sequence = "TAAGACAGACTTTGCAGGTTATATTTATGTATTTAGATACACTGTTGGGAGCCATTAAGGCAACGCTATTGTCCTGATCTCTGAATTGGGCCTCTCCCCCCAAGAAGAA"
            d2.flag = 163  # proper pair, second in pair, mate in reverse strand
            d2.reference_id = 1  # index of the chromosome in the header
            d2.reference_start = 91271706
            d2.mapping_quality = 255
            d2.cigar = ((0, 109),)
            d2.next_reference_id = 1
            d2.next_reference_start = 91271850
            d2.template_length = 167
            d2.query_qualities = pysam.qualitystring_to_array(
                "CCCCCCCCCCCCCC;CCC-CCC-CCCCCC-CC-CCCCC;CCC-;C-CCCC;CCCCCC;-CCCC--CC;;CC;CC-CCCCC;CCCCCCCC;C-C;CC;CCC;CCC;CC;C"
            )
            d2.tags = (
                ("CB", "ATAGACCCACACAGAG-1 "),
                ("UB", "TATTCATAAC"),
            )  # Changed last T to C in the UMI

            # Almost duplicate of second alignment - read 1
            e1 = pysam.AlignedSegment()
            e1.query_name = "VH00211:4:AAAHCYCM5:1:2605:56236:5601"
            e1.query_sequence = "TTTCTTATATGGGAAACCAAACTGGTAGAAGATATTAAAGAGGAAACACAAAAATCCCTTAAAGAATTGCAGGAAAACACAAC"
            e1.flag = 99  # proper pair, first in pair, mate in reverse strand
            e1.reference_id = 0
            e1.reference_start = 172196719
            e1.mapping_quality = 255
            # 14 soft-clipped (Barcode, UMI, etc) + 69 matched
            e1.cigar = ((4, 14), (0, 69))
            e1.next_reference_id = 0
            e1.next_reference_start = 172196993
            e1.template_length = 361
            e1.query_qualities = pysam.qualitystring_to_array(
                ";C-CCCCCCC;CCCC;CCCCCCCCCCCCCCC;CCCCC;CCCCCCCCCCCCC;;CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"
            )
            e1.tags = (
                ("CB", "CCTCAGTGTTAAGATG-1"),
                ("UB", "AAGTTAGGGC"),
            )  # Changed a G to a C in the UMI
            # Second alignment - read 2
            e2 = pysam.AlignedSegment()
            e2.query_name = "VH00211:4:AAAHCYCM5:1:2605:56236:5601"
            e2.query_sequence = "TTCCATAGAGAACATCGGCACAACAATCAAAGAAAATGGAAAATGCAAAAAGATCCTAACTCAAAACATCCAGGAAATCCAGGACACAATGAGAAGACCGAACCTACGG"
            e2.flag = 147  # proper pair, second in pair, reverse strand
            e2.reference_id = 0
            e2.reference_start = 172196993
            e2.mapping_quality = 255
            e2.cigar = ((0, 109),)
            e2.next_reference_id = 0
            e2.next_reference_start = 172196719
            e2.template_length = -361
            e2.query_qualities = pysam.qualitystring_to_array(
                "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC;CCCC-CCCCCCCCCCCCCCCCCCCC"
            )
            e2.tags = (
                ("CB", "CCTCAGTGTTAAGATG-1"),
                ("UB", "AAGTTAGGGC"),
            )  # Changed a G to a C in the UMI

            # Write all read alignments to a bam file
            # alignment order must be position sorted
            for align in [c2, d2, a2, a1, c1, d1, b1, e1, b2, e2]:
                outf.write(align)

        pysam.index(self.alignment)
        self.index = f"{self.alignment}.bai"

    __enter__ = enter

    def __exit__(self, exc_type, exc_value, traceback):
        os.remove(self.alignment)
        os.remove(self.index)
