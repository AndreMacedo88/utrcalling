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
    Generates synthetic BAM alignment data in conformity with what a 10x would look
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
            a1.query_name = "VH00211:1:AAAGTM7HV:1:2513:14009:16960"
            a1.query_sequence = "ACTGTAGCCGAATTTAACTGGCATTTTATTTGACCTCTAACTATGTCCCTCCGGAACTGTACAGAAAAAAAAAAAAAAAAAAACCAATGTCGAAGAATTGATGACTTGTCAAGAA"
            a1.flag = 83  # proper pair, first in pair, reverse strand
            a1.reference_id = 1  # index of the chromosome in the header
            a1.reference_start = 91271840  # Add 80 to get start.
            a1.mapping_quality = 255
            # 35 soft-clipped (Barcode, UMI, etc) + 80 matched
            a1.cigartuples = ((0, 80), (4, 35))
            a1.next_reference_id = 0
            a1.next_reference_start = 91271713
            a1.template_length = -207
            a1.query_qualities = pysam.qualitystring_to_array(
                "-C--C---CC--CC--C;;--C;-------C--CC-;---CC-C-C-;C;-C--CCC--;--;C-C;;-;--C--C-C---C;----;CCCCC;CC;CCCCCCCCCCCCCCCCCC"
            )
            a1.tags = (("NM", 1), ("NM", 0))
            # First alignment - read 2
            a2 = pysam.AlignedSegment()
            a2.query_name = "VH00211:1:AAAGTM7HV:1:2513:14009:16960"
            a2.query_sequence = "TGGACGCGTGCGCGCGTGGGTGCCGCGTCTTGGCTTGTGACTAGCGTTAGGAAATTACCCAAGATTATTGCATAATCTTCACCAGCTTGCTTTGATTCTTTTTTATGTTGGAAGT"
            a2.flag = 163  # proper pair, second in pair, mate in reverse strand
            a2.reference_id = 1  # index of the chromosome in the header
            a2.reference_start = 91271713
            a2.mapping_quality = 255
            a2.cigar = ((0, 115),)
            a2.next_reference_id = 0
            a2.next_reference_start = 91271840
            a2.template_length = 207
            a2.query_qualities = pysam.qualitystring_to_array(
                "C-CCCCC--CC-CCCCCC;CCCCCC-CCCCCCCC;CCC-CCCC-CCC-C;CCCCCCCC-CCCCCCCCCC;CCCCCCCCCC-C-C-C-CC-CCCC-CCCC-CCCCCCCCCCC-CCC"
            )
            a2.tags = (("NM", 1), ("NM", 1))

            # Second alignment - read 1
            b1 = pysam.AlignedSegment()
            b1.query_name = "VH00211:1:AAAGTM7HV:1:1308:23628:30438"
            b1.query_sequence = "GACTCTCCACAATGAACGTTCTCGCAAAATAATACGTTCTCTAAAAACATATGTTTACGGTAGAGTCAGTCACCTTTATTCCAGGGTTAGAACAAGGCTGTGCACGCTGCAGACA"
            b1.flag = 99  # proper pair, first in pair, mate in reverse strand
            b1.reference_id = 0
            b1.reference_start = 172196728
            b1.mapping_quality = 255
            # 55 soft-clipped (Barcode, UMI, etc) + 60 matched
            b1.cigar = ((4, 55), (0, 60))
            b1.next_reference_id = 0
            b1.next_reference_start = 172196993
            b1.template_length = 380
            b1.query_qualities = pysam.qualitystring_to_array(
                "CCCCCCCCCCCCCCC;CCCCCCCC;CCC;C---C----CC--C--;--;--C---;----;--CC------C--;C-;-;--;;-C---;-C---CC--C-C---C--CC-C;-;"
            )
            b1.tags = (("NM", 0), ("NM", 0))
            # Second alignment - read 2
            b2 = pysam.AlignedSegment()
            b2.query_name = "VH00211:1:AAAGTM7HV:1:1308:23628:30438"
            b2.query_sequence = "TCTCCCACCCAGTTGTACAAGGATTTCTTGTCTTAAAGGAGATGCCAGGCAAGACTAGGGAGATTTAGGAAGGACCTTTAAGGCAAAGCTGGAGTCTTCGAGGTGTGAGAGTTAA"
            b2.flag = 147  # proper pair, second in pair, reverse strand
            b2.reference_id = 0
            b2.reference_start = 172196993
            b2.mapping_quality = 255
            b2.cigar = ((0, 115),)
            b2.next_reference_id = 0
            b2.next_reference_start = 172196728
            b2.template_length = -380
            b2.query_qualities = pysam.qualitystring_to_array(
                "-CCC;CCC;C-;-CCCCC--CCCCC-CCC;CCCCCCCCCCCCCCCCCC;CCCCCCCCCCCCCCCCCCCC;CCC-CCCCCCCCCCCCCCCCCCC;-C;CCCCCCCC;CCCCCCCCC"
            )
            b2.tags = (("NM", 0), ("NM", 0))

            # Duplicate of first alignment - read 1
            c1 = pysam.AlignedSegment()
            c1.query_name = "VH00211:1:AAAGTM7HV:1:2513:14009:16961"
            c1.query_sequence = "AGCCGAATTTAACTGGCATTTTATTTGACCTCTAACTATGTCCCTCCGGAACTGTACAGAAAAAAAAAAAAAAAAAAACCAATGTCGAAGAATTGATGACTTGTCAAGAA"
            c1.flag = 83  # proper pair, first in pair, reverse strand
            c1.reference_id = 1  # index of the chromosome in the header
            c1.reference_start = 91271845  # Add 75 to get start.
            c1.mapping_quality = 255
            # 35 soft-clipped (Barcode, UMI, etc) + 75 matched
            c1.cigartuples = ((0, 75), (4, 35))
            c1.next_reference_id = 0
            c1.next_reference_start = 91271713
            c1.template_length = -207
            c1.query_qualities = pysam.qualitystring_to_array(
                "---CC--CC--C;;--C;-------C--CC-;---CC-C-C-;C;-C--CCC--;--;C-C;;-;--C--C-C---C;----;CCCCC;CC;CCCCCCCCCCCCCCCCCC"
            )
            c1.tags = (("NM", 1), ("NM", 0))
            # Duplicate of first alignment - read 2
            c2 = pysam.AlignedSegment()
            c2.query_name = "VH00211:1:AAAGTM7HV:1:2513:14009:16961"
            c2.query_sequence = "TGGACGCGTGCGCGCGTGGGTGCCGCGTCTTGGCTTGTGACTAGCGTTAGGAAATTACCCAAGATTATTGCATAATCTTCACCAGCTTGCTTTGATTCTTTTTTATGTTGGAAGT"
            c2.flag = 163  # proper pair, second in pair, mate in reverse strand
            c2.reference_id = 1  # index of the chromosome in the header
            c2.reference_start = 91271713
            c2.mapping_quality = 255
            c2.cigar = ((0, 115),)
            c2.next_reference_id = 0
            c2.next_reference_start = 91271840
            c2.template_length = 207
            c2.query_qualities = pysam.qualitystring_to_array(
                "C-CCCCC--CC-CCCCCC;CCCCCC-CCCCCCCC;CCC-CCCC-CCC-C;CCCCCCCC-CCCCCCCCCC;CCCCCCCCCC-C-C-C-CC-CCCC-CCCC-CCCCCCCCCCC-CCC"
            )
            c2.tags = (("NM", 1), ("NM", 1))

            # Almost duplicate of first alignment - read 1
            d1 = pysam.AlignedSegment()
            d1.query_name = "VH00211:1:AAAGTM7HV:1:2513:14009:16962"
            d1.query_sequence = "AGCCGAATTTAACTGGCATTTTATTTGACCTCTAACTATGTCCCTCCGGAACTGTACAGAAAAAAAAAAAAAAAAAAACCAATGTCGAAGAATcGATGACTTGTCAAGAA"  # changed a T to C at position 17 (first position of the UMI) from the end
            d1.flag = 83  # proper pair, first in pair, reverse strand
            d1.reference_id = 1  # index of the chromosome in the header
            d1.reference_start = 91271845  # Add 75 to get start.
            d1.mapping_quality = 255
            # 35 soft-clipped (Barcode, UMI, etc) + 75 matched
            d1.cigartuples = ((0, 75), (4, 35))
            d1.next_reference_id = 0
            d1.next_reference_start = 91271713
            d1.template_length = -207
            d1.query_qualities = pysam.qualitystring_to_array(
                "---CC--CC--C;;--C;-------C--CC-;---CC-C-C-;C;-C--CCC--;--;C-C;;-;--C--C-C---C;----;CCCCC;CC;CCCCCCCCCCCCCCCCCC"
            )
            d1.tags = (("NM", 1), ("NM", 0))
            # Almost duplicate of first alignment - read 2
            d2 = pysam.AlignedSegment()
            d2.query_name = "VH00211:1:AAAGTM7HV:1:2513:14009:16962"
            d2.query_sequence = "TGGACGCGTGCGCGCGTGGGTGCCGCGTCTTGGCTTGTGACTAGCGTTAGGAAATTACCCAAGATTATTGCATAATCTTCACCAGCTTGCTTTGATTCTTTTTTATGTTGGAAGT"
            d2.flag = 163  # proper pair, second in pair, mate in reverse strand
            d2.reference_id = 1  # index of the chromosome in the header
            d2.reference_start = 91271713
            d2.mapping_quality = 255
            d2.cigar = ((0, 115),)
            d2.next_reference_id = 0
            d2.next_reference_start = 91271840
            d2.template_length = 207
            d2.query_qualities = pysam.qualitystring_to_array(
                "C-CCCCC--CC-CCCCCC;CCCCCC-CCCCCCCC;CCC-CCCC-CCC-C;CCCCCCCC-CCCCCCCCCC;CCCCCCCCCC-C-C-C-CC-CCCC-CCCC-CCCCCCCCCCC-CCC"
            )
            d2.tags = (("NM", 1), ("NM", 1))

            # Almost duplicate of second alignment - read 1
            e1 = pysam.AlignedSegment()
            e1.query_name = "VH00211:1:AAAGTM7HV:1:1308:23628:30439"
            e1.query_sequence = "GACTCTCCACAATGAAtGTTCTCGCAAAATAATACGTTCTCTAAAAACATATGTTTACGGTAGAGTCAGTCACCTTTATTCCAGGGTTAGAACAAGGCTGTGCACGCTGC"  # changed a C to T at position 17 (first position of the UMI)
            e1.flag = 99  # proper pair, first in pair, mate in reverse strand
            e1.reference_id = 0
            e1.reference_start = 172196728
            e1.mapping_quality = 255
            e1.cigar = ((4, 55), (0, 55))  # 16bp Barcode + 12bp UMI + 3'UTR
            e1.next_reference_id = 0
            e1.next_reference_start = 172196993
            e1.template_length = 380
            e1.query_qualities = pysam.qualitystring_to_array(
                "CCCCCCCCCCCCCCC;CCCCCCCC;CCC;C---C----CC--C--;--;--C---;----;--CC------C--;C-;-;--;;-C---;-C---CC--C-C---C--CC"
            )
            e1.tags = (("NM", 0), ("NM", 0))
            # Almost duplicate of second alignment - read 2
            e2 = pysam.AlignedSegment()
            e2.query_name = "VH00211:1:AAAGTM7HV:1:1308:23628:30439"
            e2.query_sequence = "TCTCCCACCCAGTTGTACAAGGATTTCTTGTCTTAAAGGAGATGCCAGGCAAGACTAGGGAGATTTAGGAAGGACCTTTAAGGCAAAGCTGGAGTCTTCGAGGTGTGAGAGTTAA"
            e2.flag = 147  # proper pair, second in pair, reverse strand
            e2.reference_id = 0
            e2.reference_start = 172196993
            e2.mapping_quality = 255
            e2.cigar = ((0, 115),)
            e2.next_reference_id = 0
            e2.next_reference_start = 172196728
            e2.template_length = -380
            e2.query_qualities = pysam.qualitystring_to_array(
                "-CCC;CCC;C-;-CCCCC--CCCCC-CCC;CCCCCCCCCCCCCCCCCC;CCCCCCCCCCCCCCCCCCCC;CCC-CCCCCCCCCCCCCCCCCCC;-C;CCCCCCCC;CCCCCCCCC"
            )
            e2.tags = (("NM", 0), ("NM", 0))

            # Write all read alignments to a bam file
            # alignment order must be position sorted
            for align in [a2, c2, d2, a1, c1, d1, b1, e1, b2, e2]:
                outf.write(align)

        pysam.index(self.alignment)
        self.index = f"{self.alignment}.bai"

    __enter__ = enter

    def __exit__(self, exc_type, exc_value, traceback):
        os.remove(self.alignment)
        os.remove(self.index)
