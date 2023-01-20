import os
from tempfile import mkstemp
import pysam


class SmartSeqCapSeqBam:
    def __init__(self) -> None:
        fd, self.alignment = mkstemp(suffix=".bam", dir="./test_data/")

    def enter(self):
        # set-up the temporary alignment file
        header = {'HD': {'VN': '1.0'},
                  'SQ': [{'LN': 1575, 'SN': '1'},
                         {'LN': 1584, 'SN': '2'}]}

        with pysam.AlignmentFile(self.alignment, "wb", header=header) as outf:
            # First alignment read 1
            a1 = pysam.AlignedSegment()
            a1.query_name = "VH00211:1:AAAFH3LHV:1:1403:73731:41910"
            a1.query_sequence = (
                "TGAGAGATAACCTTTGATCACTCCACAGGGCCTGGAAATGAGGGTTTTCAGGAAGAAAGGAATTCCAAACAATGTTTTCAAACACGCAGACATTGCGCAAT"
            )
            a1.flag = 83  # proper pair, first in pair, reverse strand
            a1.reference_id = 0
            a1.reference_start = 3340581  # Add 82 to get start. Inside the UTR by 9/10bp
            a1.mapping_quality = 255
            a1.cigar = ((0, 82), (4, 19))
            a1.next_reference_id = 0
            a1.next_reference_start = 3340293  # inside the UTR
            a1.template_length = -370
            a1.query_qualities = pysam.qualitystring_to_array(
                "CCC;CCCCC;CCCCCC-;CCC-CCCCCCCC;CCCCCCCCCC-;CCCCCCCCCCCCCCCCCC-CCCCCCCCC-CCCCCCCCCCCCCCCCCCCC-CCCCC;C-"
            )
            a1.tags = (("NM", 1),
                       ("NM", 0))
            # First alignment read 2
            a2 = pysam.AlignedSegment()
            a2.query_name = "VH00211:1:AAAFH3LHV:1:1403:73731:41910"
            a2.query_sequence = (
                "GTGTACTCTGTTTCACCATTTCTCTGTCAACTTCTGCCAACCCCTGATTTTTTTTTTATTTCCTCTACAGTTTTACCATTTTTCAGGATATTCTCTGTTAC"
            )
            a2.flag = 163  # proper pair, second in pair, mate in reverse strand
            a2.reference_id = 0
            a2.reference_start = 3340293  # inside the UTR by 9/10bp
            a2.mapping_quality = 255
            a2.cigar = ((0, 101),)
            a2.next_reference_id = 0
            a2.next_reference_start = 3340581  # inside the UTR
            a2.template_length = 370
            a2.query_qualities = pysam.qualitystring_to_array(
                "CCCC;-CCCCCCCCCCCCC-CC;CCCCC;CC;;C-CCCC-CCCCCCCCCCCC;C-CC;CCCC-C-CC---CCCC;CC-CC-C----C-C-CC;CCCCCC;C"
            )
            a2.tags = (("NM", 1),
                       ("NM", 1))

            # Second alignment read 1
            b1 = pysam.AlignedSegment()
            b1.query_name = "VH00211:1:AAAFH3LHV:2:1202:74811:33600"
            b1.query_sequence = (
                "ATTGCGCAATGTTGGCAGGCTGAATGCTGAGCATTCCCTTGGAGAGGACACAATGCCCTGAAAATGGATAGCAGTTCCCTTAATCTTCTTGTACTTATGGG"
            )
            b1.flag = 99  # proper pair, first in pair, mate in reverse strand
            b1.reference_id = 0
            b1.reference_start = 3536771  # inside the UTR by 1bp
            b1.mapping_quality = 255
            b1.cigar = ((4, 19), (0, 82))  # 11bp TSO + 8bp UMI + 5'UTR
            b1.next_reference_id = 0
            b1.next_reference_start = 3536850  # outside the UTR
            b1.template_length = 46897
            b1.query_qualities = pysam.qualitystring_to_array(
                "CCCCCCCC;CCCCCCCCCCC;CCCCCCCCCCCCCCCCCCCCCCC-CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC-CCC-;CC;CCC-CCCCCC;C"
            )
            b1.tags = (("NM", 0),
                       ("NM", 0))
            # Second alignment read 2
            b2 = pysam.AlignedSegment()
            b2.query_name = "VH00211:1:AAAFH3LHV:2:1202:74811:33600"
            b2.query_sequence = (
                "GGGAGGAAAGAGAAGAGGTGGAGATATGCCCGTGTGCAGTGAGATCTGTTGGATTCTAACGGCTTTGCCTGGATGGCATTTGAAATCTCCAACTCAGACAC"
            )
            b2.flag = 147  # proper pair, second in pair, reverse strand
            b2.reference_id = 0
            b2.reference_start = 3536850  # outside the 5'UTR
            b2.mapping_quality = 255
            b2.cigar = ((0, 61), (3, 19), (0, 40))  # from the gene body
            b2.next_reference_id = 0
            b2.next_reference_start = 3536771  # outside the UTR
            b2.template_length = -46897
            b2.query_qualities = pysam.qualitystring_to_array(
                "CCC;CCCCCCCCCCCCCCCCCCC;-CCCCCCCCCCC;CCCC;CCCCCCCCCCCC-CC;CCCCC-CCCCCCCCCCCCC;C-CCCCC;CCCCCCCCCCCCCCC"
            )
            b2.tags = (("NM", 0),
                       ("NM", 0))

            # Write all read alignments to a bam file
            for align in [a2, a1, b1, b2]:  # alignment order must be position sorted
                outf.write(align)

        pysam.index(self.alignment)
        self.index = f"{self.alignment}.bai"

    __enter__ = enter

    def __exit__(self, exc_type, exc_value, traceback):
        os.remove(self.alignment)
        os.remove(self.index)
