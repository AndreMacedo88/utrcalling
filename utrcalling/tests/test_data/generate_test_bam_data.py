"""Historical script used to generate test BAM data from real alignments.

This script was used once to create the test BAM files shipped with the
repository.  It requires access to the original full-size BAM and GTF files
(hardcoded paths below) and is not run as part of the test suite.
"""

import pybedtools
import pysam
from numpy import random


def generate_genome_intervals(regions):
    intervals = []
    for region in regions:
        intervals.append(
            (region.chrom, region.start, region.stop, region.strand)
        )
    return intervals


def match_alignments(bam_path, output_path, intervals):
    with pysam.AlignmentFile(bam_path, "rb") as bam_in, \
         pysam.AlignmentFile(output_path, "wb", header=bam_in.header) as bam_out:
        for chrom, start, stop, strand in intervals:
            num_1 = random.random()
            if num_1 < 0.33:
                continue
            try:
                for alignment in bam_in.fetch(chrom, start, stop):
                    num_2 = random.random()
                    if num_2 > 0.001:
                        continue
                    bam_out.write(alignment)
            except ValueError:
                continue


gtf = pybedtools.BedTool("test_mus_musculus_GRCm38_102_5prime.gtf.gz")

bam_path = (
    "/mnt/weird_vol/projects/ifnb_nsc/seq.2021-07-06.CAPSeq-bams/"
    "AS-632883-LR-56776/Aligned.out.umi.r1.possorted.bam"
)
output_path = (
    "/mnt/weird_vol/projects/ifnb_nsc/seq.2021-07-06.CAPSeq-bams/"
    "AS-632883-LR-56776/test_CapPhospho_genome_102.bam"
)

intervals = generate_genome_intervals(gtf)
match_alignments(bam_path, output_path, intervals)
