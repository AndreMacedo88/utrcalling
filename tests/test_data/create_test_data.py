import pybedtools
import HTSeq
from numpy import random

def generate_genome_intervals(regions):
    intervals = []
    for region in regions:
        intervals.append(HTSeq.GenomicInterval(region.chrom, region.start, region.stop, 
        region.strand))
    return intervals

def match_alignments(bam, intervals):
    #set-up BAM_Writer with same header as reader:
    bam_writer = HTSeq.BAM_Writer.from_BAM_Reader(
        "/mnt/weird_vol/projects/ifnb_nsc/seq.2021-07-06.CAPSeq-bams/AS-632883-LR-56776/test_CapPhospho_genome_102.bam", bam)
    for interval in intervals:
        num_1 = random.random()
        if num_1 < 0.33:
            continue
        for alignment in bam_file[interval]:
            num_2 = random.random()
            if num_2 > 0.001:
                continue
            bam_writer.write(alignment)
    bam_writer.close()

gtf = pybedtools.BedTool('test_mus_musculus_GRCm38_102_5prime.gtf.gz')

bam_file = HTSeq.BAM_Reader(
    '//mnt/weird_vol/projects/ifnb_nsc/seq.2021-07-06.CAPSeq-bams/AS-632883-LR-56776/Aligned.out.umi.r1.possorted.bam'
    )

intervals = generate_genome_intervals(gtf)
match_alignments(bam_file, intervals)

