import HTSeq


def load_parse_alignments(bam, read='first'):
    bam_file = HTSeq.BAM_Reader(bam)
    alignments = []
    for alignment in bam_file:
        if not alignment.pe_which == read:  # get only one of the 2 paired reads
            continue
        if not alignment.proper_pair:
            continue
        if alignment.pcr_or_optical_duplicate:
            continue
        if alignment.not_primary_alignment:
            continue
        else:
            alignments.append(alignment)
    return alignments

def generate_utr_intervals(utr_regions):
    intervals = []
    for utr in utr_regions:
        intervals.append(HTSeq.GenomicInterval(utr.chrom, utr.start-1, utr.stop, 
        utr.strand))
    return intervals

def count_reads(bam, intervals, read='first'):
    bam_file = HTSeq.BAM_Reader(bam)
    for interval in intervals:
        for alignment in bam_file[interval]:
            if not alignment.pe_which == read:  # get only one of the 2 paired reads
                continue
            if not alignment.proper_pair:
                continue
            if alignment.pcr_or_optical_duplicate:
                continue
            if alignment.not_primary_alignment:
                continue
            else:
                print(alignment)