import HTSeq



def load_alignments_read1(bam):
    bam_file = HTSeq.BAM_Reader(bam)
    alignments = []
    for alignment in bam_file:
        if not alignment.pe_which == "first":
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