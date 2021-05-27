"""
Goal: Calling single cell 3'UTR peaks
Steps:
1- Extract the 3'UTR regions of transcripts (from ENSEMBL)
1.2 - Resolve overlapping transcripts so that every entry stores unique, non-overlapping, 
genomic coordinates
2-

Obs:
- Final object should be a gene ordered list in a format like GRanges object- use pyranges 
package. Should be able to reconstruct count matrix, or instead just offer count matrix 
straight away.
- Output should inherit single cell annotations, as well as gene annotation like the 
genomic coordinates (GRanges obj) and 3'UTR sequence information.
"""

import pybedtools

from .module_log import module_logger
from .utr_reference import extract_feature, prepare_annotations_index, reduce_regions,\
     get_sequences
from .utr_mapping import generate_utr_intervals, map_reads_to_utr

# Set up logging
logger = module_logger()

def main(annotation, genome, feature, alignment):
    utr_reference_regions = prepare_utr_reference_regions(annotation, genome, feature)
    utr_positions_counts = map_aligned_reads(alignment, utr_reference_regions)
    return utr_positions_counts

def prepare_utr_reference_regions(gtf, genome, feature):
    # extract 3'UTRs from all the features in the GTF file
    utr = extract_feature(gtf, feature)
    # sort by chr and start for bed merge
    utr = utr.sort_values(by=['seqname', 'start'])
    annotations = ['gene_id', 'score', 'strand', 'transcript_id']
    ann_cols, ann_idx = prepare_annotations_index(annotations)
    # Create BedTool obj using the GTF dataframe
    utr_bed = pybedtools.BedTool.from_dataframe(utr, columns=ann_cols)
    # find and merge overlapping regions
    utr_reduced = reduce_regions(utr_bed, annotations_index=ann_idx)
    # get the sequences (from the genome file) for each UTR region
    sequences = get_sequences(genome, utr_reduced)
    return utr_reduced

def map_aligned_reads(alignment, utr_regions):
    intervals = generate_utr_intervals(utr_regions)
    read_mappings = map_reads_to_utr(alignment, intervals, read='first')
    return read_mappings
    
    return

if __name__ == "__main__":
    main("/mnt/o_drive/andre/projects/3utr/youngNSC/Mus_musculus.GRCm39.104.gtf.gz")