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

from .utr_reference import extract_feature, prepare_annotations_index, reduce_regions,\
     get_sequences
from .module_log import module_logger

# Set up logging
logger = module_logger()

def main(gtf, genome, feature):
    utr_reference = prepare_reference(gtf, genome, feature)
    return utr_reference

def prepare_reference(gtf, genome, feature):
    utr = extract_feature(gtf, feature)  # extract 3'UTRs from all the features in the file
    utr = utr.sort_values(by=['seqname', 'start'])  # sort by chr and start for bed merge
    utr_bed = pybedtools.BedTool.from_dataframe(utr)  # Create BedTool obj with dataframe
    ann_idx = prepare_annotations_index(utr, 'gene_id', 'transcript_id')
    # find and merge overlapping regions
    utr_reduced = reduce_regions(utr_bed, annotations_index=ann_idx)
    sequences = get_sequences(genome, utr_reduced)
    return utr_reduced
    
if __name__ == "__main__":
    main("/mnt/o_drive/andre/projects/3utr/youngNSC/Mus_musculus.GRCm39.104.gtf.gz")