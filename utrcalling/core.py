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
import logging

import pybedtools

from .utr_reference import extract_3UTRs, prepare_annotations_index, reduce_regions_bed

# create logger with 'spam_application'
logger = logging.getLogger('Core')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('Test.log')
fh.setLevel(logging.DEBUG)
# create formatter and add it to the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
# add the handler to the logger
logger.addHandler(fh)

logger.info('test core')

def main(file):
    utr = extract_3UTRs(file)  # extract 3'UTRs from all the features in the file
    utr = utr.sort_values(by=['seqname', 'start'])  # sort by chr and start for bed merge
    utr_bed = pybedtools.BedTool.from_dataframe(utr)  # Create BedTool obj with dataframe
    ann_idx = prepare_annotations_index(utr, 'gene_id', 'transcript_id')
    # find and merge overlapping regions
    utr_reduced = reduce_regions_bed(utr_bed, annotations_index=ann_idx)

if __name__ == "__main__":
    main("/mnt/o_drive/andre/projects/3utr/youngNSC/Mus_musculus.GRCm39.104.gtf.gz")