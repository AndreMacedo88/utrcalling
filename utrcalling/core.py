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
import typing
import gzip
from gtfparse import read_gtf

def extract_3UTR(gtf_file):
    """
    Extracts the 3'UTR regions from a GTF annotation file

    Args:
        gtf_file (typing.BinaryIO): Gene annotation file in GTF format.
    """    
    if gtf_file.endswith(".gz"):
        with gzip.open(gtf_file) as f:
            gtf_df = read_gtf(f)
    else:
        with gtf_file as f:
            gtf_df = read_gtf(f)
    utr_df = gtf_df[gtf_df["feature"] == "three_prime_utr"]
    return(utr_df)
