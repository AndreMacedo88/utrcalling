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
import numpy as np
import pandas as pd
from tqdm import tqdm
from gtfparse import read_gtf

def extract_3UTR(gtf_file):
    """
    Extracts the 3'UTR regions from a GTF annotation file

    Args:
        gtf_file (typing.BinaryIO): Gene annotation file in GTF format.
    """    
    gtf_df = open_gtf(gtf_file)
    utr_df = gtf_df[gtf_df["feature"] == "three_prime_utr"]
    return(utr_df)

def open_gtf(gtf_file):
    if gtf_file.endswith(".gz"):
        with gzip.open(gtf_file) as f:
            gtf_df = read_gtf(f)
    else:
        with gtf_file as f:
            gtf_df = read_gtf(f)
    return gtf_df


def reduce_regions(df):
    df_ = df.copy()
    df_ = df_.sort_values(by=['seqname', 'start'])
    chr_list = df_['seqname'].unique()
    count = 0
    for chr in tqdm(chr_list):
        df_chr = df_[df_['seqname'] == chr]
        t = df_chr.itertuples()
        y = df_chr.itertuples()
        next(y)
        for row1, row2 in zip(t, y):
            if row2.start >= row1.end:
                continue
            elif row2.start < row1.end and row1.gene_id == row2.gene_id:
                df_.at[row1.Index, "end"] = row2.end
                df_ = df_.drop(index=row2.Index)
            else:
                count += 1
    print(f"Number of overlapping UTRs with different gene IDs: {count}")
    return df_

def reduce_regions_faster(df):
    df_ = df.copy()
    df_ = df_.sort_values(by=['seqname', 'start'])
    chr_list = df_['seqname'].unique()
    count = 0
    for chr in tqdm(chr_list):
        df_chr = df_[df_['seqname'] == chr]
        t = df_chr.iterrows()
        y = df_chr.iterrows()
        next(y)
        rows = []
        for (i, row1), (j, row2) in zip(t, y):
            if row2.start >= row1.end:
                continue
            elif row2.start < row1.end and row1.gene_id == row2.gene_id:
                row1.end = row2.end
                rows.append(row1)
                df_ = df_.drop(index=j)
            else:
                count += 1
    print(f"Number of overlapping UTRs with different gene IDs: {count}")
    return df_