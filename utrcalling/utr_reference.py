import typing
import gzip

from gtfparse import read_gtf
from numpy import generic

from .module_log import module_logger

# Set up logging
logger = module_logger()

def extract_feature(gtf_file, feature):
    """
    Extracts a feature from a GTF annotation file.

    Args:
        gtf_file (typing.BinaryIO): Gene annotation file in GTF format.
    """    
    gtf_df = open_gtf(gtf_file)
    utr_df = gtf_df[gtf_df["feature"] == feature]
    return(utr_df)

def open_gtf(gtf_file):
    if gtf_file.endswith(".gz"):
        with gzip.open(gtf_file) as f:
            gtf_df = read_gtf(f)
    else:
        gtf_df = read_gtf(gtf_file)
    return gtf_df

def prepare_annotations_index(annotations_extra):
    main = ['seqname', 'start', 'end']
    annotations_columns = main + annotations_extra
    annotations_index = \
    [index+1 for index, value in enumerate(annotations_columns) if value not in main]
    return annotations_columns, annotations_index

def reduce_regions(bed, annotations_index=[], operation_list=[]):
    if not operation_list:
        operations = ['distinct',] * len(annotations_index) + ['count']
    else:
        operations = operation_list.append('count')
    annotations_index.append(1)
    bed_merged = bed.merge(c=annotations_index, o=operations, s=True)
    return bed_merged

def get_sequences(genome, bed):
    bed.sequence(fi=genome)
    pass

# def reduce_regions(df):
#     df_ = df.copy()
#     df_ = df_.sort_values(by=['seqname', 'start'])
#     chr_list = df_['seqname'].unique()
#     count = 0
#     for chr in tqdm(chr_list):
#         df_chr = df_[df_['seqname'] == chr]
#         t = df_chr.itertuples()
#         y = df_chr.itertuples()
#         next(y)
#         for row1, row2 in zip(t, y):
#             if row2.start >= row1.end:
#                 continue
#             elif row2.start < row1.end and row1.gene_id == row2.gene_id:
#                 df_.at[row1.Index, "end"] = row2.end
#                 df_ = df_.drop(index=row2.Index)
#             else:
#                 count += 1
#     print(f"Number of overlapping UTRs with different gene IDs: {count}")
#     return df_

# def reduce_regions_faster(df):
#     df_ = df.copy()
#     df_ = df_.sort_values(by=['seqname', 'start'])
#     chr_list = df_['seqname'].unique()
#     count = 0
#     for chr in tqdm(chr_list):
#         df_chr = df_[df_['seqname'] == chr]
#         t = df_chr.iterrows()
#         y = df_chr.iterrows()
#         next(y)
#         rows = []
#         for (i, row1), (j, row2) in zip(t, y):
#             if row2.start >= row1.end:
#                 continue
#             elif row2.start < row1.end and row1.gene_id == row2.gene_id:
#                 row1.end = row2.end
#                 rows.append(row1)
#                 df_ = df_.drop(index=j)
#             else:
#                 count += 1
#     print(f"Number of overlapping UTRs with different gene IDs: {count}")
#     return df_
