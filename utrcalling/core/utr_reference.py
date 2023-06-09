import gzip

import pandas as pd
from pandas.api.types import is_numeric_dtype
from gtfparse import read_gtf

from .module_log import module_logger


def extract_feature(gtf_df, feature):
    """
    Extracts a feature from a GTF annotation file.

    Args:
        gtf_df (pd.DataFrame): Gene annotation DataFrame in Pandas format.
        feature (str): Feature to filter. e.g. three_prime_utr, five_prime_utr.
    """
    logger = module_logger()
    logger.info(
        'Filtering GTF reference file to retrieve only the UTR annotations'
    )

    utr_df = gtf_df[gtf_df["feature"] == feature]
    return (utr_df)


def open_gtf(gtf_file):
    """
    Parses a GTF annotation file to a pandas dataframe.

    Args:
        gtf_file (typing.BinaryIO): Gene annotation file in GTF format.
    """
    logger = module_logger()
    logger.info('Parsing GTF reference file, may take a minute')

    if gtf_file.endswith(".gz"):
        with gzip.open(gtf_file) as f:
            gtf_df = read_gtf(f)
    else:
        gtf_df = read_gtf(gtf_file)

    logger.info('Finished parsing GTF file')
    return gtf_df


def prepare_annotations_index(annotations_extra):
    main = ['seqname', 'start', 'end']
    annotations_columns = main + annotations_extra
    annotations_index = list(
        range(len(main) + 1, len(annotations_columns) + 1))
    return annotations_columns, annotations_index


def reduce_regions(bed, annotations_index=[], operation_list=[]):
    logger = module_logger()
    if not operation_list:
        operations = ['distinct', ] * len(annotations_index)
    else:
        operations = operation_list

    logger.info('Merging overlapping UTR regions')
    bed_merged = bed.merge(c=annotations_index, o=operations, s=True)
    return bed_merged


def turn_utr_bed_to_dataframe(utr_table):
    if not isinstance(utr_table, pd.DataFrame):
        utr_table = utr_table.to_dataframe()
    if not is_numeric_dtype(utr_table["start"]):
        utr_table["start"] = utr_table["start"].astype(int)
    if not is_numeric_dtype(utr_table["end"]):
        utr_table["end"] = utr_table["end"].astype(int)
    return utr_table


def generate_utr_name(utr_df):
    # sort by name so that name incrementation is handled correctly later on.
    utr_df.sort_values(by=["chrom", "start", "end"], inplace=True)

    names_list = []
    counter = 1
    name_previous = ''
    for row in utr_df.itertuples():
        name_next = row.name
        if name_next == name_previous:
            counter += 1
            name = f'{name_next}_{counter}'
        else:
            counter = 1
            name = name_previous = name_next
        names_list.append(name)
    utr_df['utr_name'] = names_list
    utr_df = utr_df.set_index('utr_name')
    utr_df.sort_values(by=["chrom", "start", "end"], inplace=True)
    return utr_df


def correct_column_names(utr_df):
    utr_df = utr_df.rename(columns={'thickStart': 'transcript_id',
                                    'thickEnd': 'gene_name'})
    return utr_df


def get_sequences(genome, bed):
    bed.sequence(fi=genome)
