import gzip

from gtfparse import read_gtf

from .module_log import module_logger


def extract_feature(gtf_file, feature):
    """
    Extracts a feature from a GTF annotation file.

    Args:
        gtf_file (typing.BinaryIO): Gene annotation file in GTF format.
    """
    logger = module_logger()
    logger.info('Parsing GTF reference file to retrieve only the UTR annotations')

    gtf_df = open_gtf(gtf_file)
    utr_df = gtf_df[gtf_df["feature"] == feature]
    
    logger.info('Finished parsing GTF file')
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
    annotations_index = list(range(len(main)+1, len(annotations_columns)+1))
    return annotations_columns, annotations_index

def reduce_regions(bed, annotations_index=[], operation_list=[]):
    logger = module_logger()
    if not operation_list:
        operations = ['distinct',] * len(annotations_index) + ['count']
    else:
        operations = operation_list.append('count')
    annotations_index.append(1)

    logger.info('Merging overlapping UTR regions')
    bed_merged = bed.merge(c=annotations_index, o=operations, s=True)
    return bed_merged

def get_sequences(genome, bed):
    bed.sequence(fi=genome)
