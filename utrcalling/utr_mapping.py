from functools import partial

import psutil
from pympler import asizeof
import HTSeq
from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)

from .module_log import module_logger
from .storage import UtrMappings, MappedReadInformation


def generate_utr_intervals(utr_regions):
    intervals = dict()
    counter = 0
    for utr in tqdm(utr_regions, desc='Generating genomic interval for UTR number'):
        name = utr.name
        if name in intervals.keys():
            counter += 1
            name = f'{name}_{counter}'
        else:
            counter = 0
        intervals[name] = HTSeq.GenomicInterval(utr.chrom, utr.start - 1, utr.stop,
                                                utr.strand)
    return intervals


def map_reads_to_utr(bam, intervals, read='first', version=3):
    logger = module_logger()
    logger.info(
        f'Extracting reads from {bam} and mapping them to the reference UTR')
    bam_file = HTSeq.BAM_Reader(bam)
    utr_mappings = UtrMappings()

    for name, interval in tqdm(intervals.items(),
                               desc='Mapping reads to reference UTR number'):

        mapped_read_information = MappedReadInformation(utr_name=name)
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
                read_info = extract_read_information(alignment=alignment,
                                                     interval=interval, version=version)
                mapped_read_information.read_start_positions.append(
                    read_info[0])
                mapped_read_information.read_strands.append(read_info[1])
                # mapped_read_information.read_sequences.append(read_info[2])
                mapped_read_information.read_barcodes.append(read_info[2])
                mapped_read_information.read_umis.append(read_info[3])
                mapped_read_information.read_lengths.append(read_info[4])
                mapped_read_information.utr_sizes_from_reads.append(
                    read_info[5])

        if mapped_read_information:
            utr_mappings[name] = mapped_read_information

    logger.debug(f'Size of mapped reads object in bytes: {asizeof.asizeof(utr_mappings)}\
        \n\nMemory usage:{psutil.virtual_memory().percent}')

    return utr_mappings


def extract_read_information(alignment, interval, version=3):
    """
    Extracts all the relevant information from matching reads in a HTSeq.Alignment object.

    Args:
        alignment (HTSeq.Alignment): The read alignment object.
        interval (HTSeq.GenomicInterval): The UTR interval coordinates.
        version (int, optional): Version of the 10x protocol used. Defaults to 3.

    Raises:
        RuntimeError: If the read has no matches.
        RuntimeError: If the protocol version is not currently supported.

    Returns:
        tuple: The relevant read information
    """
    logger = module_logger()

    def to_string(x):
        return repr(x)[2:-1]

    # get read strand
    read_strand = alignment.iv.strand

    # get UTR start position (which is the read starting position)
    read_start_position = alignment.iv.start_d

    # get read sequence
    query_from = None
    query_to = None
    for operation in alignment.cigar:
        if operation.type in ('M', 'I', '=', 'X'):
            if query_from is None:
                query_from = operation.query_from
            query_to = operation.query_to
    if query_from is None:
        match_fail = f'Read {alignment.read.seq} has no alignment match! \
        Please filter your alignments before.'
        logger.error(match_fail)
        raise RuntimeError(match_fail)
    if read_strand == '+':
        read_sequence = to_string(alignment.read.seq[query_from:query_to])
    else:
        if query_from == 0:
            read_sequence = to_string(alignment.read.seq[-query_to:])
        else:
            read_sequence = to_string(
                alignment.read.seq[-query_to:-query_from])
    # get the read's cell barcode and UMI
    read_barcode = to_string(alignment.read.seq[:16])
    if version >= 3:
        read_umi = to_string(alignment.read.seq[17:28])
    if version < 3:
        read_umi = to_string(alignment.read.seq[17:26])
    else:
        version_fail = f'10x genomics protocol version {version} currently not supported'
        logger.error(version_fail)
        raise RuntimeError(version_fail)

    # get read length
    read_length = len(read_sequence)

    # get the computed size of the UTR for this molecule
    utr_size = abs(read_start_position - interval.start_d)

    information = read_start_position, read_strand, read_barcode, \
        read_umi, read_length, utr_size  # , read_sequence

    return information
