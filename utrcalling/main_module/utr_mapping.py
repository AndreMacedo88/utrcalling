# system tools
from functools import partial
import psutil
from pympler import asizeof
# data handling tools
import numpy as np
import HTSeq
import pandas as pd
# QOL tools
from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)
# internal tools
from utrcalling.main_module.module_log import module_logger
from utrcalling.main_module.storage import UtrMappings, MappedReadInformation
from utrcalling.main_module.helpers import to_string


def generate_utr_intervals(utr_regions):
    logger = module_logger()
    intervals = dict()

    logger.info('Generating genomic interval for UTR number')
    utrs = utr_regions.itertuples()
    for utr in tqdm(utrs, total=len(utr_regions), colour="#ffb700", ascii=" ■", desc="UTR: "):
        try:
            intervals[utr.Index] = HTSeq.GenomicInterval(
                utr.chrom, utr.start - 1, utr.end, utr.strand)
        except TypeError:
            intervals[utr.Index] = HTSeq.GenomicInterval(
                str(utr.chrom), utr.start - 1, utr.end, utr.strand)
    return intervals


def map_reads_to_utr(bam, intervals, protocol, read='first', sample=None):
    logger = module_logger()
    logger.info(
        f'Extracting reads from {bam} and mapping them to the reference UTR')

    bam_file = HTSeq.BAM_Reader(bam)
    utr_mappings = dict()

    for name, interval in tqdm(intervals.items(), colour="#ffb700", ascii=" ■",
                               desc="Interval: "):
        read_start_positions = []
        read_strands = []
        read_barcodes = []
        read_umis = []
        read_lengths = []
        utr_sizes_from_reads = []

        mapped_read_information = [read_start_positions, read_strands,
                                  read_barcodes, read_umis,
                                  read_lengths, utr_sizes_from_reads]

        for alignment in bam_file[interval]:
            if not alignment.pe_which == read:  # get only one of the 2 paired reads
                continue
            if not alignment.proper_pair:
                continue
            if alignment.pcr_or_optical_duplicate:
                continue
            if alignment.not_primary_alignment:
                continue
            if not protocol.check_strand_matches(alignment.iv.strand, interval.strand):
                continue
            else:
                read_info = extract_read_information(alignment=alignment,
                                                     interval=interval,
                                                     protocol=protocol,
                                                     sample=sample)
                if read_info is None:
                    continue
                else:
                    read_start_positions.append(read_info[0])
                    read_strands.append(read_info[1])
                    read_barcodes.append(read_info[2])
                    read_umis.append(read_info[3])
                    read_lengths.append(read_info[4])
                    utr_sizes_from_reads.append(read_info[5])

                    # Read info: ["read_start_positions", "read_strands",
                    #             "read_barcodes", "read_umis",
                    #             "read_lengths", "utr_sizes_from_reads"]

        if read_start_positions:
            umis_unique = []
            idx_unique = []
            for idx, umi in enumerate(read_umis.copy()):
                if umi not in umis_unique:
                    idx_unique.append(idx)
                    umis_unique.append(umi)
            mapped_read_information = [[element[i] for i in idx_unique] for element in mapped_read_information]

            utr_mappings[name] = mapped_read_information

    utr_mappings = pd.DataFrame.from_dict(
        utr_mappings, orient="index",
        columns=["read_start_positions", "read_strands", "read_barcodes", "read_umis",
                 "read_lengths", "utr_sizes_from_reads"])

    utr_mappings.index.name = "utr_name"

    logger.debug(f'Size of mapped reads object in bytes: {asizeof.asizeof(utr_mappings)}\
        \n\nMemory usage:{psutil.virtual_memory().percent}')

    return utr_mappings


def map_reads_to_utr_old(bam, intervals, protocol, read='first', sample=None):
    logger = module_logger()
    logger.info(
        f'Extracting reads from {bam} and mapping them to the reference UTR')

    bam_file = HTSeq.BAM_Reader(bam)
    utr_mappings = UtrMappings()

    for name, interval in tqdm(intervals.items(), colour="#ffb700", ascii=" ■",
                               desc="Interval: "):

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
            if not protocol.check_strand_matches(alignment.iv.strand, interval.strand):
                continue
            else:
                read_info = extract_read_information(alignment=alignment,
                                                     interval=interval,
                                                     protocol=protocol,
                                                     sample=sample)
                if read_info is None:
                    continue
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
            mapped_read_information_unique_umis = mapped_read_information.get_unique_umis()
            utr_mappings[name] = mapped_read_information_unique_umis

    logger.debug(f'Size of mapped reads object in bytes: {asizeof.asizeof(utr_mappings)}\
        \n\nMemory usage:{psutil.virtual_memory().percent}')

    return utr_mappings


def extract_read_information(alignment, interval, protocol, sample=None):
    """
    Extracts all the relevant information from matching reads in a HTSeq.Alignment object.

    Args:
        alignment (HTSeq.Alignment): The read alignment object.
        interval (HTSeq.GenomicInterval): The UTR interval coordinates.
        protocol (utrcalling.Protocol): Protocol for the UTR calling run, based on
            the RNA-seq platform used.

    Returns:
        tuple: The relevant read information
    """
    logger = module_logger()

    # get read strand
    read_strand = alignment.iv.strand

    # get the read starting position. This may, or may not, be the match start position
    read_start = alignment.iv.start_d

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
        Please filter your alignments before running the tool.'
        logger.error(match_fail)
        raise RuntimeError(match_fail)
    if read_strand == '+':
        # get UTR start position from the first matching base pair from the read
        read_start_position = read_start + query_from
        read_sequence = to_string(alignment.read.seq[query_from:query_to])
    else:
        # get UTR start position from the first matching base pair from the read
        read_start_position = read_start - query_from
        if query_from == 0:
            read_sequence = to_string(alignment.read.seq[-query_to:])
        else:
            read_sequence = to_string(
                alignment.read.seq[-query_to:-query_from])
    # if the start of the read is not in the reference UTR region then we discard the read
    if interval.strand == '+':
        if not (read_start_position >= interval.start_d and read_start_position <= interval.end_d):
            return None
    else:
        if not (read_start_position <= interval.start_d and read_start_position >= interval.end_d):
            return None

    # get the read's cell barcode and UMI
    read_umi, read_barcode = protocol.get_reads_identity(alignment=alignment,
                                                         sample=sample)
    if not read_umi:  # if the UMI was not found, we discard the read
        return None
    if not read_barcode:
        return None

    # get read length
    read_length = len(read_sequence)

    # get the computed size of the UTR for this molecule
    utr_size = abs(read_start_position - interval.start_d)

    information = read_start_position, read_strand, read_barcode, \
        read_umi, read_length, utr_size  # , read_sequence

    return information
