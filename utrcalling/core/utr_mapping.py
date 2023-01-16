# general tools
from functools import partial
import psutil
from pympler import asizeof
import re
# data handling tools
import HTSeq
import pandas as pd
# QOL tools
from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)

from utrcalling.core.module_log import module_logger
from utrcalling.core.helpers import to_string


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


def map_reads_to_utr(bam, intervals, protocol, read='first', batch=None):
    logger = module_logger()
    logger.info(
        f'Extracting reads from {bam} and mapping them to the reference UTR')

    label_batches = protocol.label_batches

    bam_file = HTSeq.BAM_Reader(bam)
    utr_mappings = dict()

    # mapping statistics:

    # TODO: try to implement the following without using all the RAM or wasting 5 minutes:
    # alignments_number = len([al for al in bam_file])
    # "Number of alignments": alignments_number,

    read_strand_string = f"Reads not '{read}'"
    mapping_stats = {read_strand_string: 0,
                     "Alignments not proper pair": 0, "PCR duplicates": 0,
                     "Alignments not primary alignment": 0,
                     "Alignments in interval but wrong strand": 0
                     }
    if label_batches and batch is None:
        for record in bam_file:
            batch_name = re.search(r"([\w|-]*:\w*:[\w|-]*):.*",
                                   record.get_sam_line()).group(1)
            break
        batch = protocol.batch_codes.get(batch_name, None)

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

        for alignment in bam_file[interval]:  # BAM_Reader[GenomicInterval] is a generator
            if not alignment.pe_which == read:  # get only one of the 2 paired reads
                mapping_stats[read_strand_string] += 1
                continue
            if not alignment.proper_pair:
                mapping_stats["Alignments not proper pair"] += 1
                continue
            if alignment.pcr_or_optical_duplicate:
                mapping_stats["PCR duplicates"] += 1
                continue
            if alignment.not_primary_alignment:
                mapping_stats["Alignments not primary alignment"] += 1
                continue
            if not protocol.check_strand_matches(alignment.iv.strand, interval.strand):
                mapping_stats["Alignments in interval but wrong strand"] += 1
                continue
            else:
                read_info = extract_read_information(alignment=alignment,
                                                     interval=interval,
                                                     protocol=protocol,
                                                     batch=batch)
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
            mapped_read_information = [[element[i] for i in idx_unique]
                                       for element in mapped_read_information]

            utr_mappings[name] = mapped_read_information

    del bam_file

    utr_mappings = pd.DataFrame.from_dict(
        utr_mappings, orient="index",
        columns=["read_start_positions", "read_strands", "read_barcodes", "read_umis",
                 "read_lengths", "utr_sizes_from_reads"])

    utr_mappings.index.name = "utr_name"

    logger.info("Mapping statistics: %s", mapping_stats)
    logger.debug("Size of mapped reads object in bytes: %s \nMemory usage: %s",
                 asizeof.asizeof(utr_mappings),
                 psutil.virtual_memory().percent)

    return utr_mappings


def extract_read_information(alignment, interval, protocol, batch=None):
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
    read_start_position = alignment.iv.start_d

    # get read sequence
    read_seq = to_string(alignment.read.seq)
    query_from = None
    query_to = None
    for operation in alignment.cigar:
        if operation.type in ('M', 'I', '=', 'X'):
            if query_from is None:
                query_from = operation.query_from
            query_to = operation.query_to
    if query_from is None:
        match_fail = f'Read {read_seq} has no alignment match! \
        Please filter your alignments before running the tool.'
        logger.error(match_fail)
        raise RuntimeError(match_fail)

    if read_strand == '+':
        read_sequence = read_seq[query_from:query_to]
    else:
        if query_from == 0:
            read_sequence = read_seq[-query_to:]
        else:
            read_sequence = read_seq[-query_to:-query_from]

    # if the start of the read is not in the reference UTR region then we discard the read
    if interval.strand == '+':
        if not (read_start_position >= interval.start_d and read_start_position <= interval.end_d):
            return None
    else:
        if not (read_start_position <= interval.start_d and read_start_position >= interval.end_d):
            return None

    # get the read's barcode and UMI
    read_umi, read_barcode = protocol.get_reads_identity(alignment=alignment,
                                                         batch=batch)
    if not read_umi:  # if the UMI was not found, we discard the read
        return None
    if not read_barcode:
        return None

    # get read length
    read_length = len(read_sequence)

    # get the computed size of the UTR for this molecule
    utr_size = protocol.get_utr_size_from_read(read_start_position=read_start_position,
                                               interval=interval)

    information = read_start_position, read_strand, read_barcode, \
        read_umi, read_length, utr_size  # , read_sequence

    return information
