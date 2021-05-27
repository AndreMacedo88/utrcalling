from collections import defaultdict

import HTSeq

from .module_log import module_logger

# Set up logging
logger = module_logger()

def generate_utr_intervals(utr_regions):
    intervals = dict()
    counter = 0
    for utr in utr_regions:
        name = utr.name
        if name in intervals.keys():
            counter += 1
            name = f'{name}_{counter}'
        else:
            counter = 0
        intervals[name] = HTSeq.GenomicInterval(utr.chrom, utr.start-1, utr.stop, 
        utr.strand)
    return intervals

def map_reads_to_utr(bam, intervals, read='first', version=3):
    bam_file = HTSeq.BAM_Reader(bam)
    read_mappings = dict()
    for name, interval in intervals.items():
        read_information = defaultdict(list)
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
                read_info = extract_read_information(alignment=alignment, version=version)
            
            read_information['read_start_position'].append(read_info[0])
            read_information['read_strand'].append(read_info[1])
            read_information['read_sequence'].append(read_info[2])
            read_information['read_barcode'].append(read_info[3])
            read_information['read_length'].append(read_info[4])
        if read_information:
            read_mappings[name] = read_information
    return read_mappings

def extract_read_information(alignment, version=3):

    # get read strand
    read_strand = alignment.iv.strand

    # get UTR start position (which is the read starting position)
    if read_strand == '+':
        read_start_position = alignment.iv.start
    else:
        read_start_position = alignment.iv.end

    # get read sequence
    query_from = None
    for operation in alignment.cigar:
        if operation.type in ('M', 'I', '=', 'X'):
            if query_from is None:
                query_from = operation.query_from
            query_to = operation.query_to
    if read_strand == '+':
        read_sequence = alignment.read.seq[query_from:query_to]
    else:
        if query_from == 0:
            read_sequence = alignment.read.seq[-query_to:]
        else:
            read_sequence = alignment.read.seq[-query_to:-query_from]
    # get the read's cell barcode and UMI
    if version >= 3:
        read_barcode = alignment.read.seq[:28]
    if version < 3:
        read_barcode = alignment.read.seq[:26]
    else:
        message = f'10x genomics protocol version {version} currently not supported'
        logger.error(message)
        raise RuntimeError(message)

    # get read length
    read_length = len(read_sequence)

    return read_start_position, read_strand, read_sequence, read_barcode, read_length
