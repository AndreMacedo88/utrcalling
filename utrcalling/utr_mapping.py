import HTSeq

from .storage import UtrMappings, MappedReadInformation
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
    utr_mappings = UtrMappings()
    for name, interval in intervals.items():
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
                mapped_read_information.read_start_positions.append(read_info[0])
                mapped_read_information.read_strands.append(read_info[1])
                mapped_read_information.read_sequences.append(read_info[2])
                mapped_read_information.read_barcodes.append(read_info[3])
                mapped_read_information.read_umis.append(read_info[4])
                mapped_read_information.read_lengths.append(read_info[5])
                mapped_read_information.utr_sizes_from_reads.append(read_info[6])

        if mapped_read_information:
            utr_mappings[name] = mapped_read_information
    return utr_mappings

def extract_read_information(alignment, interval, version=3):

    # get read strand
    read_strand = alignment.iv.strand

    # get UTR start position (which is the read starting position)
    read_start_position = alignment.iv.start_d

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
    read_barcode = alignment.read.seq[:16]
    if version >= 3:
        read_umi = alignment.read.seq[17:28]
    if version < 3:
        read_umi = alignment.read.seq[17:26]
    else:
        message = f'10x genomics protocol version {version} currently not supported'
        logger.error(message)
        raise RuntimeError(message)

    # get read length
    read_length = len(read_sequence)

    # get the computed size of the UTR for this molecule
    utr_size = abs(read_start_position - interval.start_d)

    information = read_start_position, read_strand, read_sequence, read_barcode, \
    read_umi, read_length, utr_size

    return information
