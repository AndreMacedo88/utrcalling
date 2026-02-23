"""
This module contains functions to get UTR sizes from Pandas DataFrames representing
BED files with UTR regions annotations

"""
from __future__ import annotations

# general tools
from utrcalling.core.module_log import module_logger
from functools import partial
import logging
import psutil
from pympler import asizeof

# data handling tools
import HTSeq
import numpy as np
import pandas as pd

# QOL tools
from tqdm import tqdm as std_tqdm

tqdm = partial(std_tqdm, dynamic_ncols=True)


def generate_utr_intervals(utr_regions: pd.DataFrame) -> dict[str, HTSeq.GenomicInterval]:
    """Build a dict mapping UTR names to HTSeq GenomicInterval objects.

    Args:
        utr_regions: UTR reference DataFrame with columns chrom, start, end, strand,
            indexed by utr_name.

    Returns:
        A dict of {utr_name: HTSeq.GenomicInterval}.
    """
    logger = module_logger()
    intervals = dict()

    logger.info("Generating genomic interval for UTR number")
    utrs = utr_regions.itertuples()
    for utr in tqdm(
        utrs, total=len(utr_regions), colour="#ffb700", ascii=" ■", desc="UTR: "
    ):
        try:
            intervals[utr.Index] = HTSeq.GenomicInterval(
                utr.chrom, utr.start - 1, utr.end, utr.strand
            )
        except TypeError:
            intervals[utr.Index] = HTSeq.GenomicInterval(
                str(utr.chrom), utr.start - 1, utr.end, utr.strand
            )
    return intervals


def map_reads_to_utr(
    bam: str,
    intervals: dict[str, HTSeq.GenomicInterval],
    protocol: object,
    read: str = "first",
    batch_idx: str | None = None,
) -> pd.DataFrame:
    """Extract reads from a BAM file and map them to UTR intervals.

    Iterates over each UTR interval, extracts matching reads, deduplicates
    by UMI per barcode, and returns a flat DataFrame of molecules.

    Args:
        bam: Path to a BAM alignment file.
        intervals: Dict of {utr_name: HTSeq.GenomicInterval} from generate_utr_intervals.
        protocol: A Protocol instance for read parsing and identity extraction.
        read: Which mate to use ('first' or 'second').
        batch_idx: Optional batch index for observation ID suffixing.

    Returns:
        A flat DataFrame with columns (utr_name, observation_id, utr_size),
        using Categorical dtypes for string columns and int32 for sizes.
    """
    logger = module_logger()
    logger.info("Extracting reads from %s and mapping them to the reference UTR", bam)

    label_batches = protocol.label_batches

    bam_file = HTSeq.BAM_Reader(bam)

    # Flat accumulators — one row per (utr, observation, size)
    all_utr_names = []
    all_observation_ids = []
    all_utr_sizes = []

    read_strand_string = f"Reads not '{read}'"
    mapping_stats = {
        "Reads mapped to the region of interest": 0,
        read_strand_string: 0,
        "Alignments not proper pair": 0,
        "PCR duplicates": 0,
        "Alignments not primary alignment": 0,
        "Alignments in interval but wrong strand": 0,
    }

    for name, interval in tqdm(
        intervals.items(), colour="#ffb700", ascii=" ■", desc="Interval: "
    ):
        read_barcodes = []
        read_umis = []
        utr_sizes_from_reads = []

        mapped_read_information = [read_barcodes, utr_sizes_from_reads]
        try:
            for alignment in bam_file[interval]:
                mapping_stats["Reads mapped to the region of interest"] += 1
                if not alignment.pe_which == read:
                    mapping_stats[read_strand_string] += 1
                elif not alignment.proper_pair:
                    mapping_stats["Alignments not proper pair"] += 1
                elif alignment.pcr_or_optical_duplicate:
                    mapping_stats["PCR duplicates"] += 1
                elif alignment.not_primary_alignment:
                    mapping_stats["Alignments not primary alignment"] += 1
                elif not protocol.check_strand_matches(
                    alignment.iv.strand, interval.strand
                ):
                    mapping_stats["Alignments in interval but wrong strand"] += 1
                else:
                    read_info = extract_read_information(
                        alignment=alignment,
                        interval=interval,
                        protocol=protocol,
                        bam_name=bam,
                        batch_idx=batch_idx,
                    )
                    if read_info is None:
                        continue
                    else:
                        read_barcodes.append(read_info[0])
                        read_umis.append(read_info[1])
                        utr_sizes_from_reads.append(read_info[2])
        except ValueError:
            continue

        # Deduplicate reads using the UMIs
        if read_umis:
            mapped_read_information = deduplicate_reads(
                read_umis=read_umis,
                read_info=mapped_read_information,
            )

            # Extend flat accumulators
            n_reads = len(mapped_read_information[0])
            all_utr_names.extend([name] * n_reads)
            all_observation_ids.extend(mapped_read_information[0])
            all_utr_sizes.extend(mapped_read_information[1])

    del bam_file

    utr_mappings = pd.DataFrame({
        "utr_name": pd.Categorical(all_utr_names),
        "observation_id": pd.Categorical(all_observation_ids),
        "utr_size": np.array(all_utr_sizes, dtype=np.int32),
    })

    # most performant method to log
    logger.info("Mapping statistics: %s", mapping_stats)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Size of mapped reads object in bytes: %s \nMemory usage: %s",
            asizeof.asizeof(utr_mappings),
            psutil.virtual_memory().percent,
        )

    return utr_mappings


def deduplicate_reads(read_umis: list[str], read_info: list[list]) -> list[list]:
    """Remove PCR duplicates by keeping the first occurrence of each (barcode, UMI) pair.

    Args:
        read_umis: List of UMI strings, one per read.
        read_info: List of parallel lists (barcodes, sizes, ...) to be filtered
            in sync with read_umis.

    Returns:
        A filtered copy of read_info with duplicate (barcode, UMI) pairs removed.
    """
    barcodes = read_info[0]
    umis_unique = {barcode: set() for barcode in barcodes}
    idx_unique = []
    for idx, (umi, barcode) in enumerate(zip(read_umis, barcodes)):
        barcode_umis = umis_unique[barcode]
        if umi not in barcode_umis:
            idx_unique.append(idx)
            barcode_umis.add(umi)
    read_info_output = [[element[i] for i in idx_unique] for element in read_info]
    return read_info_output



def extract_read_information(
    alignment: object,
    interval: object,
    protocol: object,
    bam_name: str,
    batch_idx: str | None = None,
) -> tuple[str, str, int] | None:
    """Extract observation ID, UMI, and UTR size from a single read alignment.

    Validates that the read starts within the UTR interval, extracts barcode
    and UMI via the protocol, and computes the UTR size.

    Args:
        alignment: An HTSeq SAM_Alignment object.
        interval: An HTSeq GenomicInterval for the UTR region.
        protocol: A Protocol instance for read parsing and identity extraction.
        bam_name: Path to the source BAM file (used by some protocols for identity).
        batch_idx: Optional batch index for observation ID suffixing.

    Returns:
        A (observation_id, umi, utr_size) tuple, or None if the read is invalid.
    """
    # logger = module_logger()

    # get the read starting position. This may, or may not, be the match start position
    read_start_position = alignment.iv.start_d

    # if the start of the read is not in the reference UTR region then we discard it
    if interval.strand == "+":
        if not (
            read_start_position >= interval.start_d
            and read_start_position <= interval.end_d
        ):
            return None
    else:
        if not (
            read_start_position <= interval.start_d
            and read_start_position >= interval.end_d
        ):
            return None

    # get the read's barcode and UMI
    read_umi, read_barcode = protocol.get_reads_identity(
        alignment=alignment, bam_name=bam_name
    )
    if not read_umi:  # if the UMI was not found, we discard the read
        return None
    if not read_barcode:
        return None

    # get the computed size of the UTR for this molecule
    utr_size = protocol.get_utr_size_from_read(
        read_start_position=read_start_position, interval=interval
    )

    observation_id = protocol.get_observation_id(read_barcode, batch_idx)
    information = observation_id, read_umi, utr_size

    return information
