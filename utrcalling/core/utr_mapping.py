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
import joblib
import numpy as np
import pandas as pd
import pysam

# QOL tools
from tqdm import tqdm as std_tqdm

tqdm = partial(std_tqdm, dynamic_ncols=True)


# ---------------------------------------------------------------------------
# Lightweight adapters so pysam reads satisfy the Protocol interface
# ---------------------------------------------------------------------------

_COMPLEMENT = str.maketrans('ACGTacgtNn', 'TGCAtgcaNn')


def _reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    return seq.translate(_COMPLEMENT)[::-1]


class _PySamReadAdapter:
    """Expose pysam sequence/name as HTSeq-compatible .read attributes.

    HTSeq returns the *original* read sequence (reverse-complementing the
    stored BAM SEQ for reverse-strand reads).  pysam returns the stored
    sequence as-is.  This adapter bridges the gap so protocol identity
    extraction sees the same bytes HTSeq would provide.
    """
    __slots__ = ('seq', 'name')

    def __init__(self, pysam_read):
        seq = pysam_read.query_sequence or ''
        if pysam_read.is_reverse:
            seq = _reverse_complement(seq)
        self.seq = seq
        self.name = pysam_read.query_name or ''


class _PySamAdapter:
    """Make a pysam AlignedSegment quack like an HTSeq SAM_Alignment."""
    __slots__ = ('read', '_get_tag')

    def __init__(self, pysam_read):
        self.read = _PySamReadAdapter(pysam_read)
        self._get_tag = pysam_read.get_tag

    def optional_field(self, tag):
        return self._get_tag(tag)


class _IntervalData:
    """Picklable snapshot of an HTSeq GenomicInterval's fields."""
    __slots__ = ('chrom', 'start', 'end', 'strand', 'start_d', 'end_d')

    def __init__(self, interval):
        self.chrom = interval.chrom
        self.start = interval.start
        self.end = interval.end
        self.strand = interval.strand
        self.start_d = interval.start_d
        self.end_d = interval.end_d


# ---------------------------------------------------------------------------
# Interval processing (picklable top-level function for joblib)
# ---------------------------------------------------------------------------

def _process_interval_batch(
    bam_path: str,
    interval_batch: list[tuple[str, _IntervalData]],
    protocol: object,
    read: str,
    batch_idx: str | None,
    show_progress: bool = False,
) -> tuple[list, list, list, dict]:
    """Process a batch of UTR intervals from a BAM file.

    Each call opens its own pysam file handle so the function is safe for
    use inside joblib workers.

    Args:
        bam_path: Path to an indexed BAM file.
        interval_batch: List of (utr_name, _IntervalData) pairs.
        protocol: A Protocol instance.
        read: Which mate to use ('first' or 'second').
        batch_idx: Optional batch index for observation ID.
        show_progress: Whether to show a tqdm progress bar.

    Returns:
        A tuple of (utr_names, observation_ids, utr_sizes, mapping_stats).
    """
    # Bitmask flag filtering — replaces 4 HTSeq property accesses per read
    read_flag = 0x40 if read == "first" else 0x80
    required_flags = 0x2 | read_flag          # proper_pair + correct mate
    excluded_flags = 0x100 | 0x400            # secondary + duplicate

    read_strand_string = f"Reads not '{read}'"
    mapping_stats = {
        "Reads mapped to the region of interest": 0,
        read_strand_string: 0,
        "Alignments not proper pair": 0,
        "PCR duplicates": 0,
        "Alignments not primary alignment": 0,
        "Alignments in interval but wrong strand": 0,
    }

    # Cache protocol method lookups as local variables
    check_strand = protocol.check_strand_matches
    get_identity = protocol.get_reads_identity
    get_utr_size = protocol.get_utr_size_from_read
    get_obs_id = protocol.get_observation_id

    all_utr_names: list[str] = []
    all_observation_ids: list[str] = []
    all_utr_sizes: list[int] = []

    iterator = (
        tqdm(interval_batch, colour="#ffb700", ascii=" ■", desc="Interval: ")
        if show_progress
        else interval_batch
    )

    with pysam.AlignmentFile(bam_path, "rb") as bam_file:
        for name, interval in iterator:
            # Pre-extract interval fields once per interval
            iv_chrom = interval.chrom
            iv_start = interval.start
            iv_end = interval.end
            iv_strand = interval.strand
            iv_start_d = interval.start_d
            iv_end_d = interval.end_d

            read_barcodes: list[str] = []
            read_umis: list[str] = []
            utr_sizes_from_reads: list[int] = []

            try:
                for aln in bam_file.fetch(iv_chrom, iv_start, iv_end):
                    mapping_stats["Reads mapped to the region of interest"] += 1

                    # Fast-path flag check (2 integer ops for good reads)
                    flag = aln.flag
                    if (flag & required_flags) != required_flags or (flag & excluded_flags):
                        # Categorize rejected reads for stats
                        if not (flag & read_flag):
                            mapping_stats[read_strand_string] += 1
                        elif not (flag & 0x2):
                            mapping_stats["Alignments not proper pair"] += 1
                        elif flag & 0x400:
                            mapping_stats["PCR duplicates"] += 1
                        elif flag & 0x100:
                            mapping_stats["Alignments not primary alignment"] += 1
                        continue

                    # Strand check (inlined)
                    read_strand = "-" if aln.is_reverse else "+"
                    if not check_strand(read_strand, iv_strand):
                        mapping_stats["Alignments in interval but wrong strand"] += 1
                        continue

                    # Directional start position (matches HTSeq convention)
                    read_start_d = aln.reference_end - 1 if aln.is_reverse else aln.reference_start

                    # Position bounds check (inlined from extract_read_information)
                    if iv_strand == "+":
                        if read_start_d < iv_start_d or read_start_d > iv_end_d:
                            continue
                    else:
                        if read_start_d > iv_start_d or read_start_d < iv_end_d:
                            continue

                    # Identity extraction via adapter (only for reads that pass all checks)
                    adapter = _PySamAdapter(aln)
                    read_umi, read_barcode = get_identity(alignment=adapter, bam_name=bam_path)
                    if not read_umi or not read_barcode:
                        continue

                    utr_size = get_utr_size(read_start_d, interval)
                    observation_id = get_obs_id(read_barcode, batch_idx)

                    read_barcodes.append(observation_id)
                    read_umis.append(read_umi)
                    utr_sizes_from_reads.append(utr_size)
            except ValueError:
                continue

            # Deduplicate reads using the UMIs
            if read_umis:
                mapped_read_information = deduplicate_reads(
                    read_umis=read_umis,
                    read_info=[read_barcodes, utr_sizes_from_reads],
                )

                # Extend flat accumulators
                n_reads = len(mapped_read_information[0])
                all_utr_names.extend([name] * n_reads)
                all_observation_ids.extend(mapped_read_information[0])
                all_utr_sizes.extend(mapped_read_information[1])

    return all_utr_names, all_observation_ids, all_utr_sizes, mapping_stats


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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
    parallel: bool = True,
) -> pd.DataFrame:
    """Extract reads from a BAM file and map them to UTR intervals.

    Iterates over each UTR interval, extracts matching reads, deduplicates
    by UMI per barcode, and returns a flat DataFrame of molecules.

    When *parallel* is True and there are more than 50 intervals, processing
    is distributed across CPU cores via joblib.

    Args:
        bam: Path to a BAM alignment file.
        intervals: Dict of {utr_name: HTSeq.GenomicInterval} from generate_utr_intervals.
        protocol: A Protocol instance for read parsing and identity extraction.
        read: Which mate to use ('first' or 'second').
        batch_idx: Optional batch index for observation ID suffixing.
        parallel: If True, parallelize across intervals when >50. Set to False
            when already inside a joblib worker (multi-BAM path) to prevent
            nested parallelism.

    Returns:
        A flat DataFrame with columns (utr_name, observation_id, utr_size),
        using Categorical dtypes for string columns and int32 for sizes.
    """
    logger = module_logger()
    logger.info("Extracting reads from %s and mapping them to the reference UTR", bam)

    # Convert HTSeq intervals to picklable _IntervalData
    interval_items = [(name, _IntervalData(iv)) for name, iv in intervals.items()]

    read_strand_string = f"Reads not '{read}'"
    stat_keys = [
        "Reads mapped to the region of interest",
        read_strand_string,
        "Alignments not proper pair",
        "PCR duplicates",
        "Alignments not primary alignment",
        "Alignments in interval but wrong strand",
    ]

    if parallel and len(interval_items) > 50:
        # Parallel path — split intervals across workers
        n_workers = max(1, joblib.cpu_count() - 1)
        batch_size = max(1, len(interval_items) // (n_workers * 4))
        batches = [
            interval_items[i : i + batch_size]
            for i in range(0, len(interval_items), batch_size)
        ]

        results = joblib.Parallel(n_jobs=-2)(
            joblib.delayed(_process_interval_batch)(
                bam, batch, protocol, read, batch_idx, show_progress=False
            )
            for batch in batches
        )

        # Merge results from all workers
        all_utr_names: list[str] = []
        all_observation_ids: list[str] = []
        all_utr_sizes: list[int] = []
        merged_stats = {k: 0 for k in stat_keys}
        for utr_names, obs_ids, utr_sizes, stats in results:
            all_utr_names.extend(utr_names)
            all_observation_ids.extend(obs_ids)
            all_utr_sizes.extend(utr_sizes)
            for k, v in stats.items():
                merged_stats[k] += v
        mapping_stats = merged_stats
    else:
        # Sequential path — single batch with progress bar
        all_utr_names, all_observation_ids, all_utr_sizes, mapping_stats = (
            _process_interval_batch(
                bam, interval_items, protocol, read, batch_idx, show_progress=True
            )
        )

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
