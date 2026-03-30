"""Core pipeline orchestrator for UTR calling.

Maps RNA-seq reads (single cell or bulk) to UTRs and calls consensus peaks
based on the distribution of molecule sizes.

Pipeline steps (each a thin function calling focused modules):
1. prepare_utr_reference_regions() — parse GTF, extract & merge UTR regions
2. map_aligned_reads() — map BAM reads to UTR intervals (single or multi-BAM)
3. merge_samples_utr_sizes() — merge multiple sample DataFrames
4. merge_utrs_per_feature() — optionally aggregate to gene level
5. filter_mapped_reads() — subset reference, filter UTR sizes, filter barcodes
6. call_peaks() — run peak calling, enrich annotations, build AnnData

Output is a count matrix in AnnData format with cells/samples as obs and
UTR peaks as var.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import anndata
import pandas as pd
import pybedtools
import joblib
from tqdm import tqdm

from utrcalling.core.module_log import module_logger
from utrcalling.core.helpers import JOBLIB_DEFAULT_N_JOBS, JOBLIB_DEFAULT_BACKEND
from utrcalling.core import utr_reference
from utrcalling.core import utr_mapping
from utrcalling.core import process_mapped_reads
from utrcalling.core import count_matrix
from utrcalling.core import barcode_filtering
from utrcalling.core import storage
from utrcalling.core.protocol import Protocol
from utrcalling.core.utr_mapping import _IntervalData

if TYPE_CHECKING:
    from collections.abc import Callable
    import numpy.typing as npt


def prepare_utr_reference_regions(gtf: str, feature: str, store: bool = False) -> pd.DataFrame:
    """Parse a GTF file and build a non-overlapping UTR reference.

    Extracts UTR features from the GTF, sorts by chromosome and start position,
    merges overlapping regions via pybedtools, and returns a cleaned DataFrame
    indexed by UTR name.

    Args:
        gtf: Path to a GTF annotation file (gzipped or plain).
        feature: GTF feature type to extract (e.g., 'three_prime_utr').
        store: If True, save intermediate BED and Parquet files to ./intermediate/.

    Returns:
        A DataFrame of merged, non-overlapping UTR regions indexed by utr_name.
    """
    # open the gtf file
    gtf_df = utr_reference.open_gtf(gtf)
    # extract 3'UTRs from all the features in the GTF file
    utr = utr_reference.extract_feature(gtf_df, feature)
    # sort by chr and start for bed merge
    utr = utr.sort_values(by=["seqname", "start"])
    annotations = ["gene_id", "score", "strand", "transcript_id", "gene_name"]
    ann_cols, ann_idx = utr_reference.prepare_annotations_index(annotations)
    # Create BedTool obj using the GTF dataframe
    utr_bed = pybedtools.BedTool.from_dataframe(utr, columns=ann_cols)
    # find and merge overlapping regions
    utr_reduced = utr_reference.reduce_regions(utr_bed, annotations_index=ann_idx)
    utr_reduced = utr_reference.turn_utr_bed_to_dataframe(utr_reduced)
    utr_reduced = utr_reference.generate_utr_name(utr_reduced)
    utr_reduced = utr_reference.correct_column_names(utr_reduced)
    utr_reduced.chrom = utr_reduced.chrom.astype(str)

    if store:
        storage.store_utr_reference(utr_bed, utr_reduced)
    return utr_reduced


def _resolve_batch_idx(alignment: str, protocol: Protocol) -> str | None:
    """Resolve the batch index for a single BAM file.

    Args:
        alignment: Path to a BAM file.
        protocol: A Protocol instance with sample_idx mapping.

    Returns:
        The batch index string, or None if no sample index is configured.
    """
    if protocol.sample_idx is not None:
        return protocol.sample_idx.get(alignment)
    return None


def _map_single_bam(
    alignment: str,
    intervals: dict[str, _IntervalData],
    protocol: Protocol,
    batch_idx: str | None = None,
) -> pd.DataFrame:
    """Map reads from a single BAM file to UTR intervals.

    Args:
        alignment: Path to a BAM file.
        intervals: Dict of {utr_name: _IntervalData}.
        protocol: A Protocol instance.
        batch_idx: Optional batch index for observation ID suffixing.

    Returns:
        A flat DataFrame of (utr_name, observation_id, utr_size) rows.
    """
    return utr_mapping.map_reads_to_utr(
        alignment, intervals, read=protocol.read_in_pair,
        protocol=protocol, batch_idx=batch_idx,
    )


def _map_multiple_bams(
    alignments: list[str],
    intervals: dict[str, _IntervalData],
    protocol: Protocol,
) -> list[pd.DataFrame]:
    """Map reads from multiple BAM files in parallel via joblib.

    Args:
        alignments: List of BAM file paths.
        intervals: Dict of {utr_name: _IntervalData}.
        protocol: A Protocol instance.

    Returns:
        List of flat DataFrames, one per BAM file.
    """
    if protocol.sample_names is not None:
        batch_indices = [protocol.sample_idx[a] for a in alignments]
    else:
        batch_indices = [str(i + 1) for i in range(len(alignments))]

    return joblib.Parallel(n_jobs=JOBLIB_DEFAULT_N_JOBS, backend=JOBLIB_DEFAULT_BACKEND)(
        joblib.delayed(utr_mapping.map_reads_to_utr)(
            alignment, intervals, read=protocol.read_in_pair, protocol=protocol,
            batch_idx=bidx, parallel=False,
        )
        for alignment, bidx in tqdm(
            zip(alignments, batch_indices), colour="#ffb700", ascii=" ■"
        )
    )


def map_aligned_reads(
    alignments: str | list[str],
    utr_regions: pd.DataFrame,
    protocol: Protocol,
    store: bool = False,
) -> pd.DataFrame | list[pd.DataFrame]:
    """Map BAM-aligned reads to UTR intervals and deduplicate by UMI.

    For multiple BAM files, parallelizes across samples using joblib. Each BAM
    produces a flat DataFrame of (utr_name, observation_id, utr_size) rows.

    Args:
        alignments: Path to a single BAM file, or a list of BAM file paths.
        utr_regions: UTR reference DataFrame (from prepare_utr_reference_regions).
        protocol: A Protocol instance defining read parsing and identity extraction.
        store: If True, save per-sample Parquet files to ./intermediate/.

    Returns:
        A single flat DataFrame (one BAM) or list of flat DataFrames (multiple BAMs).
    """
    intervals = utr_mapping.generate_utr_intervals(utr_regions)

    if isinstance(alignments, (list, tuple)) and len(alignments) > 1:
        read_mappings = _map_multiple_bams(alignments, intervals, protocol)
        if store:
            storage.store_read_mappings_multi(read_mappings)
    else:
        if isinstance(alignments, (list, tuple)):
            alignments = alignments[0]
        batch_idx = _resolve_batch_idx(alignments, protocol)
        read_mappings = _map_single_bam(alignments, intervals, protocol, batch_idx)
        if store:
            storage.store_read_mappings_single(read_mappings)

    return read_mappings


def merge_samples_utr_sizes(
    utr_sizes_counts: list[pd.DataFrame],
    protocol: Protocol,
    store: bool = False,
) -> pd.DataFrame:
    """Merge flat DataFrames from multiple BAM samples into one.

    Delegates to process_mapped_reads.merge_samples_utr_sizes using the
    protocol's merge_method ('union' or 'intersection').

    Args:
        utr_sizes_counts: List of flat DataFrames, one per sample.
        protocol: A Protocol instance with a merge_method attribute.
        store: If True, save the merged Parquet file to ./intermediate/.

    Returns:
        A single merged flat DataFrame.
    """
    utr_sizes_counts = process_mapped_reads.merge_samples_utr_sizes(
        utr_sizes_counts,
        method=protocol.merge_method,
    )
    if store:
        storage.store_merged_counts(utr_sizes_counts, protocol.merge_method)
    return utr_sizes_counts


def merge_utrs_per_feature(
    utr_regions: pd.DataFrame,
    read_mappings: pd.DataFrame,
    level: str = "UTR",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Optionally merge per-UTR data up to gene level.

    When level='gene', UTR regions belonging to the same gene are merged and
    UTR sizes are offset-adjusted so they represent position within the full
    gene UTR span.

    Args:
        utr_regions: UTR reference DataFrame indexed by utr_name.
        read_mappings: Flat DataFrame of (utr_name, observation_id, utr_size) rows.
        level: Aggregation level — 'UTR' (no-op) or 'gene'.

    Returns:
        A (utr_regions, read_mappings) tuple, potentially merged to gene level.
    """
    if level.lower() == "gene":
        utr_regions, read_mappings = process_mapped_reads.merge_utrs_per_gene(
            utr_regions=utr_regions, read_mappings=read_mappings
        )
    return utr_regions, read_mappings


def filter_mapped_reads(
    utr_sizes_counts: pd.DataFrame,
    utr_reference_regions: pd.DataFrame,
    protocol: Protocol,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter mapped reads and subset the UTR reference before peak calling.

    Applies three preparation steps:
    1. Subset the UTR reference to only UTRs present in the read data.
    2. Filter out UTR sizes below the threshold.
    3. Optionally filter barcodes against a whitelist (memory-aware).

    Args:
        utr_sizes_counts: Flat DataFrame of (utr_name, observation_id, utr_size) rows.
        utr_reference_regions: Full UTR reference DataFrame.
        protocol: A Protocol instance with filtering config.

    Returns:
        A (filtered_reads, utr_reference_subset) tuple.
    """
    utr_reference_subset = utr_reference.subset_reference_data(
        utr_sizes_counts, utr_reference_regions
    )
    filtered = protocol.filter_utr_sizes(utr_sizes_counts, threshold=1)
    if protocol.barcode_whitelist is not None:
        filtered = barcode_filtering.filter_barcodes_memory_aware(
            filtered, protocol.barcode_whitelist
        )
    return filtered, utr_reference_subset


def build_count_matrix(
    filtered_reads: pd.DataFrame,
    utr_reference_subset: pd.DataFrame,
    protocol: Protocol,
    store: bool = False,
    size_transform: Callable[[npt.ArrayLike], npt.ArrayLike] | None = None,
) -> anndata.AnnData:
    """Build a sparse AnnData count matrix from filtered reads.

    Groups molecules by UTR, optionally transforms sizes via ``size_transform``,
    enriches var with reference annotations, and returns an AnnData object.

    Args:
        filtered_reads: Filtered flat DataFrame (utr_name, observation_id, utr_size).
        utr_reference_subset: Subsetted UTR reference for annotation enrichment.
        protocol: Protocol instance (for annotations, barcode config).
        store: If True, save intermediates to ./intermediate/.
        size_transform: Optional function to transform UTR sizes per group.
            None -> raw sizes. E.g., pc.get_caller("kde", bandwidth=12) for KDE peaks.

    Returns:
        An AnnData object with cells/samples as obs and UTR peaks as var.
    """
    result = count_matrix.generate_counts_and_anno(
        filtered_reads, protocol=protocol, size_transform=size_transform
    )
    peaks_count_matrix = result.count_matrix
    peaks_var = result.peaks_var
    peaks_obs = result.peaks_obs

    peaks_var = count_matrix.add_to_peaks_var(peaks_var, utr_reference_subset)

    if store:
        storage.store_peak_calling_results(
            peaks_count_matrix, peaks_var, peaks_obs,
            utr_reference_subset, protocol.merge_method,
        )

    return count_matrix.generate_count_matrix_anndata(
        count_matrix=peaks_count_matrix, var=peaks_var, obs=peaks_obs
    )


def call_peaks(
    filtered_reads: pd.DataFrame,
    utr_reference_subset: pd.DataFrame,
    protocol: Protocol,
    store: bool = False,
) -> anndata.AnnData:
    """Call consensus UTR peaks and build a sparse AnnData count matrix.

    Convenience wrapper around build_count_matrix() that uses the protocol's
    configured peak calling function as the size transform.

    Args:
        filtered_reads: Filtered flat DataFrame (from filter_mapped_reads).
        utr_reference_subset: Subsetted UTR reference (from filter_mapped_reads).
        protocol: A Protocol instance with peak calling config.
        store: If True, save intermediate files to ./intermediate/.

    Returns:
        An AnnData object with cells/samples as obs and UTR peaks as var.
    """
    return build_count_matrix(
        filtered_reads, utr_reference_subset, protocol, store,
        size_transform=protocol.peak_calling,
    )
