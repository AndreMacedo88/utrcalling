"""
# Goal
Mapping RNA-seq reads (single cell or bulk) to UTRs.

# Secondary goal
Call consensus peaks based on the distribution of molecule sizes.

# Summary
This file contains 3 functions for the 3 main steps in this procedure.

# General steps
1. Extract the UTR regions of transcripts (from ENSEMBL)
2. Map the aligned, filtered reads to the UTR regions from step 1.
3. Clean-up the data and call consensus peaks if required.

# More detailed steps
1.1. Resolve overlapping transcripts so that every entry stores unique, non-overlapping,
genomic coordinates

# Observations
- Final object is a count matrix in AnnData format.
- Output should inherit single cell annotations, as well as gene annotation like the
genomic coordinates and optionally UTR sequence information.
"""
from __future__ import annotations

import os
import psutil

import anndata
import numpy as np
import pandas as pd
import pybedtools
import joblib
from scipy.sparse import save_npz
from tqdm import tqdm

from utrcalling.core import helpers
from utrcalling.core.module_log import module_logger
from utrcalling.core import utr_reference
from utrcalling.core import utr_mapping
from utrcalling.core import process_mapped_reads
from utrcalling.core import peak_calling


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
        directory = "./intermediate"
        helpers.check_if_exists_and_mkdir(directory)
        utr_bed.saveas(os.path.join(directory, "utrs_unmerged.bed"))
        utr_reduced.to_parquet(os.path.join(directory, "utr_reference_regions.parquet"))
    return utr_reduced


def map_aligned_reads(
    alignments: str | list[str],
    utr_regions: pd.DataFrame,
    protocol: object,
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
    sample_names = protocol.sample_names

    if store:
        filename = "UTR_sizes_counts"
        directory = "./intermediate"
        helpers.check_if_exists_and_mkdir(directory)

    if isinstance(alignments, (list, tuple)) and len(alignments) > 1:
        n_cores = -2
        # Compute batch indices for each BAM
        if sample_names is not None:
            batch_indices = [protocol.sample_idx[a] for a in alignments]
        else:
            batch_indices = [str(i + 1) for i in range(len(alignments))]

        read_mappings = joblib.Parallel(n_jobs=n_cores)(
            joblib.delayed(utr_mapping.map_reads_to_utr)(
                alignment, intervals, read="first", protocol=protocol, batch_idx=bidx
            )
            for alignment, bidx in tqdm(
                zip(alignments, batch_indices), colour="#ffb700", ascii=" ■"
            )
        )

        if store:
            for counter, utr_sizes_counts in enumerate(read_mappings):
                file_path = os.path.join(directory, f"{filename}_{counter + 1}.parquet")
                utr_sizes_counts.to_parquet(file_path)
    else:
        if isinstance(alignments, (list, tuple)):
            alignments = alignments[0]

        # For single BAM: use sample_idx if available (SmartSeq), else no batch
        batch_idx = None
        if protocol.sample_idx is not None:
            batch_idx = protocol.sample_idx.get(alignments)

        read_mappings = utr_mapping.map_reads_to_utr(
            alignments, intervals, read="first", protocol=protocol, batch_idx=batch_idx
        )
        if store:
            read_mappings.to_parquet(os.path.join(directory, f"{filename}.parquet"))

    return read_mappings


def merge_samples_utr_sizes(
    utr_sizes_counts: list[pd.DataFrame],
    protocol: object,
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
        directory = "./intermediate"
        helpers.check_if_exists_and_mkdir(directory)
        utr_sizes_counts.to_parquet(
            os.path.join(
                directory, f"UTR_sizes_counts_merged_{protocol.merge_method}.parquet"
            )
        )
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


def call_peaks(
    utr_sizes_counts: pd.DataFrame,
    utr_reference_regions: pd.DataFrame,
    protocol: object,
    store: bool = False,
) -> anndata.AnnData:
    """Filter, optionally barcode-filter, and call consensus UTR peaks.

    Applies UTR size filtering, optional barcode whitelist filtering (with
    memory-aware chunking), then runs peak calling on each UTR's size
    distribution to produce a sparse AnnData count matrix.

    Args:
        utr_sizes_counts: Flat DataFrame of (utr_name, observation_id, utr_size) rows.
        utr_reference_regions: Full UTR reference DataFrame.
        protocol: A Protocol instance with peak calling and filtering config.
        store: If True, save intermediate sparse matrix and DataFrames to ./intermediate/.

    Returns:
        An AnnData object with cells/samples as obs and UTR peaks as var.
    """
    logger = module_logger()

    # if we are to add the reference data to the results later, we must subset the UTRs
    utr_reference_subset = process_mapped_reads.subset_reference_data(
        utr_sizes_counts, utr_reference_regions
    )

    del utr_reference_regions  # to save memory

    # get only utr sizes bigger than X bp
    filtered_positions = protocol.filter_utr_sizes(utr_sizes_counts, threshold=1)

    del utr_sizes_counts  # to save memory

    if protocol.barcode_whitelist is not None:
        # Estimate whether barcode filtering will fit in available RAM.
        # filter_barcodes() copies the DataFrame and does string operations,
        # so peak usage is roughly 3x the data size.
        data_bytes = filtered_positions.memory_usage(deep=True).sum()
        available_bytes = psutil.virtual_memory().available
        headroom = data_bytes * 3

        if headroom > available_bytes:
            n_chunks = max(2, int(np.ceil(headroom / available_bytes)))
            logger.info(
                "Splitting barcode filtering into %d chunks to fit in available memory",
                n_chunks,
            )
            num = filtered_positions.shape[0]
            idx_splits = np.cumsum(
                [num // n_chunks + (1 if x < num % n_chunks else 0) for x in range(n_chunks)]
            )
            idx_splits = np.insert(idx_splits, 0, 0)

            split_generator = (
                filtered_positions.iloc[idx_splits[n] : idx_splits[n + 1]]
                for n in range(n_chunks)
            )

            filtered_list = [
                process_mapped_reads.filter_barcodes(
                    pos, whitelist=protocol.barcode_whitelist
                )
                for pos in split_generator
            ]

            filtered_positions = pd.concat(filtered_list, ignore_index=True)
            # Re-categorize after concat to merge category codes
            filtered_positions["utr_name"] = pd.Categorical(filtered_positions["utr_name"])
            filtered_positions["observation_id"] = pd.Categorical(filtered_positions["observation_id"])
        else:
            filtered_positions = process_mapped_reads.filter_barcodes(
                filtered_positions, whitelist=protocol.barcode_whitelist
            )

    # calculate broad peaks based on individual UMI utr sizes
    peaks_count_matrix, *peaks_var_and_obs = peak_calling.generate_peak_counts_and_anno(
        filtered_positions, protocol=protocol
    )

    try:
        del filtered_positions  # to save memory
    except NameError:
        pass

    # retrieve var and obs information
    peaks_var = peaks_var_and_obs[0]
    if len(peaks_var_and_obs) == 2:
        peaks_obs = peaks_var_and_obs[1]
    else:
        peaks_obs = None

    # append the reference information to the new UTR peaks
    peaks_var = peak_calling.add_to_peaks_var(peaks_var, utr_reference_subset)

    if store:  # store the final DFs. Useful for debugging or to change annotations.
        logger.info("Storing intermediate results")
        directory = "./intermediate"
        helpers.check_if_exists_and_mkdir(directory)

        # store count matrix (sparse)
        save_npz(
            os.path.join(
                directory, f"peaks_count_matrix_{protocol.merge_method}.npz"
            ),
            peaks_count_matrix,
        )

        # store var dataframe
        try:
            peaks_var.to_parquet(
                os.path.join(directory, f"peaks_var_{protocol.merge_method}.parquet")
            )
        except Exception:
            peaks_var.to_csv(
                os.path.join(directory, f"peaks_var_{protocol.merge_method}.csv")
            )

        # store subsetted UTR reference dataframe
        try:
            utr_reference_subset.to_parquet(
                os.path.join(
                    directory, f"UTR_reference_subset_{protocol.merge_method}.parquet"
                )
            )
        except Exception:
            utr_reference_subset.to_csv(
                os.path.join(
                    directory, f"UTR_reference_subset_{protocol.merge_method}.csv"
                )
            )

        # store obs dataframe
        try:
            peaks_obs.to_parquet(
                os.path.join(directory, f"peaks_obs_{protocol.merge_method}.parquet")
            )
        except Exception:
            peaks_obs.to_csv(
                os.path.join(directory, f"peaks_obs_{protocol.merge_method}.csv")
            )

    # merge results and annotations into
    count_matrix_anndata = peak_calling.generate_count_matrix_anndata(
        count_matrix=peaks_count_matrix, var=peaks_var, obs=peaks_obs
    )
    return count_matrix_anndata
    # return
