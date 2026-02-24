"""Intermediate file storage utilities for the UTR calling pipeline."""
from __future__ import annotations

import os

import pandas as pd
from scipy.sparse import save_npz, csr_matrix

from utrcalling.core import helpers
from utrcalling.core.module_log import module_logger


def store_utr_reference(
    utr_bed: object,
    utr_reduced: pd.DataFrame,
    directory: str = "./intermediate",
) -> None:
    """Save unmerged BED and merged parquet UTR reference files.

    Args:
        utr_bed: A pybedtools BedTool object with unmerged UTR regions.
        utr_reduced: The merged UTR reference DataFrame.
        directory: Directory to save files into.
    """
    helpers.check_if_exists_and_mkdir(directory)
    utr_bed.saveas(os.path.join(directory, "utrs_unmerged.bed"))
    utr_reduced.to_parquet(os.path.join(directory, "utr_reference_regions.parquet"))


def store_read_mappings_multi(
    read_mappings_list: list[pd.DataFrame],
    directory: str = "./intermediate",
    filename: str = "UTR_sizes_counts",
) -> None:
    """Save per-sample read mapping parquet files (multi-BAM).

    Args:
        read_mappings_list: List of DataFrames, one per sample.
        directory: Directory to save files into.
        filename: Base filename for the parquet files.
    """
    helpers.check_if_exists_and_mkdir(directory)
    for counter, utr_sizes_counts in enumerate(read_mappings_list):
        file_path = os.path.join(directory, f"{filename}_{counter + 1}.parquet")
        utr_sizes_counts.to_parquet(file_path)


def store_read_mappings_single(
    read_mappings: pd.DataFrame,
    directory: str = "./intermediate",
    filename: str = "UTR_sizes_counts",
) -> None:
    """Save a single read mapping parquet file (single-BAM).

    Args:
        read_mappings: A single flat DataFrame of mapped reads.
        directory: Directory to save files into.
        filename: Base filename for the parquet file.
    """
    helpers.check_if_exists_and_mkdir(directory)
    read_mappings.to_parquet(os.path.join(directory, f"{filename}.parquet"))


def store_merged_counts(
    utr_sizes_counts: pd.DataFrame,
    merge_method: str,
    directory: str = "./intermediate",
) -> None:
    """Save merged UTR sizes counts parquet file.

    Args:
        utr_sizes_counts: Merged flat DataFrame of UTR sizes counts.
        merge_method: The merge method used ('union' or 'intersection').
        directory: Directory to save files into.
    """
    helpers.check_if_exists_and_mkdir(directory)
    utr_sizes_counts.to_parquet(
        os.path.join(directory, f"UTR_sizes_counts_merged_{merge_method}.parquet")
    )


def store_peak_calling_results(
    peaks_count_matrix: csr_matrix,
    peaks_var: pd.DataFrame,
    peaks_obs: pd.DataFrame | None,
    utr_reference_subset: pd.DataFrame,
    merge_method: str,
    directory: str = "./intermediate",
) -> None:
    """Save peak calling intermediate files.

    Saves the sparse count matrix (.npz), var DataFrame, obs DataFrame,
    and subsetted UTR reference. Falls back to CSV if parquet writing fails.

    Args:
        peaks_count_matrix: Sparse (observations x peaks) count matrix.
        peaks_var: Peaks variable DataFrame.
        peaks_obs: Observations DataFrame (may be None).
        utr_reference_subset: Subsetted UTR reference DataFrame.
        merge_method: The merge method used ('union' or 'intersection').
        directory: Directory to save files into.
    """
    logger = module_logger()
    logger.info("Storing intermediate results")
    helpers.check_if_exists_and_mkdir(directory)

    # store count matrix (sparse)
    save_npz(
        os.path.join(directory, f"peaks_count_matrix_{merge_method}.npz"),
        peaks_count_matrix,
    )

    # store var dataframe
    try:
        peaks_var.to_parquet(
            os.path.join(directory, f"peaks_var_{merge_method}.parquet")
        )
    except Exception:
        peaks_var.to_csv(
            os.path.join(directory, f"peaks_var_{merge_method}.csv")
        )

    # store subsetted UTR reference dataframe
    try:
        utr_reference_subset.to_parquet(
            os.path.join(directory, f"UTR_reference_subset_{merge_method}.parquet")
        )
    except Exception:
        utr_reference_subset.to_csv(
            os.path.join(directory, f"UTR_reference_subset_{merge_method}.csv")
        )

    # store obs dataframe
    try:
        peaks_obs.to_parquet(
            os.path.join(directory, f"peaks_obs_{merge_method}.parquet")
        )
    except Exception:
        peaks_obs.to_csv(
            os.path.join(directory, f"peaks_obs_{merge_method}.csv")
        )
