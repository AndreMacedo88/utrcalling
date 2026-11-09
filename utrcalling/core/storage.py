"""File storage for the UTR calling pipeline: intermediate files and h5ad outputs."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa
import pybedtools
from scipy.sparse import csr_matrix, save_npz

from utrcalling.core import helpers
from utrcalling.core.module_log import module_logger

if TYPE_CHECKING:
    import anndata


def store_utr_reference(
    utr_bed: pybedtools.BedTool,
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


def _to_parquet_or_csv(df: pd.DataFrame, directory: str, stem: str) -> None:
    """Write *df* to ``<stem>.parquet``, or to ``<stem>.csv`` if parquet cannot store it.

    Parquet needs one type per column; a column mixing e.g. numbers and text
    makes pyarrow raise, and CSV can still hold it.
    """
    try:
        df.to_parquet(os.path.join(directory, f"{stem}.parquet"))
    except pa.ArrowException as err:
        module_logger().warning(f"Cannot write {stem} as parquet ({err}), writing CSV instead")
        df.to_csv(os.path.join(directory, f"{stem}.csv"))


def store_peak_calling_results(
    peaks_count_matrix: csr_matrix,
    peaks_var: pd.DataFrame,
    peaks_obs: pd.DataFrame,
    utr_reference_subset: pd.DataFrame,
    merge_method: str,
    directory: str = "./intermediate",
) -> None:
    """Save peak calling intermediate files.

    Saves the sparse count matrix (.npz), var DataFrame, obs DataFrame,
    and subsetted UTR reference. Tables that parquet cannot store go to CSV.

    Args:
        peaks_count_matrix: Sparse (observations x peaks) count matrix.
        peaks_var: Peaks variable DataFrame.
        peaks_obs: Observations DataFrame.
        utr_reference_subset: Subsetted UTR reference DataFrame.
        merge_method: The merge method used ('union' or 'intersection').
        directory: Directory to save files into.
    """
    logger = module_logger()
    logger.info("Storing intermediate results")
    helpers.check_if_exists_and_mkdir(directory)

    save_npz(
        os.path.join(directory, f"peaks_count_matrix_{merge_method}.npz"),
        peaks_count_matrix,
    )
    _to_parquet_or_csv(peaks_var, directory, f"peaks_var_{merge_method}")
    _to_parquet_or_csv(utr_reference_subset, directory, f"UTR_reference_subset_{merge_method}")
    _to_parquet_or_csv(peaks_obs, directory, f"peaks_obs_{merge_method}")


def write_anndata(adata: anndata.AnnData, filename: str) -> None:
    """Write *adata* to an h5ad file.

    h5ad cannot store a text column that also holds other values, such as an
    annotation column mixing numbers and text. If writing fails on such
    columns of ``.obs`` or ``.var``, they are written as text, with a warning.

    Args:
        adata: The AnnData object to write.
        filename: Path of the h5ad file.

    Raises:
        TypeError: If the file still cannot be written.
    """
    try:
        adata.write_h5ad(filename)
    except TypeError:
        for axis, frame in (("obs", adata.obs), ("var", adata.var)):
            for column in frame.columns:
                values = frame[column]
                if values.dtype == object and values.dropna().map(type).nunique() > 1:
                    module_logger().warning(
                        f"{axis} column {column} mixes value types: writing it as text"
                    )
                    frame[column] = values.astype(str)
        adata.write_h5ad(filename)


def write_table(table: pd.DataFrame, filename: str) -> None:
    """Write a table as tab-separated text, with its index as the first column.

    Missing values are written as empty fields, which pandas and R read as missing.

    Args:
        table: The table to write.
        filename: Path of the text file.
    """
    table.to_csv(filename, sep="\t")
