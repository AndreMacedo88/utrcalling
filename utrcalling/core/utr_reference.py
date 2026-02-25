"""
This module contains functions that retrieve UTR information from annotation files
such as GTF and GFF
"""
from __future__ import annotations

import gzip

import pandas as pd
import pybedtools
from pandas.api.types import is_numeric_dtype
from gtfparse import read_gtf

from utrcalling.core.module_log import module_logger


def open_gtf(gtf_file: str) -> pd.DataFrame:
    """Parse a GTF annotation file to a pandas DataFrame.

    Args:
        gtf_file: Path to a GTF annotation file (plain or gzipped).

    Returns:
        A DataFrame with all features from the GTF file.
    """
    logger = module_logger()
    logger.info("Parsing GTF reference file, may take a minute")

    if gtf_file.endswith(".gz"):
        with gzip.open(gtf_file) as f:
            gtf_df = read_gtf(f)
    else:
        gtf_df = read_gtf(gtf_file)

    logger.info("Finished parsing GTF file")
    return gtf_df


def extract_feature(gtf_df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """Extract rows matching a specific feature type from a GTF DataFrame.

    Args:
        gtf_df: GTF annotation DataFrame.
        feature: Feature type to filter (e.g., 'three_prime_utr', 'five_prime_utr').

    Returns:
        A filtered DataFrame containing only rows with the specified feature.
    """
    logger = module_logger()
    logger.info("Filtering GTF reference file to retrieve only the UTR annotations")

    utr_df = gtf_df[gtf_df["feature"] == feature]
    return utr_df


def prepare_annotations_index(
    annotations_extra: list[str],
) -> tuple[list[str], list[int]]:
    """Build column list and index positions for BED merge operations.

    Args:
        annotations_extra: List of additional annotation column names beyond
            the standard seqname/start/end.

    Returns:
        A (columns, indices) tuple where columns is the full column list and
        indices are 1-based positions for the extra annotations.
    """
    main = ["seqname", "start", "end"]
    annotations_columns = main + annotations_extra
    annotations_index = list(range(len(main) + 1, len(annotations_columns) + 1))
    return annotations_columns, annotations_index


def reduce_regions(
    bed: pybedtools.BedTool,
    annotations_index: list[int] | None = None,
    operation_list: list[str] | None = None,
) -> pybedtools.BedTool:
    """Merge overlapping genomic regions in a BED object.

    Args:
        bed: A pybedtools BedTool with UTR regions.
        annotations_index: Column indices to aggregate during merge.
        operation_list: Aggregation operations per column. Defaults to 'distinct'.

    Returns:
        A merged BedTool with non-overlapping regions.
    """
    logger = module_logger()
    if annotations_index is None:
        annotations_index = []
    if not operation_list:
        operations = [
            "distinct",
        ] * len(annotations_index)
    else:
        operations = operation_list

    logger.info("Merging overlapping UTR regions")
    bed_merged = bed.merge(c=annotations_index, o=operations, s=True)
    return bed_merged


def turn_utr_bed_to_dataframe(
    utr_table: pd.DataFrame | pybedtools.BedTool,
) -> pd.DataFrame:
    """Convert a BED-like object to a DataFrame with numeric coordinates.

    Args:
        utr_table: A BedTool or DataFrame with start/end columns.

    Returns:
        A DataFrame with integer start and end columns.
    """
    if not isinstance(utr_table, pd.DataFrame):
        utr_table = utr_table.to_dataframe()
    if not is_numeric_dtype(utr_table["start"]):
        utr_table["start"] = utr_table["start"].astype(int)
    if not is_numeric_dtype(utr_table["end"]):
        utr_table["end"] = utr_table["end"].astype(int)
    return utr_table


def generate_utr_name(utr_df: pd.DataFrame) -> pd.DataFrame:
    """Generate unique UTR names by appending occurrence counts.

    For UTRs with the same gene-level name, appends '_2', '_3', etc.
    to distinguish them. Sets 'utr_name' as the DataFrame index.

    Uses vectorized groupby+cumcount instead of row-by-row iteration.

    Args:
        utr_df: A UTR DataFrame with a 'name' column.

    Returns:
        The DataFrame with a new 'utr_name' index column.

    Raises:
        TypeError: If input is not a DataFrame.
    """
    if not isinstance(utr_df, pd.DataFrame):
        raise TypeError(f"Please turn your {type(utr_df)} object to a pandas DataFrame")
    # sort by name so that name incrementation is handled correctly later on.
    utr_df.sort_values(by=["chrom", "start", "end"], inplace=True)

    # Vectorized: compute occurrence number within each name group
    occurrence = utr_df.groupby("name", sort=False).cumcount()
    # First occurrence (0) gets no suffix; subsequent ones get _2, _3, ...
    suffix = occurrence.map(lambda x: f"_{x + 1}" if x > 0 else "")
    utr_df["utr_name"] = utr_df["name"].astype(str) + suffix

    utr_df = utr_df.set_index("utr_name")
    return utr_df


def correct_column_names(utr_df: pd.DataFrame) -> pd.DataFrame:
    """Rename BED thickStart/thickEnd columns to transcript_id/gene_name.

    Args:
        utr_df: A UTR DataFrame from BED parsing.

    Returns:
        The DataFrame with renamed columns.
    """
    utr_df = utr_df.rename(
        columns={"thickStart": "transcript_id", "thickEnd": "gene_name"}
    )
    return utr_df


def subset_reference_data(
    utr_sizes: pd.DataFrame, utr_reference_regions: pd.DataFrame
) -> pd.DataFrame:
    """Subset the UTR reference to only UTRs present in the read data.

    Args:
        utr_sizes: Flat DataFrame with a 'utr_name' column.
        utr_reference_regions: Full UTR reference DataFrame indexed by utr_name.

    Returns:
        A deduplicated subset of the reference containing only matching UTRs.
    """
    logger = module_logger()
    logger.info("Subsetting UTR reference data")

    utr_names = utr_sizes["utr_name"].unique()
    sub_utr_reference = utr_reference_regions[
        utr_reference_regions.index.isin(utr_names)
    ]
    sub_utr_reference_unique = sub_utr_reference.drop_duplicates()

    return sub_utr_reference_unique
