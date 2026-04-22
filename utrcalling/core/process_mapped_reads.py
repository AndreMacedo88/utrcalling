from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np

from utrcalling.core import module_log


# ---------------------------------------------------------------------------
# Module-level private helpers (extracted for testability)
# ---------------------------------------------------------------------------

def _join_unique(x: pd.Series) -> str:
    """Join unique values in a Series with '|' delimiter."""
    return "|".join(x.unique().astype(str))


def _split_and_then_join_unique(x: pd.Series) -> str:
    """Split delimited values, deduplicate, and rejoin with '|'.

    Handles both ',' (from BED merge) and '|' (from prior aggregation) as
    input delimiters, and always outputs '|' as the bioinformatics convention.
    """
    split = x.str.split(r"[,|]", regex=True)
    split = split.explode()
    return "|".join(np.unique(split))


def _compute_gene_offsets(df: pd.DataFrame) -> dict[str, int]:
    """Compute per-region offsets for UTR sizes within a single gene group.

    Args:
        df: DataFrame subset for one gene, with columns start, end, strand, utr_name.

    Returns:
        Dict mapping utr_name to its positional offset within the gene.
    """
    strand = df["strand"].iloc[0]
    if strand == "-":
        df = df.copy()
        df[["start", "end"]] = df[["end", "start"]]
        df = df.sort_values(by=["start", "end"], ascending=False)
    else:
        df = df.sort_values(by=["start", "end"])

    # Compute offset per utr_name within this gene
    seen_utrs = {}
    start = df["start"].iloc[0]
    end = df["end"].iloc[0]
    left_over = 0

    for row in df.itertuples():
        utr_name = row.utr_name
        if utr_name in seen_utrs:
            continue

        if strand == "+":
            if row.start == start:
                seen_utrs[utr_name] = left_over
                if row.end > end:
                    end = row.end
            elif row.start <= end:
                seen_utrs[utr_name] = left_over + (row.start - start)
                if row.end > end:
                    end = row.end
            elif row.start > end:
                left_over += end - start
                seen_utrs[utr_name] = left_over
                start = row.start
                end = row.end
        elif strand == "-":
            if row.start == start:
                seen_utrs[utr_name] = left_over
                if row.end < end:
                    end = row.end
            elif row.start >= end:
                seen_utrs[utr_name] = left_over + (start - row.start)
                if row.end < end:
                    end = row.end
            elif row.start < end:
                left_over += start - end
                seen_utrs[utr_name] = left_over
                start = row.start
                end = row.end

    return seen_utrs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def merge_samples_utr_sizes(
    utr_sizes: list[pd.DataFrame], method: str = "intersection"
) -> pd.DataFrame:
    """Merge flat-format DataFrames from multiple samples.

    If method='intersection', only UTRs present in all samples are kept.
    If method='union', all UTRs from all samples are kept.

    Batch labeling (the "-N" suffix on observation_ids) is handled at read time
    via Protocol.get_observation_id(), not here.

    Args:
        utr_sizes: List of flat DataFrames (utr_name, observation_id, utr_size),
            one per sample.
        method: 'intersection' or 'union'.

    Returns:
        Merged flat DataFrame with re-categorized string columns.
    """
    logger = module_log.module_logger()
    logger.info("Merging all batches' UTR sizes")
    if method == "union":
        merged = pd.concat(utr_sizes, ignore_index=True)
    elif method == "intersection":
        # Keep only UTRs present in ALL samples
        common_utrs = set(utr_sizes[0]["utr_name"].unique())
        for batch in utr_sizes[1:]:
            common_utrs &= set(batch["utr_name"].unique())
        merged = pd.concat(utr_sizes, ignore_index=True)
        merged = merged[merged["utr_name"].isin(common_utrs)].copy()
    else:
        raise AttributeError(f"Method {method} not supported.")

    # Re-categorize after concat
    merged["utr_name"] = pd.Categorical(merged["utr_name"])
    merged["observation_id"] = pd.Categorical(merged["observation_id"])

    return merged


def merge_utrs_per_gene(
    utr_regions: pd.DataFrame, read_mappings: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge per-UTR regions and read mappings up to gene level.

    For each gene, computes positional offsets so that UTR sizes from different
    UTR regions of the same gene are placed on a single coordinate axis.

    Args:
        utr_regions: UTR reference DataFrame indexed by utr_name.
        read_mappings: Flat DataFrame of (utr_name, observation_id, utr_size) rows.

    Returns:
        A (gene_regions, gene_read_mappings) tuple where utr_name has been
        replaced with the gene name and utr_size is offset-adjusted.
    """
    logger = module_log.module_logger()

    # Join read_mappings with utr_regions on utr_name
    # read_mappings is flat: utr_name, observation_id, utr_size
    # utr_regions is indexed by utr_name
    combined = read_mappings.merge(
        utr_regions.reset_index(), on="utr_name", how="inner"
    )
    combined["gene_name"] = combined["gene_name"].fillna(value="None")

    # Build aggregated utr_regions per gene
    utr_regions_sorted = utr_regions.copy()
    utr_regions_sorted["gene_name"] = utr_regions_sorted["gene_name"].fillna("None")
    utr_regions_sorted = utr_regions_sorted.sort_values(by=["start", "end"])
    utr_regions_sorted["utr_name"] = utr_regions_sorted.index

    grouped_regions = utr_regions_sorted.groupby("name")
    utr_regions_per_gene = grouped_regions.agg(
        {
            "chrom": "first",
            "start": _join_unique,
            "end": _join_unique,
            "strand": "first",
            "transcript_id": _split_and_then_join_unique,
            "gene_name": _split_and_then_join_unique,
        }
    )
    utr_regions_per_gene.index.name = "utr_name"
    utr_regions_per_gene["name"] = utr_regions_per_gene.index

    # Single pass: compute offsets, utr→gene mapping, and sort order
    offset_map = {}
    utr_name_to_gene = {}
    utr_sort_order = {}
    for gene_name, group in utr_regions_sorted.groupby("name"):
        # Offsets
        offsets = _compute_gene_offsets(group)
        offset_map.update(offsets)

        # utr_name → gene_name mapping
        for un in group["utr_name"].values:
            utr_name_to_gene[un] = gene_name

        # Sort order for reads within this gene
        strand = group["strand"].iloc[0]
        if strand == "-":
            g = group.copy()
            g[["start", "end"]] = g[["end", "start"]]
            g = g.sort_values(by=["start", "end"], ascending=False)
        else:
            g = group
        for pos, utr_name in enumerate(g["utr_name"].values):
            utr_sort_order[utr_name] = pos

    # Apply offsets to the flat read_mappings.
    # Cast utr_name out of Categorical before .map(): pandas >= 2.1 preserves
    # Categorical dtype when every category is mapped injectively, which then
    # makes .fillna(0) raise because 0 is not a valid category.
    utr_name_obj = combined["utr_name"].astype(object)
    combined["offset"] = utr_name_obj.map(offset_map).fillna(0).astype(np.int32)
    combined["utr_size"] = combined["utr_size"] + combined["offset"]

    # Add sort key for ordering reads within each gene
    combined["_sort_order"] = utr_name_obj.map(utr_sort_order)
    combined["_gene"] = utr_name_obj.map(utr_name_to_gene)

    # Sort by gene and then by the per-gene sort order
    combined = combined.sort_values(by=["_gene", "_sort_order"])

    # Replace utr_name with gene name
    combined["utr_name"] = combined["_gene"]

    # Build flat result
    result = combined[["utr_name", "observation_id", "utr_size"]].copy()
    result["utr_name"] = pd.Categorical(result["utr_name"])
    result["observation_id"] = pd.Categorical(result["observation_id"])
    result["utr_size"] = result["utr_size"].astype(np.int32)
    result = result.reset_index(drop=True)

    return utr_regions_per_gene, result
