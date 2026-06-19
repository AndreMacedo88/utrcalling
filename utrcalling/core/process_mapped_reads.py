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

    # Single pass over the small reference (~tens of thousands of UTRs):
    # per-UTR offset, utr_name -> gene-name, and per-gene read sort order.
    offset_map: dict[str, int] = {}
    utr_name_to_gene: dict[str, str] = {}
    utr_sort_order: dict[str, int] = {}
    for gene_name, group in utr_regions_sorted.groupby("name"):
        offset_map.update(_compute_gene_offsets(group))
        for un in group["utr_name"].values:
            utr_name_to_gene[un] = gene_name

        strand = group["strand"].iloc[0]
        if strand == "-":
            g = group.copy()
            g[["start", "end"]] = g[["end", "start"]]
            g = g.sort_values(by=["start", "end"], ascending=False)
        else:
            g = group
        for pos, un in enumerate(g["utr_name"].values):
            utr_sort_order[un] = pos

    # Relabel utr_name -> gene and apply per-UTR offsets on the (potentially
    # huge) flat read table WITHOUT a wide merge or a string-keyed global sort.
    #
    # The previous implementation `.merge()`d the full reference onto every read
    # row — pulling in transcript_id/gene_name string columns the result never
    # uses — and then `.sort_values()` on string columns across the whole table.
    # On dozens of bulk RNA-seq samples (~1e9 rows) that inflated an ~9 GB table
    # past 120 GB and triggered the OOM killer. Here we map everything through
    # the small Categorical code space and order rows with an integer `lexsort`,
    # keeping only int arrays the size of the read table in memory. The output
    # (rows ordered by gene name, then by genomic position within the gene) is
    # identical to the old behaviour.
    utr_cat = pd.Categorical(read_mappings["utr_name"])
    categories = np.asarray(utr_cat.categories)
    codes = utr_cat.codes.astype(np.int32, copy=False)
    n_cat = len(categories)

    # Per-category lookups (small: one entry per reference UTR).
    in_ref = np.fromiter(
        (c in utr_name_to_gene for c in categories), dtype=bool, count=n_cat
    )
    offset_by_cat = np.fromiter(
        (offset_map.get(c, 0) for c in categories), dtype=np.int32, count=n_cat
    )
    sortorder_by_cat = np.fromiter(
        (utr_sort_order.get(c, 0) for c in categories), dtype=np.int32, count=n_cat
    )
    gene_of_cat = np.array(
        [utr_name_to_gene.get(c, None) for c in categories], dtype=object
    )

    # Gene categories = sorted unique gene names, matching the categories that
    # `pd.Categorical(<gene names>)` produced in the old implementation. Sorting
    # rows by the resulting integer gene code is therefore equivalent to the old
    # string sort on the gene name.
    gene_categories = np.unique(gene_of_cat[in_ref].astype(str))
    gene_code_of = {g: i for i, g in enumerate(gene_categories)}
    utrcat_to_genecode = np.full(n_cat, -1, dtype=np.int32)
    for i in range(n_cat):
        if in_ref[i]:
            utrcat_to_genecode[i] = gene_code_of[str(gene_of_cat[i])]

    # Inner-join semantics: drop reads whose utr_name is missing (code < 0) or
    # absent from the reference.
    safe_codes = np.where(codes >= 0, codes, 0)
    keep = (codes >= 0) & in_ref[safe_codes]
    kept_codes = safe_codes[keep]
    del codes, safe_codes

    gene_codes = utrcat_to_genecode[kept_codes]
    new_sizes = (
        read_mappings["utr_size"].to_numpy()[keep] + offset_by_cat[kept_codes]
    ).astype(np.int32)
    obs_kept = pd.Categorical(read_mappings["observation_id"])[keep]

    # Order rows by gene, then by per-gene genomic position (stable lexsort on
    # integer keys — primary key is the last argument).
    order = np.lexsort((sortorder_by_cat[kept_codes], gene_codes))

    result = pd.DataFrame(
        {
            "utr_name": pd.Categorical.from_codes(
                gene_codes[order], categories=gene_categories
            ),
            "observation_id": obs_kept[order],
            "utr_size": new_sizes[order],
        }
    )
    result["observation_id"] = pd.Categorical(result["observation_id"])
    result = result.reset_index(drop=True)

    return utr_regions_per_gene, result
