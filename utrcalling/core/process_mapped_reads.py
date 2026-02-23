from __future__ import annotations

from functools import reduce
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
from rapidfuzz import process, distance

from utrcalling.core import helpers, module_log


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

    def join_unique(x):
        return ",".join(x.unique().astype(str))

    def split_and_then_join_unique(x):
        split = x.str.split(",")
        split = split.explode()
        return ",".join(np.unique(split))

    def transform_utr_sizes_flat(df):
        """Compute per-region offsets and add them to utr_size for a gene group."""
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
            "start": join_unique,
            "end": join_unique,
            "strand": "first",
            "transcript_id": split_and_then_join_unique,
            "gene_name": split_and_then_join_unique,
        }
    )
    utr_regions_per_gene.index.name = "utr_name"
    utr_regions_per_gene["name"] = utr_regions_per_gene.index

    # Compute per-gene offsets for each utr_name
    offset_map = {}
    for gene_name, group in utr_regions_sorted.groupby("name"):
        offsets = transform_utr_sizes_flat(group)
        offset_map.update(offsets)

    # Build utr_name → gene_name mapping and utr_name → sort order for reads
    utr_to_gene = utr_regions_sorted.groupby("name").apply(
        lambda g: g["utr_name"].tolist()
    )
    utr_name_to_gene = {}
    for gene_name, utr_names_list in utr_to_gene.items():
        for un in utr_names_list:
            utr_name_to_gene[un] = gene_name

    # For each gene, compute the sort-order position for each utr_name.
    # For negative strand genes, we need the reverse of the start/end sorted order.
    utr_sort_order = {}
    for gene_name, group in utr_regions_sorted.groupby("name"):
        strand = group["strand"].iloc[0]
        if strand == "-":
            # Swap start/end and sort descending (same as old transform_utr_sizes)
            g = group.copy()
            g[["start", "end"]] = g[["end", "start"]]
            g = g.sort_values(by=["start", "end"], ascending=False)
        else:
            g = group
        for pos, utr_name in enumerate(g["utr_name"].values):
            utr_sort_order[utr_name] = pos

    # Apply offsets to the flat read_mappings
    combined["offset"] = combined["utr_name"].map(offset_map).fillna(0).astype(np.int32)
    combined["utr_size"] = combined["utr_size"] + combined["offset"]

    # Add sort key for ordering reads within each gene
    combined["_sort_order"] = combined["utr_name"].map(utr_sort_order)
    combined["_gene"] = combined["utr_name"].map(utr_name_to_gene)

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
    logger = module_log.module_logger()
    logger.info("Subsetting UTR reference data")

    utr_names = utr_sizes["utr_name"].unique()
    sub_utr_reference = utr_reference_regions[
        utr_reference_regions.index.isin(utr_names)
    ]
    sub_utr_reference_unique = sub_utr_reference.drop_duplicates()

    return sub_utr_reference_unique


def filter_barcodes(utr_sizes: pd.DataFrame, whitelist: np.ndarray) -> pd.DataFrame:
    """Filter observations to those matching a barcode whitelist.

    Exact matches are kept directly. For non-matching barcodes, a fuzzy match
    (Hamming distance <= 1) is attempted via rapidfuzz. Matched barcodes are
    replaced with their whitelist counterpart.

    Args:
        utr_sizes: Flat DataFrame with an 'observation_id' column.
        whitelist: Array of allowed barcode strings.

    Returns:
        Filtered DataFrame with corrected observation_ids and re-categorized dtypes.
    """
    logger = module_log.module_logger()
    logger.info("Filtering cell barcodes")

    barcodes_to_test = set(utr_sizes["observation_id"].unique())
    barcodes_correct = set()
    barcodes_replace = {}
    for barcode in barcodes_to_test:
        if barcode in whitelist:
            barcodes_correct.add(barcode)
        else:
            try:
                match = process.extract(
                    barcode, whitelist, scorer=distance.Hamming.distance, score_cutoff=1
                )
            except ValueError as err:
                logger.error(
                    f"\nError matching sample barcode/ID: {barcode}\nPlease check if "
                    f"the barcodes in the annotation file are correct or, if using "
                    f"different batches with similar barcodes/IDs, if all the batches "
                    f"are correctly described in the annotations file."
                )
                logger.error(f"Barcodes from the annotation file:\n{whitelist}")
                raise err
            if match:
                barcode_correct = [bc[0] for bc in match]
                for bc in barcode_correct:
                    try:  # check if the batch is the same between barcodes
                        if barcode.split("-")[1] == bc.split("-")[1]:
                            barcodes_correct.add(bc)
                            barcodes_replace[barcode] = bc
                        else:
                            continue
                    except IndexError:  # if working without batches, no need to check
                        barcodes_correct.add(bc)
                        barcodes_replace[barcode] = bc
                        break
            else:
                continue

    # Apply replacements and filter — flat format, no explode/regroup needed
    if barcodes_replace:
        # Need a copy to mutate observation_id before filtering
        result = utr_sizes.copy()
        result["observation_id"] = (
            result["observation_id"].astype(str).replace(barcodes_replace).astype("category")
        )
    else:
        result = utr_sizes

    result = result[result["observation_id"].isin(barcodes_correct)].copy()
    result["observation_id"] = pd.Categorical(result["observation_id"])
    result["utr_name"] = pd.Categorical(result["utr_name"])

    return result
