"""Barcode whitelist filtering with memory-aware chunking."""
from __future__ import annotations

import numpy as np
import pandas as pd
import psutil
from rapidfuzz import process, distance

from utrcalling.core import module_log


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
        result = utr_sizes.copy()
        # Map through category codes to avoid full str conversion
        obs = result["observation_id"]
        if isinstance(obs.dtype, pd.CategoricalDtype):
            codes = obs.cat.codes.values
            old_cats = obs.cat.categories.values
            # Build a new category array with replacements applied
            new_cats = np.array(
                [barcodes_replace.get(c, c) for c in old_cats], dtype=object
            )
            new_values = pd.Categorical(new_cats[codes])
            result["observation_id"] = new_values
        else:
            result["observation_id"] = (
                result["observation_id"].astype(str).replace(barcodes_replace).astype("category")
            )
    else:
        result = utr_sizes

    result = result[result["observation_id"].isin(barcodes_correct)].copy()
    result["observation_id"] = pd.Categorical(result["observation_id"])
    result["utr_name"] = pd.Categorical(result["utr_name"])

    return result


def filter_barcodes_memory_aware(
    utr_sizes: pd.DataFrame, whitelist: np.ndarray
) -> pd.DataFrame:
    """Apply barcode filtering with automatic memory-aware chunking.

    Estimates whether barcode filtering will fit in available RAM.
    If not, splits the data into chunks, filters each chunk, and
    re-concatenates. Otherwise, filters in a single pass.

    Args:
        utr_sizes: Flat DataFrame with an 'observation_id' column.
        whitelist: Array of allowed barcode strings.

    Returns:
        Filtered DataFrame with corrected observation_ids and re-categorized dtypes.
    """
    logger = module_log.module_logger()

    # Estimate whether barcode filtering will fit in available RAM.
    # filter_barcodes() copies the DataFrame and does string operations,
    # so peak usage is roughly 3x the data size.
    data_bytes = utr_sizes.memory_usage(deep=True).sum()
    available_bytes = psutil.virtual_memory().available
    headroom = data_bytes * 3

    if headroom > available_bytes:
        n_chunks = max(2, int(np.ceil(headroom / available_bytes)))
        logger.info(
            "Splitting barcode filtering into %d chunks to fit in available memory",
            n_chunks,
        )
        num = utr_sizes.shape[0]
        idx_splits = np.cumsum(
            [num // n_chunks + (1 if x < num % n_chunks else 0) for x in range(n_chunks)]
        )
        idx_splits = np.insert(idx_splits, 0, 0)

        split_generator = (
            utr_sizes.iloc[idx_splits[n] : idx_splits[n + 1]]
            for n in range(n_chunks)
        )

        filtered_list = [
            filter_barcodes(pos, whitelist=whitelist)
            for pos in split_generator
        ]

        result = pd.concat(filtered_list, ignore_index=True)
        # Re-categorize after concat to merge category codes
        result["utr_name"] = pd.Categorical(result["utr_name"])
        result["observation_id"] = pd.Categorical(result["observation_id"])
        return result
    else:
        return filter_barcodes(utr_sizes, whitelist=whitelist)
