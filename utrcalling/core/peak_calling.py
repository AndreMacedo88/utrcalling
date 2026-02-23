"""
peak_calling.py - module with functions to deal with calling peaks
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import anndata
import joblib
from scipy.sparse import coo_matrix, csr_matrix

from utrcalling.core import module_log

warnings.filterwarnings("ignore", category=FutureWarning)


def _process_utr_group(
    utr_name: str,
    sizes: np.ndarray,
    obs: np.ndarray,
    obs_to_idx: dict[str, int],
    func: object,
) -> tuple[list[np.ndarray], list[np.ndarray], list[str], list[int]]:
    """Process a single UTR group: call peaks and compute sparse triplets.

    This is a module-level function so it can be pickled by joblib workers.

    Args:
        utr_name: Name of the UTR being processed.
        sizes: Array of UTR sizes for this group.
        obs: Array of observation IDs for this group.
        obs_to_idx: Dict mapping observation ID to sparse matrix row index.
        func: Peak calling function (e.g., KDE-based).

    Returns:
        A tuple of (rows_list, data_list, peak_names, peak_sizes) where
        rows_list and data_list are parallel lists of arrays (one per peak),
        and peak_names/peak_sizes are parallel lists of scalars.
    """
    sort_idx = np.argsort(sizes)
    sizes = sizes[sort_idx]
    obs = obs[sort_idx]

    peaks = func(sizes.reshape(-1, 1))
    peaks = peaks.flatten()

    rows_list = []
    data_list = []
    peak_names = []
    peak_sizes = []

    for peak_val in np.unique(peaks):
        mask = peaks == peak_val
        obs_in_peak = obs[mask]
        obs_unique, counts = np.unique(obs_in_peak, return_counts=True)
        rows = np.array([obs_to_idx[o] for o in obs_unique], dtype=np.int32)

        rows_list.append(rows)
        data_list.append(counts.astype(np.int32))
        peak_names.append(f"{utr_name}-{int(peak_val)}")
        peak_sizes.append(int(peak_val))

    return rows_list, data_list, peak_names, peak_sizes


# Threshold for parallelizing peak calling — avoids joblib overhead on small datasets
_PARALLEL_UTR_THRESHOLD = 50


def generate_peak_counts_and_anno(
    utr_sizes: pd.DataFrame,
    protocol: object,
) -> tuple[csr_matrix, pd.DataFrame] | tuple[csr_matrix, pd.DataFrame, pd.DataFrame]:
    """Call consensus peaks per UTR and build a sparse count matrix.

    Groups molecules by UTR, runs the protocol's peak calling function on each
    UTR's size distribution, and assembles a sparse (observations x peaks) count
    matrix with associated var and obs DataFrames. Parallelizes across UTRs
    with joblib when the number of UTRs exceeds the threshold.

    Args:
        utr_sizes: Flat DataFrame with columns (utr_name, observation_id, utr_size).
        protocol: A Protocol instance with peak_calling, annotations, and
            barcode_column_name attributes.

    Returns:
        A tuple of (sparse_matrix, peaks_var) or (sparse_matrix, peaks_var, peaks_obs)
        when barcode annotations are available.
    """
    logger = module_log.module_logger()

    annotations = protocol.annotations
    func = protocol.peak_calling

    logger.info("Generating and counting UTR size peaks")

    # Build observation vocabulary — deterministic alphabetical row ordering
    obs_ids = np.sort(utr_sizes["observation_id"].unique())
    obs_to_idx = {obs: i for i, obs in enumerate(obs_ids)}
    n_obs = len(obs_ids)

    # Pre-extract groups as (name, sizes_array, obs_array) tuples
    groups = [
        (utr_name, group["utr_size"].values, group["observation_id"].values)
        for utr_name, group in utr_sizes.groupby("utr_name", observed=True)
    ]
    n_utrs = len(groups)

    logger.info("Calling peaks for %d UTRs", n_utrs)

    if n_utrs > _PARALLEL_UTR_THRESHOLD:
        logger.info("Parallelizing peak calling across UTRs")
        results = joblib.Parallel(n_jobs=-2, backend="loky")(
            joblib.delayed(_process_utr_group)(name, sizes, obs, obs_to_idx, func)
            for name, sizes, obs in groups
        )
    else:
        results = [
            _process_utr_group(name, sizes, obs, obs_to_idx, func)
            for name, sizes, obs in groups
        ]

    # Assemble sparse triplets from all worker results
    all_rows, all_cols, all_data = [], [], []
    peak_names = []
    peak_utr_sizes = []
    col_offset = 0

    for rows_list, data_list, names_list, sizes_list in results:
        for rows, data, name, size in zip(rows_list, data_list, names_list, sizes_list):
            all_rows.append(rows)
            all_cols.append(np.full(len(rows), col_offset, dtype=np.int32))
            all_data.append(data)
            peak_names.append(name)
            peak_utr_sizes.append(size)
            col_offset += 1

    logger.info("Finished generating and counting UTR size peaks")

    # Build sparse matrix once
    if all_rows:
        rows = np.concatenate(all_rows)
        cols = np.concatenate(all_cols)
        data = np.concatenate(all_data)
    else:
        rows = np.array([], dtype=np.int32)
        cols = np.array([], dtype=np.int32)
        data = np.array([], dtype=np.int32)

    X = csr_matrix(coo_matrix((data, (rows, cols)), shape=(n_obs, col_offset)))

    # Build var
    peaks_var = pd.DataFrame({"utr_size": peak_utr_sizes}, index=peak_names)

    logger.info("Adding annotations")
    if protocol.barcode_column_name:
        peaks_obs = annotations.set_index(protocol.barcode_column_name)
        peaks_obs = peaks_obs.reindex(index=obs_ids)
        return X, peaks_var, peaks_obs
    else:
        return X, peaks_var


def add_to_peaks_var(peak_var: pd.DataFrame, annotations: pd.DataFrame) -> pd.DataFrame:
    """Join UTR reference annotations onto the peaks var DataFrame.

    Splits each peak name (e.g., 'GENE-100') to recover the source utr_name,
    then left-joins the reference annotations.

    Args:
        peak_var: Peaks variable DataFrame indexed by peak names.
        annotations: UTR reference DataFrame indexed by utr_name.

    Returns:
        An enriched peaks var DataFrame with reference columns appended.
    """
    logger = module_log.module_logger()
    logger.info("Adding reference annotations to the final result")

    annotations["utr_name"] = annotations.index

    peak_var_new = peak_var.copy(deep=True)
    peak_var_new["utr_name"] = peak_var_new.index.str.split(pat=r"-").str[0]

    peak_var_new = peak_var_new.join(
        annotations, on="utr_name", how="left", rsuffix="_right"
    )

    peak_var_new = peak_var_new.drop("utr_name_right", axis=1)

    return peak_var_new


def generate_count_matrix_anndata(
    count_matrix: csr_matrix, var: pd.DataFrame, obs: pd.DataFrame | None = None
) -> anndata.AnnData:
    """Wrap a sparse count matrix and metadata into an AnnData object.

    Args:
        count_matrix: Sparse (observations x peaks) count matrix.
        var: Peaks variable DataFrame (one row per peak).
        obs: Optional observations DataFrame (one row per cell/sample).

    Returns:
        An AnnData object ready for downstream analysis.
    """
    logger = module_log.module_logger()
    logger.info("Generating Count Matrix")

    count_matrix_anndata = anndata.AnnData(X=count_matrix, var=var, obs=obs)
    return count_matrix_anndata
