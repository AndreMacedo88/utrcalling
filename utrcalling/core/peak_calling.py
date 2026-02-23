"""
peak_calling.py - module with functions to deal with calling peaks
"""

import warnings

import numpy as np
import pandas as pd
import anndata
from scipy.sparse import coo_matrix, csr_matrix

from utrcalling.core import module_log

warnings.filterwarnings("ignore", category=FutureWarning)


def generate_peak_counts_and_anno(
    utr_sizes,
    protocol,
):
    logger = module_log.module_logger()

    annotations = protocol.annotations
    func = protocol.peak_calling

    logger.info("Generating and counting UTR size peaks")

    # Build observation vocabulary — deterministic alphabetical row ordering
    obs_ids = np.sort(utr_sizes["observation_id"].unique())
    obs_to_idx = {obs: i for i, obs in enumerate(obs_ids)}
    n_obs = len(obs_ids)

    # Accumulate sparse triplets
    all_rows, all_cols, all_data = [], [], []
    peak_names = []
    peak_utr_sizes = []
    col_offset = 0

    logger.info("Calling peaks for each UTR")
    for utr_name, group in utr_sizes.groupby("utr_name", observed=True):
        sizes = group["utr_size"].values
        obs = group["observation_id"].values
        sort_idx = np.argsort(sizes)
        sizes = sizes[sort_idx]
        obs = obs[sort_idx]

        peaks = func(sizes.reshape(-1, 1))
        peaks = peaks.flatten()

        for peak_val in np.unique(peaks):
            mask = peaks == peak_val
            obs_in_peak = obs[mask]
            obs_unique, counts = np.unique(obs_in_peak, return_counts=True)
            rows = np.array([obs_to_idx[o] for o in obs_unique], dtype=np.int32)

            all_rows.append(rows)
            all_cols.append(np.full(len(rows), col_offset, dtype=np.int32))
            all_data.append(counts.astype(np.int32))

            peak_names.append(f"{utr_name}-{int(peak_val)}")
            peak_utr_sizes.append(int(peak_val))
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


def add_to_peaks_var(peak_var, annotations):
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


def generate_count_matrix_anndata(count_matrix, var, obs=None):
    logger = module_log.module_logger()
    logger.info("Generating Count Matrix")

    count_matrix_anndata = anndata.AnnData(X=count_matrix, var=var, obs=obs)
    return count_matrix_anndata
