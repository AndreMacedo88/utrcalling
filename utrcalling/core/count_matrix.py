"""
count_matrix.py - sparse count matrix building from UTR size data.
"""

from __future__ import annotations

import warnings
from typing import NamedTuple, TYPE_CHECKING

import numpy as np
import pandas as pd
import anndata
import joblib
from scipy.sparse import coo_matrix, csr_matrix

from utrcalling.core import module_log, helpers
from utrcalling.core.helpers import JOBLIB_DEFAULT_BACKEND
from utrcalling.core.protocol import Protocol

if TYPE_CHECKING:
    from collections.abc import Callable
    import numpy.typing as npt


class CountMatrixResult(NamedTuple):
    """Result of generate_counts_and_anno()."""

    count_matrix: csr_matrix
    peaks_var: pd.DataFrame
    peaks_obs: pd.DataFrame | None = None


warnings.filterwarnings("ignore", category=FutureWarning)


def _process_utr_group_batch(
    batch: list[tuple[int, str, npt.NDArray[np.int32], npt.NDArray[np.int32]]],
    func: Callable[[npt.ArrayLike], npt.ArrayLike] | None = None,
) -> list[tuple[int, list[npt.NDArray[np.int32]], list[npt.NDArray[np.int32]], list[str], list[int]]]:
    """Process a batch of UTR groups in a single worker task.

    Module-level so loky can pickle it. Batching amortizes joblib dispatch
    overhead across many UTRs — the per-task work otherwise dominated by
    pickling rather than the actual size-transform compute.

    Args:
        batch: List of (orig_idx, utr_name, sizes_int32, obs_idx_int32) tuples.
            ``orig_idx`` is the UTR's position in the original (pre-sort) group
            list — preserved so callers can re-sort to byte-stable column order.
            ``obs_idx_int32`` is the per-row sparse-matrix row index, replacing
            the previous string-keyed obs-to-row dict so workers receive only
            small int arrays (no large dict pickled per task).
        func: Optional size transform (e.g., KDE peak caller). None = identity.

    Returns:
        A list of (orig_idx, rows_list, data_list, peak_names, peak_sizes), one
        entry per input UTR.
    """
    out = []
    for orig_idx, utr_name, sizes, obs_idx in batch:
        sort_idx = np.argsort(sizes)
        sizes = sizes[sort_idx]
        obs_idx_sorted = obs_idx[sort_idx]

        if func is not None:
            peaks = func(sizes.reshape(-1, 1)).flatten()
        else:
            peaks = sizes

        rows_list: list[npt.NDArray[np.int32]] = []
        data_list: list[npt.NDArray[np.int32]] = []
        peak_names: list[str] = []
        peak_sizes: list[int] = []

        for peak_val in np.unique(peaks):
            mask = peaks == peak_val
            obs_in_peak = obs_idx_sorted[mask]
            obs_unique, counts = np.unique(obs_in_peak, return_counts=True)
            rows_list.append(obs_unique.astype(np.int32, copy=False))
            data_list.append(counts.astype(np.int32))
            peak_names.append(f"{utr_name}-{int(peak_val)}")
            peak_sizes.append(int(peak_val))

        out.append((orig_idx, rows_list, data_list, peak_names, peak_sizes))
    return out


# Threshold for parallelizing UTR processing — avoids joblib overhead on small datasets
_PARALLEL_UTR_THRESHOLD = 50


def generate_counts_and_anno(
    utr_sizes: pd.DataFrame,
    protocol: Protocol,
    size_transform: Callable[[npt.ArrayLike], npt.ArrayLike] | None = None,
    n_jobs: int | None = None,
) -> CountMatrixResult:
    """Transform UTR sizes per group and build a sparse count matrix.

    Groups molecules by UTR, optionally transforms each UTR's size distribution
    via ``size_transform`` (e.g. KDE peak calling), and assembles a sparse
    (observations x peaks) count matrix with associated var and obs DataFrames.
    Parallelizes across UTRs with joblib when the number of UTRs exceeds the
    threshold; per-UTR work is bundled into ``n_workers * 4`` batches and
    BLAS/OpenMP threads are pinned to one per worker to avoid oversubscription
    from the KDE inside ``size_transform``.

    Args:
        utr_sizes: Flat DataFrame with columns (utr_name, observation_id, utr_size).
        protocol: A Protocol instance with annotations and
            barcode_column_name attributes.
        size_transform: Optional callable to transform UTR sizes per group.
            When None, raw sizes are used directly (identity).
        n_jobs: Worker process count. ``None`` falls back to
            :data:`utrcalling.core.helpers.JOBLIB_DEFAULT_N_JOBS`. Non-positive
            values resolve to ``get_number_cores(left_out=1)``.

    Returns:
        A CountMatrixResult(count_matrix, peaks_var, peaks_obs) namedtuple.
        peaks_obs is None when barcode annotations are not available.
    """
    logger = module_log.module_logger()

    annotations = protocol.annotations
    func = size_transform

    logger.info("Generating and counting UTR sizes")

    # Build observation vocabulary — deterministic alphabetical row ordering.
    # `observation_id` is already pd.Categorical, so `.unique()` returns a
    # Categorical whose code-order is not lexicographic. Cast to a string ndarray
    # before sorting to lock ordering down independent of category dtype origin.
    obs_ids_sorted = np.sort(np.asarray(utr_sizes["observation_id"].astype(str).unique()))
    n_obs = len(obs_ids_sorted)

    # Pre-encode observation_id to int32 row indices ONCE. Workers receive only
    # int arrays — no per-task pickling of an ~n_obs-entry dict, no Python-level
    # dict lookups in the hot inner loop.
    obs_cat = pd.Categorical(
        utr_sizes["observation_id"].astype(str), categories=obs_ids_sorted
    )
    obs_idx_arr = obs_cat.codes.astype(np.int32, copy=False)
    if (obs_idx_arr < 0).any():
        raise ValueError("Found unmapped observation_id (NaN?) in utr_sizes")
    utr_sizes = utr_sizes.assign(_obs_idx=obs_idx_arr)

    # Pre-extract groups with their original index. Keeping orig_idx lets us
    # re-sort batched results back to category order so peak column ordering
    # (and the `var` row ordering it controls) stays byte-stable vs serial.
    groups: list[tuple[int, str, npt.NDArray[np.int32], npt.NDArray[np.int32]]] = [
        (
            i,
            utr_name,
            group["utr_size"].values.astype(np.int32, copy=False),
            group["_obs_idx"].values.astype(np.int32, copy=False),
        )
        for i, (utr_name, group) in enumerate(
            utr_sizes.groupby("utr_name", observed=True)
        )
    ]
    n_utrs = len(groups)

    logger.info("Processing %d UTRs", n_utrs)

    if n_utrs > _PARALLEL_UTR_THRESHOLD:
        # Resolve worker count: explicit n_jobs, else JOBLIB_DEFAULT_N_JOBS.
        # Non-positive resolves to all-but-one (cgroup/affinity-aware).
        n_jobs_resolved = helpers.resolve_n_jobs(n_jobs)
        n_workers = (
            n_jobs_resolved if n_jobs_resolved > 0
            else max(1, helpers.get_number_cores(left_out=1))
        )

        # Sort UTRs descending by molecule count, then round-robin into
        # n_workers*4 batches. loky doesn't work-steal mid-task, so without
        # this static load balance the long-tail UTRs cluster onto a few
        # workers and stall the rest.
        groups.sort(key=lambda g: -len(g[2]))
        n_batches = max(1, min(n_workers * 4, n_utrs))
        batches = [groups[i::n_batches] for i in range(n_batches)]

        logger.info(
            "Parallelizing across UTRs (%d workers, %d batches)", n_workers, n_batches
        )

        # inner_max_num_threads=1 prevents BLAS/OpenMP inside sklearn's KDE
        # from spawning extra threads per worker — without it 27 workers ×
        # ~8 BLAS threads = 200+ threads competing for cores on a 27-core box.
        # joblib requires `backend` here when passing inner_max_num_threads.
        try:
            with joblib.parallel_config(
                backend=JOBLIB_DEFAULT_BACKEND, inner_max_num_threads=1
            ):
                batch_results = list(
                    joblib.Parallel(n_jobs=n_workers, return_as="generator")(
                        joblib.delayed(_process_utr_group_batch)(b, func)
                        for b in batches
                    )
                )
        finally:
            helpers.shutdown_joblib_pool()

        # Flatten and re-sort to original (category) order so peak columns
        # come out in the same order as the serial path.
        flat = [item for br in batch_results for item in br]
        flat.sort(key=lambda x: x[0])
    else:
        flat = _process_utr_group_batch(groups, func)

    # Assemble sparse triplets in original UTR order
    all_rows, all_cols, all_data = [], [], []
    peak_names = []
    peak_utr_sizes = []
    col_offset = 0

    for _orig_idx, rows_list, data_list, names_list, sizes_list in flat:
        for rows, data, name, size in zip(rows_list, data_list, names_list, sizes_list):
            all_rows.append(rows)
            all_cols.append(np.full(len(rows), col_offset, dtype=np.int32))
            all_data.append(data)
            peak_names.append(name)
            peak_utr_sizes.append(size)
            col_offset += 1

    logger.info("Finished generating and counting UTR sizes")

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
        peaks_obs = peaks_obs.reindex(index=obs_ids_sorted)
        return CountMatrixResult(X, peaks_var, peaks_obs)
    else:
        # No per-observation annotation columns, but still carry the observation
        # identities as the obs index. Without this, AnnData falls back to
        # positional integer names ('0', '1', ...) and the sample/cell identity
        # (e.g. a TruSeq sample_id) is lost.
        peaks_obs = pd.DataFrame(index=pd.Index(obs_ids_sorted, name="observation_id"))
        return CountMatrixResult(X, peaks_var, peaks_obs)


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
    # Split on the *last* hyphen to handle UTR names that contain hyphens
    # e.g., "RP11-123B3.1-42" → utr_name="RP11-123B3.1", peak_size="42"
    peak_var_new["utr_name"] = peak_var_new.index.str.replace(
        r"-(\d+)$", "", regex=True
    )

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
