"""Peak calling functions operating on AnnData objects.

Provides secondary peak calling on existing AnnData objects, including KDE-based
peak consolidation, PAS distance scoring, and bandwidth estimation.
"""

from __future__ import annotations

import math
from typing import Any

import regex
import numpy as np
import numpy.typing as npt
import pandas as pd
import anndata as ad
import scipy.sparse
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema

import pybedtools
import joblib
from tqdm import tqdm

from utrcalling.core import module_log, helpers
from utrcalling.core.helpers import JOBLIB_DEFAULT_BACKEND


def _resolve_workers(n_jobs: int | None) -> int:
    """Resolve a positive worker count from an n_jobs argument.

    Mirrors the policy used in count_matrix.py: explicit positive value wins;
    otherwise fall back to the project default; non-positive results map to
    ``get_number_cores(left_out=1)``. Returns a strictly positive int suitable
    for both ``Parallel(n_jobs=...)`` and the batch-count formula.
    """
    n_jobs_resolved = helpers.resolve_n_jobs(n_jobs)
    return (
        n_jobs_resolved if n_jobs_resolved > 0
        else max(1, helpers.get_number_cores(left_out=1))
    )


# ---------------------------------------------------------------------------
# Shared per-UTR peak calling logic (module-level for joblib picklability)
# ---------------------------------------------------------------------------


def _peak_call_single_utr(
    adata_utr: ad.AnnData, method: str, bandwidth: float
) -> tuple[npt.NDArray[Any], pd.DataFrame]:
    """Run peak calling on a single UTR slice of an AnnData.

    Extracts gene name and UTR numbering, builds a size vector from read
    counts, calls peaks via the specified method, and regroups the count
    matrix according to the new peak assignments.

    Args:
        adata_utr: AnnData subset for one UTR (all peaks within that UTR).
        method: Peak calling method (currently only ``"kde"``).
        bandwidth: KDE bandwidth parameter.

    Returns:
        A tuple of ``(count_matrix, var_df)`` where *count_matrix* is a 2-D
        numpy array (obs x new_peaks) and *var_df* is the corresponding
        ``.var`` DataFrame.
    """
    gene = adata_utr.var.gene_name.iloc[0]
    utr_name_parts = adata_utr.var.utr_name.iloc[0].split("_")
    if len(utr_name_parts) > 1:
        utr_number = f"UTR{utr_name_parts[1]}_"
    else:
        utr_number = ""
    utr_var = adata_utr.var.iloc[0]

    # generate a vector of one UTR size per UMI count
    umi_utr_sizes = _utr_sizes_vector_from_read_counts(
        adata_utr.X, adata_utr.var.utr_size.astype(int)
    )

    if method.lower() == "kde":
        peaks = _call_peaks_using_kernel_density_estimator(
            umi_utr_sizes.values, bandwidth=bandwidth
        )
        peak_sizes = pd.Series(data=peaks, index=umi_utr_sizes.index)
        peak_sizes = peak_sizes[~peak_sizes.index.duplicated()]
        peak_sizes_unique = peak_sizes.unique()

    # create new matrix with the sum of UMIs for each new peak
    mtx_new_t = adata_utr[:, peak_sizes.index.values].to_df().T
    mtx_new_t["sizes"] = peak_sizes.tolist()
    mtx_new = mtx_new_t.groupby("sizes").sum().T

    # create new polyA names
    polya_names = [
        f"{gene}_{utr_number}pA{x}" for x in range(1, len(mtx_new.columns) + 1)
    ]

    # generate new utr_var
    utr_var_new = pd.DataFrame([utr_var] * len(mtx_new.columns))
    utr_var_new.index = polya_names
    utr_var_new["utr_size"] = peak_sizes_unique

    return mtx_new.to_numpy(), utr_var_new


# ---------------------------------------------------------------------------
# Compact per-UTR workers — operate on (X_sub, var_sub) tuples instead of
# AnnData slices, so workers receive only the bytes they need rather than the
# full AnnData object pickled per batch. With ~18k UTRs and 26 workers the
# previous "ship full adata per batch" approach hit OOM during worker spawn
# (26 simultaneous unpickles of a multi-hundred-MB adata). Pre-extracting in
# the parent reduces per-batch wire size by ~100x.
# ---------------------------------------------------------------------------


def _peak_call_single_utr_compact(
    X_sub: npt.NDArray[Any] | scipy.sparse.spmatrix,
    var_sub: pd.DataFrame,
    method: str,
    bandwidth: float,
) -> tuple[npt.NDArray[Any] | scipy.sparse.spmatrix, pd.DataFrame]:
    """Per-UTR peak-calling on raw ``(X, var)`` slices.

    Output type tracks the input: sparse in -> sparse out, dense in -> dense
    out. Preserving sparsity is critical for memory: the caller accumulates
    ``mtx_list`` across thousands of UTRs and concatenates at the end.
    Materializing each per-UTR result as dense (n_obs * n_peaks) blows up to
    multiple GB in aggregate even when each UTR is tiny.

    The peak-summation step (replacing the AnnData
    ``[:, peak_sizes].to_df().T -> groupby(sum) -> .T`` chain) uses a sparse
    aggregator matrix where ``aggregator[col, peak] = 1`` iff that input
    column belongs to that peak. Then ``X_sub_subset @ aggregator`` produces
    an (n_obs x n_peaks) result in one shot, sparse or dense as appropriate.
    """
    gene = var_sub.gene_name.iloc[0]
    utr_name_parts = var_sub.utr_name.iloc[0].split("_")
    utr_number = f"UTR{utr_name_parts[1]}_" if len(utr_name_parts) > 1 else ""
    utr_var = var_sub.iloc[0]

    umi_utr_sizes = _utr_sizes_vector_from_read_counts(
        X_sub, var_sub.utr_size.astype(int)
    )

    if method.lower() == "kde":
        peaks = _call_peaks_using_kernel_density_estimator(
            umi_utr_sizes.values, bandwidth=bandwidth
        )
        peak_sizes = pd.Series(data=peaks, index=umi_utr_sizes.index)
        peak_sizes = peak_sizes[~peak_sizes.index.duplicated()]
        peak_sizes_unique = peak_sizes.unique()

    cols_pos = var_sub.index.get_indexer(peak_sizes.index)
    sizes_arr = np.asarray(peak_sizes.values)
    unique_sizes_sorted = np.sort(np.unique(sizes_arr))

    n_in = len(cols_pos)
    n_out = len(unique_sizes_sorted)

    # Build the col-to-peak aggregator matrix (one 1 per input column).
    size_to_col = {int(s): k for k, s in enumerate(unique_sizes_sorted)}
    agg_rows = np.arange(n_in, dtype=np.int32)
    agg_cols = np.array(
        [size_to_col[int(s)] for s in sizes_arr], dtype=np.int32
    )
    agg_data = np.ones(n_in, dtype=np.int32)
    agg = scipy.sparse.csc_matrix(
        (agg_data, (agg_rows, agg_cols)), shape=(n_in, n_out)
    )

    if scipy.sparse.issparse(X_sub):
        X_peaks = X_sub[:, cols_pos]
        new_mtx = (X_peaks @ agg).tocsc()
    else:
        X_peaks = np.asarray(X_sub)[:, cols_pos]
        new_mtx = X_peaks @ agg.toarray()

    polya_names = [
        f"{gene}_{utr_number}pA{x}" for x in range(1, n_out + 1)
    ]
    utr_var_new = pd.DataFrame([utr_var] * n_out)
    utr_var_new.index = polya_names
    # Match original behavior: utr_size column uses peak_sizes.unique() (insertion
    # order), which equals sorted order in practice when var is utr_size-sorted.
    utr_var_new["utr_size"] = peak_sizes_unique

    return new_mtx, utr_var_new


def _get_distances_per_utr_compact(
    X_sub: npt.NDArray[Any] | scipy.sparse.spmatrix,
    var_sub: pd.DataFrame,
    hamming: int,
    genome: Any,
) -> npt.NDArray[Any] | None:
    """Compute PAS distances for one UTR, given its raw ``(X, var)`` slices."""
    pas_end, _, seq_strand = _find_pas(var_sub, hamming=hamming, genome=genome)

    if not pas_end:
        return None

    umi_utr_sizes = _utr_sizes_vector_from_read_counts(X_sub, var_sub.utr_size)
    distances = _calculate_umi_distance_to_pas(
        umi_utr_sizes=umi_utr_sizes, pas_end=pas_end, seq_strand=seq_strand
    )
    return distances.to_numpy().flatten()


def _get_score_distances_per_utr_compact(
    X_sub: npt.NDArray[Any] | scipy.sparse.spmatrix,
    var_sub: pd.DataFrame,
    hamming: int,
) -> list[float | tuple[float, npt.NDArray[Any]]] | None:
    """Score PAS distances for one UTR, given its raw ``(X, var)`` slices."""
    pas_end, _, seq_strand = _find_pas(var_sub, hamming=hamming)

    if not pas_end:
        return None

    umi_utr_sizes = _utr_sizes_vector_from_read_counts(X_sub, var_sub.utr_size)
    distances = _calculate_umi_distance_to_pas(
        umi_utr_sizes=umi_utr_sizes, pas_end=pas_end, seq_strand=seq_strand
    )

    distances_ = distances.to_numpy().flatten()

    total_umis = len(distances)
    threshold = int(math.ceil(total_umis * 0.01))
    distances_bigger_ten = distances[distances.groupby(level=0).count() > threshold]
    in_range_of_pas = distances_bigger_ten.between(10, 30)

    try:
        return [in_range_of_pas.value_counts().loc[True] / total_umis, distances_]
    except KeyError:
        return [0, distances_]


# ---------------------------------------------------------------------------
# Batch workers — receive a list of pre-extracted per-UTR slices. The parent
# materializes the slices once and ships only the slices each batch needs;
# workers never see the full AnnData. ``orig_idx`` lets the parent re-sort
# flattened results back to deterministic UTR order.
# ---------------------------------------------------------------------------

CompactUtrItem = tuple[int, str, Any, pd.DataFrame]


def _peak_call_utr_batch(
    batch: list[CompactUtrItem], method: str, bandwidth: float
) -> list[tuple[int, npt.NDArray[Any], pd.DataFrame]]:
    out = []
    for orig_idx, _utr_name, X_sub, var_sub in batch:
        mtx, var_df = _peak_call_single_utr_compact(
            X_sub, var_sub, method, bandwidth
        )
        out.append((orig_idx, mtx, var_df))
    return out


def _pas_distance_utr_batch(
    batch: list[CompactUtrItem], hamming: int, genome: Any
) -> list[tuple[int, npt.NDArray[Any] | None]]:
    out = []
    for orig_idx, _utr_name, X_sub, var_sub in batch:
        result = _get_distances_per_utr_compact(X_sub, var_sub, hamming, genome)
        out.append((orig_idx, result))
    return out


def _score_distances_utr_batch(
    batch: list[CompactUtrItem], hamming: int
) -> list[tuple[int, list[float | tuple[float, npt.NDArray[Any]]] | None]]:
    out = []
    for orig_idx, _utr_name, X_sub, var_sub in batch:
        result = _get_score_distances_per_utr_compact(X_sub, var_sub, hamming)
        out.append((orig_idx, result))
    return out


def _extract_per_utr_inputs(adata: ad.AnnData) -> list[CompactUtrItem]:
    """Pre-extract per-UTR ``(orig_idx, utr_name, X_sub, var_sub)`` slices.

    The parent runs this once before dispatching; each slice is small (only
    the columns belonging to that UTR), so each batch task ships ~1/N_UTRs of
    the original X plus the matching var rows — a fraction of one full adata.

    CSR is converted to CSC once before the loop — CSR column slicing is
    O(nnz) per call (it scans the whole matrix), which would balloon the
    extraction to O(N_UTRs * nnz) when called per-UTR. CSC column slicing is
    O(nnz_in_slice), so the whole extraction drops to O(nnz) total. On real
    inputs (~18k UTRs over a 70k-peak matrix) this is the difference between
    multiple minutes and seconds.
    """
    var_groups = adata.var.groupby("utr_name", observed=True).indices
    utrs_in_order = adata.var.utr_name.unique()
    if scipy.sparse.issparse(adata.X):
        X_for_slicing = adata.X.tocsc()
    else:
        X_for_slicing = adata.X
    out: list[CompactUtrItem] = []
    for i, utr in enumerate(utrs_in_order):
        cols = var_groups[utr]
        X_sub = X_for_slicing[:, cols]
        var_sub = adata.var.iloc[cols]
        out.append((i, utr, X_sub, var_sub))
    return out


def _build_compact_batches(
    per_utr_inputs: list[CompactUtrItem], n_workers: int
) -> list[list[CompactUtrItem]]:
    """Split per-UTR inputs into ``n_workers * 4`` round-robin batches."""
    n_batches = max(1, min(n_workers * 4, len(per_utr_inputs)))
    return [per_utr_inputs[i::n_batches] for i in range(n_batches)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def peak_call_anndata_parallel(
    adata: ad.AnnData,
    method: str,
    bandwidth: float = 1,
    n_jobs: int | None = None,
) -> ad.AnnData:
    """Call peaks on an AnnData using joblib parallelism across UTRs.

    Each UTR's peaks are consolidated in parallel via
    :func:`_peak_call_single_utr`, then results are merged into a new
    AnnData. UTRs are bundled into ``n_workers * 4`` batches before dispatch
    to amortize joblib pickling overhead.

    Args:
        adata: AnnData with UMI counts in ``.X`` and UTR metadata in
            ``.var`` (must contain ``utr_name``, ``gene_name``, ``utr_size``).
        method: Peak calling method (currently only ``"kde"``).
        bandwidth: KDE bandwidth. ``0`` means no peak calling is performed.
        n_jobs: Worker process count. ``None`` falls back to
            JOBLIB_DEFAULT_N_JOBS; non-positive resolves to all-but-one core.

    Returns:
        A new AnnData with peaks consolidated.
    """
    logger = module_log.module_logger()

    if bandwidth == 0:
        logger.notice("Using Bandwidth = 0 means no changes were made")
        return adata

    sparse_adata = scipy.sparse.issparse(adata.X)

    n_workers = _resolve_workers(n_jobs)
    per_utr_inputs = _extract_per_utr_inputs(adata)
    batches = _build_compact_batches(per_utr_inputs, n_workers)

    logger.info(
        "Calling peaks across %d UTRs (%d workers, %d batches)",
        len(per_utr_inputs), n_workers, len(batches),
    )

    try:
        with joblib.parallel_config(
            backend=JOBLIB_DEFAULT_BACKEND, inner_max_num_threads=1
        ):
            batch_results = list(
                joblib.Parallel(n_jobs=n_workers, return_as="generator")(
                    joblib.delayed(_peak_call_utr_batch)(batch, method, bandwidth)
                    for batch in tqdm(batches, colour="#ffb700", ascii=" ■")
                )
            )
    finally:
        helpers.shutdown_joblib_pool()

    flat = [item for br in batch_results for item in br]
    flat.sort(key=lambda x: x[0])
    mtx_list = [item[1] for item in flat]
    var_list = [item[2] for item in flat]

    var_df = pd.concat(var_list, axis=0).drop("pA", axis=1, errors="ignore")
    # Stay sparse end-to-end when input was sparse — np.hstack on dense per-UTR
    # results was the OOM trigger at 100% (intermediate dense matrix ~ n_obs *
    # total_peaks * 4 bytes, then doubled during AnnData's re-sparsify step).
    if sparse_adata:
        X_combined = scipy.sparse.hstack(mtx_list, format="csr")
    else:
        X_combined = np.hstack(mtx_list)
    new_adata = ad.AnnData(X=X_combined, obs=adata.obs, var=var_df)

    # `.copy()` materializes the slice so callers receive an actual AnnData;
    # otherwise downstream `.var` mutations (e.g. anndata's auto-categorical
    # conversion on disk write) trigger ImplicitModificationWarning per column.
    new_adata = new_adata[new_adata.obs.sort_index().index, :].copy()
    return new_adata


def peak_call_anndata(
    adata: ad.AnnData,
    method: str,
    bandwidth: float = 1,
    min_umis: int | None = None,
) -> ad.AnnData:
    """Call peaks on an AnnData, processing UTRs sequentially.

    Iterates over each UTR, optionally skipping those below a minimum UMI
    count, and consolidates peaks using :func:`_peak_call_single_utr`.
    Sparse input is preserved (output is re-sparsified if the input was
    sparse).

    Args:
        adata: AnnData with UMI counts in ``.X`` and UTR metadata in
            ``.var`` (must contain ``utr_name``, ``gene_name``, ``utr_size``).
        method: Peak calling method (currently only ``"kde"``).
        bandwidth: KDE bandwidth. ``0`` means no peak calling is performed.
        min_umis: If set, UTRs with fewer total UMIs are skipped.

    Returns:
        A new AnnData with peaks consolidated.
    """
    logger = module_log.module_logger()

    if bandwidth == 0:
        logger.notice("Using Bandwidth = 0 means no changes were made")
        return adata

    sparse_adata = scipy.sparse.issparse(adata.X)

    new_obs = adata.obs
    utrs = adata.var.utr_name.unique()

    mtx_list = []
    var_list = []

    logger.info("Calling peaks for each UTR")
    for utr in tqdm(utrs, colour="#ffb700", ascii=" ■", desc="UTR: "):
        adata_utr = adata[:, adata.var.utr_name == utr]

        # Only process utr if it has a minimum number of UMIs, if provided
        if min_umis:
            total_umis = adata_utr.X.sum()
            if total_umis < min_umis:
                logger.info(
                    f"Skipping UTR {utr} with only {total_umis} UMIs (< {min_umis})"
                )
                continue

        mtx, var = _peak_call_single_utr(adata_utr, method=method, bandwidth=bandwidth)
        mtx_list.append(mtx)
        var_list.append(var)

    var_df = pd.concat(var_list, axis=0).drop("pA", axis=1, errors="ignore")
    new_adata = ad.AnnData(X=np.hstack(mtx_list), obs=new_obs, var=var_df)

    if sparse_adata:
        logger.info("Sparsifying data as initially provided.")
        new_adata.X = scipy.sparse.csr_matrix(new_adata.X)

    new_adata = new_adata[new_adata.obs.sort_index().index, :].copy()
    return new_adata


def remove_peaks_below_count(
    adata: ad.AnnData,
    threshold: float = 0.1,
    minimum_counts: int = 10,
) -> ad.AnnData:
    """Remove UTR peaks whose total counts fall below a threshold.

    For each UTR the threshold is ``max(total_utr_counts * threshold,
    minimum_counts)``.  Peaks below this are dropped.

    Args:
        adata: The AnnData object to filter.
        threshold: Fractional threshold applied to each UTR's total count
            (e.g. ``0.1`` means 10%).
        minimum_counts: Absolute minimum count a peak must reach.

    Returns:
        A filtered AnnData with low-count peaks removed.
    """
    utr_names = adata.var.loc[:, "utr_name"].values

    # Sparse-safe: sum columns directly instead of materializing dense matrix
    col_sums = np.asarray(adata.X.sum(axis=0)).ravel()
    utr_total_counts = pd.Series(col_sums).groupby(utr_names, observed=True).sum()
    utrs_thresholds = utr_total_counts.apply(
        lambda x: max(x * threshold, minimum_counts)
    )

    thresholds = adata.var.join(utrs_thresholds.to_frame("thresh"), on="utr_name")[
        "thresh"
    ].to_numpy()

    bigger_than_threshold = col_sums >= thresholds

    x_ = adata.X[:, bigger_than_threshold]
    var_ = adata.var[bigger_than_threshold]

    return ad.AnnData(X=x_, var=var_, obs=adata.obs)


def estimate_best_bandwidth(
    adata: ad.AnnData,
    method: str = "kde",
    hamming: int = 1,
    bandwidths: tuple[float, ...] | list[float] = (0.5, 1, 1.5, 2, 2.5, 3),
    return_distances_to_hex: bool = False,
    n_jobs: int | None = None,
) -> list[float] | list[tuple[float, npt.NDArray[Any]]]:
    """Estimate the best KDE bandwidth by scoring PAS proximity.

    Runs peak calling at each candidate bandwidth, then scores each result
    by how many UMIs fall within 10-30 nt of a PAS hexamer.

    The bandwidths are evaluated sequentially; parallelism is consumed by the
    inner per-UTR loops in :func:`peak_call_anndata_parallel` and
    :func:`_score_peak_assignment` so all cores stay productive without
    nesting joblib pools.

    Args:
        adata: AnnData with UMI counts and UTR metadata.
        method: Peak calling method (currently only ``"kde"``).
        hamming: Maximum Hamming distance for PAS matching.
        bandwidths: Candidate bandwidth values to evaluate.
        return_distances_to_hex: If ``True``, also return per-UMI distance
            arrays alongside scores.
        n_jobs: Worker process count for the per-UTR loops.

    Returns:
        A list of scores (one per bandwidth), or a list of
        ``(score, distances_array)`` tuples if *return_distances_to_hex*
        is ``True``.
    """
    if isinstance(bandwidths, (int, float)):
        bandwidths = [bandwidths]

    estimation_results: list[float | tuple[float, npt.NDArray[Any]]] = []
    for bandwidth in bandwidths:
        adata_bandwidth = peak_call_anndata_parallel(
            adata=adata, method=method, bandwidth=bandwidth, n_jobs=n_jobs
        )
        score = _score_peak_assignment(
            adata=adata_bandwidth,
            hamming=hamming,
            return_distances_to_hex=return_distances_to_hex,
            n_jobs=n_jobs,
        )
        estimation_results.append(score)
    return estimation_results


def pas_distance(
    adata: ad.AnnData,
    genome: str,
    hamming: int = 1,
    n_jobs: int | None = None,
) -> npt.NDArray[Any]:
    """Calculate UMI distances to the nearest polyadenylation signal (PAS).

    For each UTR, finds PAS hexamer sites (AATAAA with up to *hamming*
    mismatches) in the genome sequence and computes the distance of each
    UMI's cleavage site to the nearest PAS.

    Args:
        adata: AnnData with UMI counts per UTR, UTR sizes in
            ``.var.utr_size``, and UTR names in ``.var.utr_name``.
        genome: Path to a genome FASTA file.
        hamming: Maximum Hamming distance for fuzzy PAS matching.
        n_jobs: Worker process count for the per-UTR parallel loop.

    Returns:
        1-D array of distances (one per UMI across all UTRs with a PAS).
    """
    logger = module_log.module_logger()

    n_workers = _resolve_workers(n_jobs)
    per_utr_inputs = _extract_per_utr_inputs(adata)
    batches = _build_compact_batches(per_utr_inputs, n_workers)

    logger.info(
        "Computing PAS distances across %d UTRs (%d workers, %d batches)",
        len(per_utr_inputs), n_workers, len(batches),
    )

    try:
        with joblib.parallel_config(
            backend=JOBLIB_DEFAULT_BACKEND, inner_max_num_threads=1
        ):
            batch_results = list(
                joblib.Parallel(n_jobs=n_workers, return_as="generator")(
                    joblib.delayed(_pas_distance_utr_batch)(batch, hamming, genome)
                    for batch in tqdm(batches, colour="#ffb700", ascii=" ■")
                )
            )
    finally:
        helpers.shutdown_joblib_pool()

    flat = [item for br in batch_results for item in br]
    flat.sort(key=lambda x: x[0])
    distances_list = [arr for _idx, arr in flat if arr is not None]
    distances_array = np.concatenate(distances_list)

    return distances_array


def _score_peak_assignment(
    adata: ad.AnnData,
    hamming: int = 1,
    return_distances_to_hex: bool = False,
    n_jobs: int | None = None,
) -> float | tuple[float, npt.NDArray[Any]]:
    """Score peak assignments by proximity to PAS hexamers.

    Args:
        adata: AnnData with peak-called UTR counts.
        hamming: Maximum Hamming distance for PAS matching.
        return_distances_to_hex: If ``True``, also return the raw distance
            array.
        n_jobs: Worker process count for the per-UTR parallel loop.

    Returns:
        A score (fraction of UMIs within 10-30 nt of a PAS), or a
        ``(score, distances_array)`` tuple.
    """
    logger = module_log.module_logger()
    n_workers = _resolve_workers(n_jobs)
    per_utr_inputs = _extract_per_utr_inputs(adata)
    batches = _build_compact_batches(per_utr_inputs, n_workers)

    logger.info(
        "Scoring peaks across %d UTRs (%d workers, %d batches)",
        len(per_utr_inputs), n_workers, len(batches),
    )

    try:
        with joblib.parallel_config(
            backend=JOBLIB_DEFAULT_BACKEND, inner_max_num_threads=1
        ):
            batch_results = list(
                joblib.Parallel(n_jobs=n_workers, return_as="generator")(
                    joblib.delayed(_score_distances_utr_batch)(batch, hamming)
                    for batch in tqdm(batches, colour="#ffb700", ascii=" ■", desc="UTR: ")
                )
            )
    finally:
        helpers.shutdown_joblib_pool()

    flat = [item for br in batch_results for item in br]
    flat.sort(key=lambda x: x[0])

    scores = []
    distances_list = []
    for _idx, result in flat:
        if result is None:
            continue
        scores.append(result[0])
        distances_list.append(result[1])

    score = sum(scores) / len(per_utr_inputs)
    if return_distances_to_hex:
        distances_array = np.concatenate(distances_list)
        return (score, distances_array)
    else:
        return score


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_pas(
    var: pd.DataFrame, genome: Any, hamming: int = 1
) -> tuple[list[int], list[str], str]:
    """Find polyadenylation signal (PAS) sites in a UTR sequence.

    Extracts the genomic sequence described in *var* (one UTR's metadata
    rows) and searches for AATAAA motifs (with up to *hamming* mismatches).

    Args:
        var: Per-UTR var rows with chrom/start/end/strand columns. All rows
            describe the same UTR, so only the first is consulted.
        genome: Path to a genome FASTA file (or a pybedtools BedTool).
        hamming: Maximum Hamming distance for fuzzy matching.

    Returns:
        A tuple ``(pas_end, pas_seq, strand)`` where *pas_end* is a list
        of 0-based end positions, *pas_seq* the matched sequences, and
        *strand* the genomic strand.
    """
    seq_chrom = var.chrom.iloc[0]
    seq_start = var.start.iloc[0] - 1
    seq_stop = var.end.iloc[0]
    seq_strand = var.strand.iloc[0]

    query = pybedtools.BedTool(
        f"{seq_chrom} {seq_start} {seq_stop} three_prime_utr 0 {seq_strand}",
        from_string=True,
    )
    query = query.sequence(fi=genome, s=True)

    with open(query.seqfn) as q:
        next(q)
        utr_seq = q.read().strip()

    # Compile regex once before the search loop
    if hamming > 0:
        pas_pattern = regex.compile(rf"(AATAAA){{s<={hamming}}}")
    elif hamming == 0:
        pas_pattern = regex.compile(r"(AATAAA)")
    else:
        raise ValueError(f"Hamming distance of {hamming} not supported")

    pas_seq: list[str] = []
    pas_end: list[int] = []
    start_cut = 0
    while True:
        search = pas_pattern.search(utr_seq)
        try:
            span = search.span()
        except AttributeError:
            break
        end = span[1] + start_cut
        start_cut += span[0] + 1
        pas_end.append(end)
        pas_seq.append(search.group(0))
        utr_seq = utr_seq[span[0] + 1 :]

    return (pas_end, pas_seq, seq_strand)


def _calculate_umi_distance_to_pas(
    umi_utr_sizes: pd.Series,
    pas_end: list[int],
    seq_strand: str,
) -> pd.Series:
    """Calculate the distance from each UMI to the nearest PAS site.

    Args:
        umi_utr_sizes: Series of UTR sizes (one per UMI).
        pas_end: List of PAS end positions within the UTR.
        seq_strand: Genomic strand (``"+"`` or ``"-"``).

    Returns:
        A Series of signed distances to the chosen PAS site.
    """
    if seq_strand not in ("+", "-"):
        raise ValueError(f"Strand {seq_strand} not supported")

    sign = 1 if seq_strand == "+" else -1

    if len(pas_end) == 1:
        return (umi_utr_sizes - pas_end[0]) * sign

    # Build distances matrix for all PAS sites
    distances_list = [(umi_utr_sizes - end) * sign for end in pas_end]
    distances = pd.concat(distances_list, axis=1, ignore_index=True)

    # Vectorised selection of closest PAS
    dist_arr = distances.values
    in_range = (dist_arr >= 10) & (dist_arr <= 30)
    n_in_range = in_range.sum(axis=1)

    abs_dist = np.abs(dist_arr)

    # Default: absolute minimum distance
    chosen = abs_dist.argmin(axis=1)

    # Where exactly one is in range: pick that one
    single_mask = n_in_range == 1
    if single_mask.any():
        chosen[single_mask] = np.argmax(in_range[single_mask], axis=1)

    # Where multiple are in range: pick the closest (min abs) among those in range
    multi_mask = n_in_range > 1
    if multi_mask.any():
        masked = np.where(in_range[multi_mask], abs_dist[multi_mask], np.inf)
        chosen[multi_mask] = masked.argmin(axis=1)

    return pd.Series(
        dist_arr[np.arange(len(dist_arr)), chosen],
        index=distances.index,
    )


def _utr_sizes_vector_from_read_counts(
    counts_arr: npt.NDArray[Any] | scipy.sparse.spmatrix,
    sizes_arr: pd.Series,
) -> pd.Series:
    """Expand a sizes vector by repeating each size according to its count.

    Args:
        counts_arr: 2-D count matrix (obs x var), dense or sparse.
        sizes_arr: Series of UTR sizes indexed by var names.

    Returns:
        A pandas Series with each size repeated by the column-sum of its
        counts, preserving the original index (var names).
    """
    if scipy.sparse.issparse(counts_arr):
        counts_arr = counts_arr.toarray()
    counts_sum = np.asarray(np.sum(counts_arr, axis=0)).ravel().astype(int)
    return sizes_arr.repeat(counts_sum)


def _call_peaks_using_kernel_density_estimator(
    utr_sizes_vector: npt.NDArray[Any], bandwidth: float = 1
) -> npt.NDArray[np.int32]:
    """Assign each UMI to a consensus peak via Gaussian KDE.

    Fits a KDE to the distribution of UTR sizes and uses local extrema to
    partition molecules into peaks.

    Args:
        utr_sizes_vector: 1-D or 2-D array of UTR sizes to cluster.
        bandwidth: KDE bandwidth parameter.

    Returns:
        1-D array of peak assignments (one per input molecule), where each
        value is the UTR size of the assigned peak centre.
    """
    if len(utr_sizes_vector) <= 3:
        peaks = np.repeat(round(np.median(utr_sizes_vector)), len(utr_sizes_vector))
        return peaks

    if np.ndim(utr_sizes_vector) == 1:
        utr_sizes_vector = utr_sizes_vector.reshape(-1, 1)

    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(utr_sizes_vector)

    # calculate an appropriate window to check the model
    minimum_size, maximum_size = utr_sizes_vector.min(), utr_sizes_vector.max()
    padding = max((maximum_size - minimum_size) * 0.1, 10)
    window = np.arange(round(minimum_size - padding), round(maximum_size + padding) + 1)
    # get the model predictions for that window. Specifically, the extrema.
    evaluation = kde.score_samples(window.reshape(-1, 1))
    minima = argrelextrema(evaluation, np.less_equal, order=1)[0][1:]
    maxima = argrelextrema(evaluation, np.greater_equal, order=1)[0]
    # Note: [1:] because less_equal always outputs index 0,
    # since padding guarantees that kde rises until first value.

    minima = [minimum for minimum in minima if minimum - 1 not in minima]
    maxima = [maximum for maximum in maxima if maximum - 1 not in maxima]

    if minima[0] < maxima[0]:
        raise RuntimeError(
            "First minima is larger than the first maxima. "
            "Peak calling using this method is incorrect for this UTR."
        )

    # create a container to hold the UTR peak sizes
    peaks = np.zeros(len(utr_sizes_vector), dtype=np.int32)

    # attribute a peak to each cell according to their UTR sizes and the kde model:
    # first iterate over the function' minima
    last_min_idx = len(minima) - 1
    for i, val in enumerate(window[minima]):
        # get the maximum upstream that minimum (the peak) and store it
        try:
            max_ = round(window[maxima][i])
        except IndexError:
            continue
        for j, cond in enumerate(utr_sizes_vector < val):
            if cond and peaks[j] == 0:
                peaks[j] = max_

    # get the last maximum, just downstream the last minimum.
    try:
        max_ = round(window[maxima][last_min_idx + 1])
    except IndexError:  # for those cases with just one peak, there will be no minimum
        # so we take the median as the peak
        max_ = round(np.median(utr_sizes_vector))

    # place the last maximum in those molecules that haven't yet had their
    # peaks scored
    peaks[peaks == 0] = max_
    return peaks
