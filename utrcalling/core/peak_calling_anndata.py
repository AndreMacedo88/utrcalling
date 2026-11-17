"""Peak calling functions operating on AnnData objects.

Provides secondary peak calling on existing AnnData objects, including KDE-based
peak consolidation, PAS distance scoring, and the scores of peak-calling
settings (see peak_calling.scores).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any, cast

import anndata as ad
import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
import regex
import scipy.sparse
from tqdm import tqdm

from utrcalling.core import anndata_processing, helpers, module_log, provenance
from utrcalling.core import genome as genome_
from utrcalling.core.helpers import JOBLIB_DEFAULT_BACKEND
from utrcalling.core.peak_calling.bandwidth_rules import utr_bandwidth
from utrcalling.core.peak_calling.callers import SizeTransform, get_caller
from utrcalling.core.peak_calling.kde import call_peaks_using_kernel_density_estimator
from utrcalling.core.peak_calling.scores import FREQUENT_SITE_SHARE, ScoreContext, make_score
from utrcalling.core.peak_calling.settings import (
    CANDIDATE_BANDWIDTHS,
    DEFAULT_SCORES,
    Bandwidth,
    check_kernel,
    parse_bandwidth,
)

# ---------------------------------------------------------------------------
# Shared per-UTR peak calling logic (module-level for joblib picklability)
# ---------------------------------------------------------------------------


def _peak_call_single_utr(
    adata_utr: ad.AnnData,
    bandwidth: Bandwidth,
    kernel: str = "gaussian",
    caller: SizeTransform | None = None,
) -> tuple[npt.NDArray[Any], pd.DataFrame]:
    """Run peak calling on a single UTR slice of an AnnData.

    Builds a size vector from read counts, calls peaks, and regroups the count
    matrix according to the new peak assignments.

    Args:
        adata_utr: AnnData subset for one UTR (all peaks within that UTR).
        bandwidth: KDE bandwidth: a number of nt, or a rule.
        kernel: The KDE's kernel.
        caller: The peak caller (see peak_calling.callers); by default the KDE
            with *bandwidth* and *kernel*.

    Returns:
        A tuple of ``(count_matrix, var_df)`` where *count_matrix* is a 2-D
        numpy array (obs x new_peaks) and *var_df* is the corresponding
        ``.var`` DataFrame, with the UTR's ``bandwidth`` if a rule set it.
    """
    utr_var = adata_utr.var.iloc[0]

    # generate a vector of one UTR size per UMI count
    umi_utr_sizes = _utr_sizes_vector_from_read_counts(
        adata_utr.X, adata_utr.var.utr_size.astype(int)
    )

    peaks = _caller(bandwidth, kernel, caller)(umi_utr_sizes.values)
    peak_sizes = pd.Series(data=peaks, index=umi_utr_sizes.index)
    peak_sizes = peak_sizes[~peak_sizes.index.duplicated()]

    # create new matrix with the sum of UMIs for each new peak
    mtx_new_t = adata_utr[:, peak_sizes.index.values].to_df().T
    mtx_new_t["sizes"] = peak_sizes.tolist()
    mtx_new = mtx_new_t.groupby("sizes").sum().T

    # generate new utr_var; the caller names the peaks of all UTRs together
    utr_var_new = pd.DataFrame([utr_var] * len(mtx_new.columns))
    # The counts are summed per peak in size order: label the peaks in that order.
    # int64, as in the count matrices of run and call-peaks.
    utr_var_new["utr_size"] = np.sort(peak_sizes.unique()).astype(np.int64)
    if isinstance(bandwidth, str):
        utr_var_new["bandwidth"] = utr_bandwidth(umi_utr_sizes.to_numpy(), bandwidth)

    return mtx_new.to_numpy(), utr_var_new


def _caller(bandwidth: Bandwidth, kernel: str, caller: SizeTransform | None) -> SizeTransform:
    """*caller*, or the KDE with *bandwidth* and *kernel*."""
    if caller is not None:
        return caller
    return partial(call_peaks_using_kernel_density_estimator, bandwidth=bandwidth, kernel=kernel)


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
    bandwidth: Bandwidth,
    kernel: str = "gaussian",
    caller: SizeTransform | None = None,
) -> tuple[npt.NDArray[Any] | scipy.sparse.spmatrix, pd.DataFrame]:
    """Per-UTR peak-calling on raw ``(X, var)`` slices.

    Calls the peaks with *caller* (see peak_calling.callers), by default the
    KDE with *bandwidth* and *kernel*.

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
    utr_var = var_sub.iloc[0]

    umi_utr_sizes = _utr_sizes_vector_from_read_counts(X_sub, var_sub.utr_size.astype(int))

    peaks = _caller(bandwidth, kernel, caller)(umi_utr_sizes.values)
    peak_sizes = pd.Series(data=peaks, index=umi_utr_sizes.index)
    peak_sizes = peak_sizes[~peak_sizes.index.duplicated()]

    cols_pos = var_sub.index.get_indexer(peak_sizes.index)
    sizes_arr = np.asarray(peak_sizes.values)
    unique_sizes_sorted = np.sort(np.unique(sizes_arr))

    n_in = len(cols_pos)
    n_out = len(unique_sizes_sorted)

    # Build the col-to-peak aggregator matrix (one 1 per input column).
    size_to_col = {int(s): k for k, s in enumerate(unique_sizes_sorted)}
    agg_rows = np.arange(n_in, dtype=np.int32)
    agg_cols = np.array([size_to_col[int(s)] for s in sizes_arr], dtype=np.int32)
    agg_data = np.ones(n_in, dtype=np.int32)
    agg = scipy.sparse.csc_matrix((agg_data, (agg_rows, agg_cols)), shape=(n_in, n_out))

    if scipy.sparse.issparse(X_sub):
        X_peaks = X_sub[:, cols_pos]
        new_mtx = (X_peaks @ agg).tocsc()
    else:
        X_peaks = np.asarray(X_sub)[:, cols_pos]
        new_mtx = X_peaks @ agg.toarray()

    # the caller names the peaks of all UTRs together
    utr_var_new = pd.DataFrame([utr_var] * n_out)
    # The counts are summed per peak in size order: label the peaks in that order.
    # int64, as in the count matrices of run and call-peaks.
    utr_var_new["utr_size"] = np.sort(peak_sizes.unique()).astype(np.int64)
    if isinstance(bandwidth, str):  # the bandwidth the rule gives this UTR
        utr_var_new["bandwidth"] = utr_bandwidth(umi_utr_sizes.to_numpy(), bandwidth)

    return new_mtx, utr_var_new


def _get_distances_per_utr_compact(
    X_sub: npt.NDArray[Any] | scipy.sparse.spmatrix,
    var_sub: pd.DataFrame,
    hamming: int,
    genome: genome_.GenomeFasta,
) -> npt.NDArray[Any] | None:
    """Compute PAS distances for one UTR, given its raw ``(X, var)`` slices."""
    pas_end, _, _ = _find_pas(var_sub, hamming=hamming, genome=genome)

    if not pas_end:
        return None

    umi_utr_sizes = _utr_sizes_vector_from_read_counts(X_sub, var_sub.utr_size)
    distances = _calculate_umi_distance_to_pas(umi_utr_sizes=umi_utr_sizes, pas_end=pas_end)
    return distances.to_numpy().flatten()


def _get_score_distances_per_utr_compact(
    X_sub: npt.NDArray[Any] | scipy.sparse.spmatrix,
    var_sub: pd.DataFrame,
    hamming: int,
    genome: genome_.GenomeFasta,
) -> list[float | tuple[float, npt.NDArray[Any]]] | None:
    """Score PAS distances for one UTR, given its raw ``(X, var)`` slices."""
    pas_end, _, _ = _find_pas(var_sub, hamming=hamming, genome=genome)

    if not pas_end:
        return None

    umi_utr_sizes = _utr_sizes_vector_from_read_counts(X_sub, var_sub.utr_size)
    distances = _calculate_umi_distance_to_pas(umi_utr_sizes=umi_utr_sizes, pas_end=pas_end)

    distances_ = distances.to_numpy().flatten()

    total_umis = len(distances)
    threshold = math.ceil(total_umis * FREQUENT_SITE_SHARE)
    distances_bigger_ten = distances[distances.groupby(level=0).count() > threshold]
    in_range_of_pas = distances_bigger_ten.between(10, 30)

    return [in_range_of_pas.sum() / total_umis, distances_]


def _pas_table_per_utr_compact(
    X_sub: npt.NDArray[Any] | scipy.sparse.spmatrix,
    var_sub: pd.DataFrame,
    hamming: int,
    genome: genome_.GenomeFasta,
) -> tuple[npt.NDArray[np.int64], list[str] | None, npt.NDArray[np.int64] | None]:
    """Each site's molecules, nearest PAS hexamer and distance to it, for one UTR.

    Returns:
        ``(molecules, hexamers, distances)``, one value per row of *var_sub*;
        hexamers and distances are None when the UTR has no hexamer.
    """
    molecules = np.asarray(X_sub.sum(axis=0)).ravel().astype(np.int64)
    pas_end, pas_seq, _ = _find_pas(var_sub, hamming=hamming, genome=genome)
    if not pas_end:
        return molecules, None, None

    sizes = var_sub.utr_size.to_numpy(dtype=np.int64)
    distances = sizes[:, np.newaxis] - np.asarray(pas_end)[np.newaxis, :]
    nearest = np.abs(distances).argmin(axis=1)
    hexamers = [pas_seq[hexamer] for hexamer in nearest]
    return molecules, hexamers, distances[np.arange(len(sizes)), nearest]


def _peak_stats_per_utr_compact(
    X_sub: npt.NDArray[Any] | scipy.sparse.spmatrix,
    var_sub: pd.DataFrame,
    bandwidth: Bandwidth,
    kernel: str = "gaussian",
    caller: SizeTransform | None = None,
    bootstrap: int = 0,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """The peaks of one UTR's raw sizes, with the spread of the sizes in each peak.

    Args:
        X_sub: The UTR's counts (cells x sizes).
        var_sub: The UTR's sizes, in ``utr_size``.
        bandwidth: KDE bandwidth: a number of nt, or a rule.
        kernel: The KDE's kernel.
        caller: The peak caller; by default the KDE with *bandwidth* and *kernel*.
        bootstrap: How many resamples of the molecules give each peak's
            stability (see :func:`_peak_stability`); 0 for none.
        rng: Random numbers for the resamples.

    Returns:
        A row per peak: ``utr_size`` (the peak), ``molecules``, the UTR's
        ``bandwidth`` (see peak_calling.bandwidth_rules), the mean, standard
        deviation, minimum and maximum of its molecules' raw sizes, and with
        *bootstrap*, its ``stability``.
    """
    counts = np.asarray(X_sub.sum(axis=0)).ravel().astype(np.int64)
    sizes = var_sub.utr_size.to_numpy(dtype=np.int64)[counts > 0]
    counts = counts[counts > 0]
    molecules = np.repeat(sizes, counts)
    call = _caller(bandwidth, kernel, caller)
    peaks = call(molecules)
    # All molecules of a size go to the same peak.
    first_of_size = np.concatenate([[0], np.cumsum(counts)[:-1]])
    sizes_in = pd.DataFrame({"peak": peaks[first_of_size], "size": sizes, "count": counts})
    sizes_in["weighted"] = sizes_in["size"] * sizes_in["count"]
    by_peak = sizes_in.groupby("peak", sort=True)
    stats = pd.DataFrame(
        {
            "molecules": by_peak["count"].sum(),
            "size_mean": by_peak["weighted"].sum() / by_peak["count"].sum(),
            "size_min": by_peak["size"].min(),
            "size_max": by_peak["size"].max(),
        }
    )
    sizes_in["squares"] = (
        sizes_in["count"] * (sizes_in["size"] - sizes_in["peak"].map(stats["size_mean"])) ** 2
    )
    stats["size_sd"] = np.sqrt(sizes_in.groupby("peak")["squares"].sum() / stats["molecules"])
    stats["bandwidth"] = utr_bandwidth(molecules, bandwidth)
    if bootstrap:
        stats["stability"] = _peak_stability(
            sizes,
            counts,
            stats.index.to_numpy(),
            stats["bandwidth"].iloc[0],
            call,
            bootstrap,
            rng if rng is not None else np.random.default_rng(),
        )
    stats.index.name = "utr_size"
    return stats.reset_index()


def _peak_stability(
    sizes: npt.NDArray[np.int64],
    counts: npt.NDArray[np.int64],
    peaks: npt.NDArray[np.int64],
    tolerance: float,
    call: SizeTransform,
    resamples: int,
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """How often each peak comes back when the UTR's molecules are drawn again.

    Draws as many molecules as the UTR has, with replacement, *resamples*
    times, and calls their peaks. A peak comes back in a resample with a peak
    within *tolerance* nt (the UTR's bandwidth) of it.

    Returns:
        The share of resamples in which each of *peaks* comes back; NaN for
        all if *tolerance* is (3 molecules or fewer, which take their median).
    """
    if math.isnan(tolerance):
        return np.full(len(peaks), np.nan)
    total = int(counts.sum())
    back = np.zeros(len(peaks), dtype=np.int64)
    for _ in range(resamples):
        drawn = rng.multinomial(total, counts / total)
        resampled_peaks = np.unique(call(np.repeat(sizes, drawn)))
        # The resampled peaks next to each peak, on either side.
        following = np.searchsorted(resampled_peaks, peaks)
        right = resampled_peaks[np.minimum(following, len(resampled_peaks) - 1)]
        left = resampled_peaks[np.maximum(following - 1, 0)]
        back += np.minimum(np.abs(right - peaks), np.abs(left - peaks)) <= tolerance
    return back / resamples


# ---------------------------------------------------------------------------
# Batch workers — receive a list of pre-extracted per-UTR slices. The parent
# materializes the slices once and ships only the slices each batch needs;
# workers never see the full AnnData. ``orig_idx`` lets the parent re-sort
# flattened results back to deterministic UTR order.
# ---------------------------------------------------------------------------

CompactUtrItem = tuple[int, str, Any, pd.DataFrame]


def _peak_call_utr_batch(
    batch: list[CompactUtrItem], bandwidth: Bandwidth, caller: SizeTransform
) -> list[tuple[int, npt.NDArray[Any], pd.DataFrame]]:
    out = []
    for orig_idx, _utr_name, X_sub, var_sub in batch:
        mtx, var_df = _peak_call_single_utr_compact(X_sub, var_sub, bandwidth, caller=caller)
        out.append((orig_idx, mtx, var_df))
    return out


def _pas_distance_utr_batch(
    batch: list[CompactUtrItem], hamming: int, genome: genome_.GenomeFasta
) -> list[tuple[int, npt.NDArray[Any] | None]]:
    out = []
    with genome_.open_genome(genome) as fasta:  # once per batch
        for orig_idx, _utr_name, X_sub, var_sub in batch:
            result = _get_distances_per_utr_compact(X_sub, var_sub, hamming, fasta)
            out.append((orig_idx, result))
    return out


def _pas_table_utr_batch(
    batch: list[CompactUtrItem], hamming: int, genome: genome_.GenomeFasta
) -> list[tuple[int, npt.NDArray[np.int64], list[str] | None, npt.NDArray[np.int64] | None]]:
    out = []
    with genome_.open_genome(genome) as fasta:  # once per batch
        for orig_idx, _utr_name, X_sub, var_sub in batch:
            molecules, hexamers, distances = _pas_table_per_utr_compact(
                X_sub, var_sub, hamming, fasta
            )
            out.append((orig_idx, molecules, hexamers, distances))
    return out


def _peak_stats_utr_batch(
    batch: list[CompactUtrItem],
    bandwidth: Bandwidth,
    caller: SizeTransform,
    bootstrap: int = 0,
    seed: int = 0,
) -> list[tuple[int, pd.DataFrame]]:
    # Each UTR draws from its own random numbers, so the resamples do not
    # depend on how the UTRs are batched.
    return [
        (
            orig_idx,
            _peak_stats_per_utr_compact(
                X_sub,
                var_sub,
                bandwidth,
                caller=caller,
                bootstrap=bootstrap,
                rng=np.random.default_rng([seed, orig_idx]),
            ),
        )
        for orig_idx, _utr_name, X_sub, var_sub in batch
    ]


def _score_distances_utr_batch(
    batch: list[CompactUtrItem], hamming: int, genome: genome_.GenomeFasta
) -> list[tuple[int, list[float | tuple[float, npt.NDArray[Any]]] | None]]:
    out = []
    with genome_.open_genome(genome) as fasta:  # once per batch
        for orig_idx, _utr_name, X_sub, var_sub in batch:
            result = _get_score_distances_per_utr_compact(X_sub, var_sub, hamming, fasta)
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
    bandwidth: Bandwidth = 1,
    n_jobs: int | None = None,
    kernel: str = "gaussian",
) -> ad.AnnData:
    """Call peaks on an AnnData using joblib parallelism across UTRs.

    Each UTR's peaks are consolidated in parallel via
    :func:`_peak_call_single_utr`, then results are merged into a new
    AnnData. UTRs are bundled into ``n_workers * 4`` batches before dispatch
    to amortize joblib pickling overhead.

    Args:
        adata: AnnData with UMI counts in ``.X`` and UTR metadata in
            ``.var`` (must contain ``utr_name``, ``gene_name``, ``utr_size``).
        method: The peak caller's name (see peak_calling.callers), ``kde`` for
            the KDE.
        bandwidth: KDE bandwidth: a number of nt, or a rule that sets one per
            UTR (see peak_calling.bandwidth_rules), whose value goes to the
            sites' ``bandwidth`` column. ``0`` means no peak calling is
            performed.
        n_jobs: Worker process count. ``None`` falls back to
            JOBLIB_DEFAULT_N_JOBS; non-positive resolves to all-but-one core.
        kernel: The KDE's kernel, one of peak_calling.settings.KERNELS.

    Returns:
        A new AnnData with peaks consolidated.

    Raises:
        ValueError: If no peak caller has that name, or the bandwidth or the
            kernel is not one.
    """
    logger = module_log.module_logger()

    check_kernel(kernel)
    bandwidth = parse_bandwidth(bandwidth, allow_zero=True)
    if bandwidth == 0:
        logger.info("Using Bandwidth = 0 means no changes were made")
        return adata
    caller = get_caller(method, bandwidth=bandwidth, kernel=kernel)

    sparse_adata = scipy.sparse.issparse(adata.X)

    n_workers = helpers.resolve_n_jobs(n_jobs)
    per_utr_inputs = _extract_per_utr_inputs(adata)
    batches = _build_compact_batches(per_utr_inputs, n_workers)

    logger.info(
        "Calling peaks across %d UTRs (%d workers, %d batches)",
        len(per_utr_inputs),
        n_workers,
        len(batches),
    )

    try:
        with joblib.parallel_config(backend=JOBLIB_DEFAULT_BACKEND, inner_max_num_threads=1):
            batch_results = list(
                joblib.Parallel(n_jobs=n_workers, return_as="generator")(
                    joblib.delayed(_peak_call_utr_batch)(batch, bandwidth, caller)
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
    var_df.index = anndata_processing.site_names(var_df)
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
    bandwidth: Bandwidth = 1,
    min_umis: int | None = None,
    kernel: str = "gaussian",
) -> ad.AnnData:
    """Call peaks on an AnnData, processing UTRs sequentially.

    Iterates over each UTR, optionally skipping those below a minimum UMI
    count, and consolidates peaks using :func:`_peak_call_single_utr`.
    Sparse input is preserved (output is re-sparsified if the input was
    sparse).

    Args:
        adata: AnnData with UMI counts in ``.X`` and UTR metadata in
            ``.var`` (must contain ``utr_name``, ``gene_name``, ``utr_size``).
        method: The peak caller's name (see peak_calling.callers), ``kde`` for
            the KDE.
        bandwidth: KDE bandwidth: a number of nt, or a rule that sets one per
            UTR (see peak_calling.bandwidth_rules), whose value goes to the
            sites' ``bandwidth`` column. ``0`` means no peak calling is
            performed.
        min_umis: If set, UTRs with fewer total UMIs are skipped.
        kernel: The KDE's kernel, one of peak_calling.settings.KERNELS.

    Returns:
        A new AnnData with peaks consolidated.

    Raises:
        ValueError: If no peak caller has that name, or the bandwidth or the
            kernel is not one.
    """
    logger = module_log.module_logger()

    check_kernel(kernel)
    bandwidth = parse_bandwidth(bandwidth, allow_zero=True)
    if bandwidth == 0:
        logger.info("Using Bandwidth = 0 means no changes were made")
        return adata
    caller = get_caller(method, bandwidth=bandwidth, kernel=kernel)

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
                logger.info(f"Skipping UTR {utr} with only {total_umis} UMIs (< {min_umis})")
                continue

        mtx, var = _peak_call_single_utr(adata_utr, bandwidth=bandwidth, caller=caller)
        mtx_list.append(mtx)
        var_list.append(var)

    var_df = pd.concat(var_list, axis=0).drop("pA", axis=1, errors="ignore")
    var_df.index = anndata_processing.site_names(var_df)
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
    group_by: str | None = None,
) -> ad.AnnData:
    """Remove UTR peaks whose counts fall below a threshold.

    For each UTR the threshold is ``max(total_utr_counts * threshold,
    minimum_counts)``.  Peaks below this are dropped.

    With *group_by*, the counts are those of each group of cells (such as a
    treatment), and a peak stays if it reaches its group's threshold in at
    least one group. A site that only some conditions use, such as one that a
    treatment raises, then stays, though it is a small share of all counts
    (the filter of Eugenia Guerrero Aruffo's master's thesis, 2026, by IFN
    treatment). The minimum count holds in each group, so a site spread
    thinly over the groups can go, though all cells together would keep it.
    Cells without a group count in none.

    Args:
        adata: The AnnData object to filter.
        threshold: Fractional threshold applied to each UTR's total count
            (e.g. ``0.1`` means 10%).
        minimum_counts: Absolute minimum count a peak must reach.
        group_by: A column of ``.obs`` that groups the cells, or None for all
            cells together.

    Returns:
        A filtered AnnData with low-count peaks removed.

    Raises:
        ValueError: If *group_by* is not a column of ``.obs``.
    """
    utr_names = adata.var.loc[:, "utr_name"].values
    if group_by is None:
        # Sparse-safe: sum columns directly instead of materializing dense matrix
        site_counts = np.asarray(adata.X.sum(axis=0)).ravel()[np.newaxis, :]
    elif group_by in adata.obs:
        # A row per group: which cells it has (code -1: no group).
        codes = pd.Categorical(adata.obs[group_by]).codes
        in_a_group = np.flatnonzero(codes >= 0)
        membership = scipy.sparse.csr_matrix(
            (np.ones(len(in_a_group)), (codes[in_a_group], in_a_group)),
            shape=(codes.max() + 1, adata.n_obs),
        )
        group_counts = membership @ adata.X
        site_counts = (
            group_counts.toarray()
            if scipy.sparse.issparse(group_counts)
            else np.asarray(group_counts)
        )
    else:
        raise ValueError(
            f"No column {group_by!r} in the cell annotations (.obs): "
            f"{', '.join(map(str, adata.obs.columns))}"
        )
    # Each group's (row's) counts per UTR, and the share of them a site needs.
    utr_totals = pd.DataFrame(site_counts.T).groupby(utr_names, observed=True, sort=False).sum()
    utr_thresholds = np.maximum(utr_totals.to_numpy() * threshold, minimum_counts)
    site_thresholds = utr_thresholds[utr_totals.index.get_indexer(utr_names)].T
    bigger_than_threshold = (site_counts >= site_thresholds).any(axis=0)

    x_ = adata.X[:, bigger_than_threshold]
    var_ = adata.var[bigger_than_threshold]

    return ad.AnnData(X=x_, var=var_, obs=adata.obs)


def estimate_best_bandwidth(
    adata: ad.AnnData,
    genome: genome_.GenomeFasta,
    method: str = "kde",
    hamming: int = 1,
    bandwidths: float | Sequence[float] = (0.5, 1, 1.5, 2, 2.5, 3),
    return_distances_to_hex: bool = False,
    n_jobs: int | None = None,
) -> list[float | tuple[float, npt.NDArray[Any]]]:
    """Estimate the best KDE bandwidth by scoring PAS proximity.

    Runs peak calling at each candidate bandwidth, then scores each result
    by how many UMIs fall within 10-30 nt of a PAS hexamer.

    The bandwidths are evaluated sequentially; parallelism is consumed by the
    inner per-UTR loops in :func:`peak_call_anndata_parallel` and
    :func:`_score_peak_assignment` so all cores stay productive without
    nesting joblib pools.

    Args:
        adata: AnnData with UMI counts and UTR metadata.
        genome: Genome FASTA (path, or opened with pysam) to search for PAS
            hexamers in.
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
            genome=genome,
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

    n_workers = helpers.resolve_n_jobs(n_jobs)
    per_utr_inputs = _extract_per_utr_inputs(adata)
    batches = _build_compact_batches(per_utr_inputs, n_workers)

    logger.info(
        "Computing PAS distances across %d UTRs (%d workers, %d batches)",
        len(per_utr_inputs),
        n_workers,
        len(batches),
    )

    try:
        with joblib.parallel_config(backend=JOBLIB_DEFAULT_BACKEND, inner_max_num_threads=1):
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
    if not distances_list:  # no UTR contains a polyadenylation signal
        return np.empty(0, dtype=np.int64)
    distances_array = np.concatenate(distances_list)

    return distances_array


def pas_distance_table(
    adata: ad.AnnData,
    genome: genome_.GenomeFasta,
    hamming: int = 0,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    """Each site's distance to the nearest polyadenylation signal (PAS) of its UTR.

    The distance is defined as in :func:`pas_distance`: the site's UTR size
    minus the end of the nearest PAS hexamer (AATAAA with up to *hamming*
    mismatches), so sites 10-30 nt after a hexamer give 10 to 30. All the
    molecules of a site share its distance: weight the distances by the
    ``molecules`` column for their distribution over molecules.

    Args:
        adata: Count matrix with ``utr_name``, ``utr_size``, ``chrom``,
            ``start``, ``end`` and ``strand`` in ``.var``.
        genome: Genome FASTA (path, or opened with pysam) of the annotation.
        hamming: Maximum number of mismatches to AATAAA.
        n_jobs: Worker process count for the per-UTR parallel loop.

    Returns:
        One row per site, indexed by the site names in the order of
        ``adata.var``, with the columns ``gene_name`` (if ``.var`` has it),
        ``utr_name``, ``utr_size``, ``molecules`` (summed over all cells),
        ``pas_hexamer`` (the sequence of the nearest hexamer) and
        ``pas_distance``. The last two are missing for UTRs without a hexamer.
    """
    n_workers = helpers.resolve_n_jobs(n_jobs)
    per_utr_inputs = _extract_per_utr_inputs(adata)
    batches = _build_compact_batches(per_utr_inputs, n_workers)
    columns = adata.var.groupby("utr_name", observed=True).indices

    try:
        with joblib.parallel_config(backend=JOBLIB_DEFAULT_BACKEND, inner_max_num_threads=1):
            batch_results = list(
                joblib.Parallel(n_jobs=n_workers, return_as="generator")(
                    joblib.delayed(_pas_table_utr_batch)(batch, hamming, genome)
                    for batch in tqdm(batches, colour="#ffb700", ascii=" ■", desc="UTR: ")
                )
            )
    finally:
        helpers.shutdown_joblib_pool()

    utr_names = [utr for _idx, utr, _X, _var in per_utr_inputs]
    molecules = np.zeros(adata.n_vars, dtype=np.int64)
    hexamers = pd.Series(pd.NA, index=range(adata.n_vars), dtype="string")
    distances = pd.Series(pd.NA, index=range(adata.n_vars), dtype="Int64")
    for result in batch_results:
        for orig_idx, utr_molecules, utr_hexamers, utr_distances in result:
            positions = columns[utr_names[orig_idx]]
            molecules[positions] = utr_molecules
            if utr_distances is not None:
                hexamers.iloc[positions] = utr_hexamers
                distances.iloc[positions] = utr_distances

    table = adata.var[[c for c in ("gene_name", "utr_name", "utr_size") if c in adata.var]].copy()
    table["molecules"] = molecules
    table["pas_hexamer"] = hexamers.array
    table["pas_distance"] = distances.array  # whole numbers, missing without a hexamer
    table.index.name = "site"
    return table


def peak_table(
    adata: ad.AnnData,
    bandwidth: Bandwidth = 12,
    n_jobs: int | None = None,
    kernel: str = "gaussian",
    method: str = "kde",
    bootstrap: int = 0,
    seed: int = 0,
) -> pd.DataFrame:
    """The peaks that a bandwidth calls from raw sizes, and how spread out each one is.

    Calls the peaks of each UTR as `call-peaks` does, and describes the raw
    sizes that each peak collects. A peak whose sizes spread much wider than
    the bandwidth may join neighbouring sites, or noise.

    With *bootstrap*, also gives each peak's stability: draws as many
    molecules as each UTR has from its molecules, with replacement,
    *bootstrap* times, calls their peaks, and counts how often a peak comes
    back within the UTR's bandwidth. Peaks of few molecules, or on the flank
    of a larger one, come back less often. Stability grows with the
    bandwidth (one peak per UTR always comes back): it compares the peaks of
    one setting, not settings.

    Args:
        adata: Raw-sizes count matrix, with ``utr_name``, ``utr_size`` and
            optionally ``gene_name`` in ``.var``.
        bandwidth: KDE bandwidth of the peak calling: a number of nt, or a
            rule (see peak_calling.bandwidth_rules).
        n_jobs: Worker process count for the per-UTR loop.
        kernel: The KDE's kernel, one of peak_calling.settings.KERNELS.
        method: The peak caller's name (see peak_calling.callers).
        bootstrap: How many resamples give the stability; 0 for none.
        seed: Seed of the resamples' random numbers: the same seed gives the
            same stabilities.

    Returns:
        A row per peak, indexed by its site name (as `call-peaks` names it),
        in UTR order, with ``gene_name`` (if ``.var`` has it), ``utr_name``,
        ``utr_size`` (the peak), ``molecules``, ``utr_share`` (of the UTR's
        molecules), ``utr_sites`` (peaks in the UTR), ``bandwidth`` (the UTR's;
        missing for 3 molecules or fewer, which take their median), the
        ``size_mean``, ``size_sd`` (weighted by molecules), ``size_min`` and
        ``size_max`` of the raw sizes it collects, and with *bootstrap*, its
        ``stability`` (the share of resamples it comes back in; missing for 3
        molecules or fewer).

    Raises:
        ValueError: If *bootstrap* is negative, or for the reasons of
            peak_calling.callers.get_caller.
    """
    if bootstrap < 0:
        raise ValueError(f"bootstrap {bootstrap}: give 0 resamples or more")
    check_kernel(kernel)
    bandwidth = parse_bandwidth(bandwidth)
    caller = get_caller(method, bandwidth=bandwidth, kernel=kernel)
    n_workers = helpers.resolve_n_jobs(n_jobs)
    has_molecules = np.asarray(adata.X.sum(axis=0)).ravel() > 0
    per_utr_inputs = _extract_per_utr_inputs(adata[:, has_molecules])
    batches = _build_compact_batches(per_utr_inputs, n_workers)

    try:
        with joblib.parallel_config(backend=JOBLIB_DEFAULT_BACKEND, inner_max_num_threads=1):
            batch_results = list(
                joblib.Parallel(n_jobs=n_workers, return_as="generator")(
                    joblib.delayed(_peak_stats_utr_batch)(batch, bandwidth, caller, bootstrap, seed)
                    for batch in tqdm(batches, colour="#ffb700", ascii=" ■", desc="UTR: ")
                )
            )
    finally:
        helpers.shutdown_joblib_pool()

    per_utr = dict(item for result in batch_results for item in result)
    annotations = [c for c in ("gene_name", "utr_name") if c in adata.var]
    tables = []
    for orig_idx, _utr_name, _X_sub, var_sub in per_utr_inputs:
        stats = per_utr[orig_idx]
        for column in annotations:
            stats[column] = var_sub[column].iloc[0]
        stats["utr_share"] = stats["molecules"] / stats["molecules"].sum()
        stats["utr_sites"] = len(stats)
        tables.append(stats)
    columns = [*annotations, "utr_size", "molecules", "utr_share", "utr_sites", "bandwidth"]
    columns += ["size_mean", "size_sd", "size_min", "size_max"]
    columns += ["stability"] if bootstrap else []
    if not tables:
        return pd.DataFrame(columns=columns, index=pd.Index([], name="site"))
    table = pd.concat(tables, ignore_index=True)[columns]
    table.index = pd.Index(anndata_processing.site_names(table), name="site")
    return table


def score_bandwidths(
    adata: ad.AnnData,
    genome: genome_.GenomeFasta | None = None,
    bandwidths: Sequence[Bandwidth] | None = None,
    hamming: int = 1,
    n_jobs: int | None = None,
    kernel: str | Sequence[str] = "gaussian",
    utr_end: str | None = None,
    *,
    scores: Sequence[str] | None = None,
    known_sites: str | Path | None = None,
    window: int = 10,
    method: str = "kde",
) -> pd.DataFrame:
    """Score peak-calling settings by how well their sites fall where true sites are.

    Calls the peaks of a raw-sizes count matrix with each kernel and each
    bandwidth, and scores the sites of each setting (see
    peak_calling.scores): per UTR, the share of its molecules at sites that
    look true, averaged over UTRs. Higher is better. By default, the score of
    the matrix's UTR end:

    - 3' data, ``pas``: sites 10-30 nt after the nearest polyadenylation
      signal (AATAAA with up to *hamming* mismatches), as
      :func:`estimate_best_bandwidth` scores them. It changes little between
      Gaussian bandwidths of 6 and 20 nt.
    - 5' data, ``initiator``: sites whose genome reads a pyrimidine (C or T)
      at -1 and a purine (A or G) at +1, the site, where RNA polymerase II
      starts most transcripts.

    ``known-sites`` scores sites within *window* nt of the known sites of a
    database, for either end. A setting that splits sites gives many sites.

    Args:
        adata: Raw-sizes count matrix, with UTR metadata in ``.var``.
        genome: Genome FASTA (path, or opened with pysam) of the annotation,
            for the ``pas`` and ``initiator`` scores.
        bandwidths: The bandwidths to score: numbers of nt, or rules (see
            peak_calling.bandwidth_rules); 0 scores the raw sizes. By default,
            those of settings.CANDIDATE_BANDWIDTHS for the matrix's UTR end.
        hamming: Maximum number of mismatches to AATAAA (``pas``).
        n_jobs: Worker process count for the per-UTR loops.
        kernel: The KDE's kernel, or several kernels to score each bandwidth
            with, of peak_calling.settings.KERNELS.
        utr_end: Which end of the UTR the sizes count from: ``three_prime`` or
            ``five_prime``. By default, what the matrix records, or
            ``three_prime`` for matrices without a record.
        scores: The scores' names (see settings.SCORES); by default that of the
            UTR end (settings.DEFAULT_SCORES).
        known_sites: BED file of known sites, for the ``known-sites`` score.
        window: Distance (nt) up to which a site is near a known site.
        method: The peak caller's name (see peak_calling.callers).

    Returns:
        A row per setting, indexed by ``kernel`` and ``bandwidth`` (kernels in
        the order given, each with all bandwidths), with a column per score,
        named after it, then ``sites`` (the number of peaks) and
        ``sites_per_utr``.

    Raises:
        ValueError: If *utr_end* is neither UTR end, a bandwidth, a kernel or
            a score is not one, a score does not score this UTR end or lacks
            what it needs, or a matrix is at gene level or on another assembly.
    """
    utr_end = utr_end or provenance.recorded(adata).get("utr_end", "three_prime")
    if utr_end not in genome_.UTR_ENDS:
        raise ValueError(f"utr_end {utr_end!r}: expected three_prime or five_prime")
    kernels = [kernel] if isinstance(kernel, str) else list(kernel)
    for kernel_ in kernels:
        check_kernel(kernel_)
    if bandwidths is None:
        bandwidths = CANDIDATE_BANDWIDTHS[utr_end]
    bandwidths = [parse_bandwidth(bandwidth, allow_zero=True) for bandwidth in bandwidths]
    context = ScoreContext(
        utr_end=utr_end,
        genome=None if genome is None else str(getattr(genome, "fn", genome)),
        known_sites=None if known_sites is None else str(known_sites),
        hamming=hamming,
        window=window,
        n_jobs=n_jobs,
    )
    names = scores if scores else [DEFAULT_SCORES[utr_end]]
    scorers = {name.lower(): make_score(name, adata, context) for name in dict.fromkeys(names)}

    rows = []
    for kernel_ in kernels:
        for bandwidth in bandwidths:
            peaks = peak_call_anndata_parallel(
                adata, method=method, bandwidth=bandwidth, n_jobs=n_jobs, kernel=kernel_
            )
            row: dict[str, Any] = {"kernel": kernel_, "bandwidth": bandwidth}
            row |= {name: scorer.score(peaks) for name, scorer in scorers.items()}
            row["sites"] = peaks.n_vars
            row["sites_per_utr"] = peaks.n_vars / peaks.var.utr_name.nunique()
            rows.append(row)
    return pd.DataFrame(rows).set_index(["kernel", "bandwidth"])


def best_setting(scores: pd.DataFrame, score: str | None = None) -> tuple[str, Bandwidth]:
    """The kernel and the bandwidth of the highest score in a table of :func:`score_bandwidths`.

    Of equal scores, the first setting of the table wins. The bandwidth comes
    as peak_calling.settings.parse_bandwidth writes it, also from a table
    read from text.

    Args:
        scores: The table.
        score: The column of the score to maximise; by default the first.

    Raises:
        ValueError: If no setting has that score (no UTR could be scored).
    """
    values = scores[scores.columns[0] if score is None else score]
    if values.isna().all():
        raise ValueError(
            f"No setting has a {values.name} score: no UTR could be scored. Check that the "
            "genome or the known sites come from the matrix's genome assembly."
        )
    kernel, bandwidth = cast(tuple[str, Bandwidth], values.idxmax())
    return str(kernel), parse_bandwidth(bandwidth, allow_zero=True)


def _score_peak_assignment(
    adata: ad.AnnData,
    genome: genome_.GenomeFasta,
    hamming: int = 1,
    return_distances_to_hex: bool = False,
    n_jobs: int | None = None,
) -> float | tuple[float, npt.NDArray[Any]]:
    """Score peak assignments by proximity to PAS hexamers.

    Args:
        adata: AnnData with peak-called UTR counts.
        genome: Genome FASTA (path, or opened with pysam) to search for PAS
            hexamers in.
        hamming: Maximum Hamming distance for PAS matching.
        return_distances_to_hex: If ``True``, also return the raw distance
            array.
        n_jobs: Worker process count for the per-UTR parallel loop.

    Returns:
        A score (fraction of UMIs within 10-30 nt of a PAS), or a
        ``(score, distances_array)`` tuple.
    """
    logger = module_log.module_logger()
    n_workers = helpers.resolve_n_jobs(n_jobs)
    per_utr_inputs = _extract_per_utr_inputs(adata)
    batches = _build_compact_batches(per_utr_inputs, n_workers)

    logger.info(
        "Scoring peaks across %d UTRs (%d workers, %d batches)",
        len(per_utr_inputs),
        n_workers,
        len(batches),
    )

    try:
        with joblib.parallel_config(backend=JOBLIB_DEFAULT_BACKEND, inner_max_num_threads=1):
            batch_results = list(
                joblib.Parallel(n_jobs=n_workers, return_as="generator")(
                    joblib.delayed(_score_distances_utr_batch)(batch, hamming, genome)
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
    var: pd.DataFrame, genome: genome_.GenomeFasta, hamming: int = 1
) -> tuple[list[int], list[str], str]:
    """Find polyadenylation signal (PAS) sites in a UTR sequence.

    Extracts the genomic sequence described in *var* (one UTR's metadata
    rows) and searches for AATAAA motifs (with up to *hamming* mismatches).

    Args:
        var: Per-UTR var rows with chrom/start/end/strand columns. All rows
            describe the same UTR, so only the first is consulted.
        genome: The genome FASTA, its path or opened with pysam.
        hamming: Maximum Hamming distance for fuzzy matching.

    Returns:
        A tuple ``(pas_end, pas_seq, strand)`` where *pas_end* is a list
        of 0-based end positions, *pas_seq* the matched sequences, and
        *strand* the genomic strand.
    """
    seq_strand = var.strand.iloc[0]
    utr_seq = genome_.region_sequence(
        genome, var.chrom.iloc[0], int(var.start.iloc[0]), int(var.end.iloc[0]), seq_strand
    )

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


def _calculate_umi_distance_to_pas(umi_utr_sizes: pd.Series, pas_end: list[int]) -> pd.Series:
    """Calculate the distance from each UMI to its nearest PAS hexamer.

    The definition of Fig. 1B of Macedo et al. 2026. UTR sizes and hexamer ends
    are both positions along the UTR in transcript orientation (``_find_pas``
    reads the reverse complement of - strand UTRs), so the distance is
    ``utr_size - pas_end`` on both strands: positive when the molecule ends
    downstream of the hexamer. With several hexamers the nearest one counts,
    by absolute distance (the first of equally near ones).

    Args:
        umi_utr_sizes: Series of UTR sizes (one per UMI).
        pas_end: PAS end positions within the UTR, as returned by ``_find_pas``.

    Returns:
        A Series of signed distances, with the index of *umi_utr_sizes*.
    """
    sizes = umi_utr_sizes.to_numpy()
    distances = sizes[:, np.newaxis] - np.asarray(pas_end)[np.newaxis, :]
    nearest = np.abs(distances).argmin(axis=1)
    return pd.Series(distances[np.arange(len(sizes)), nearest], index=umi_utr_sizes.index)


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
    # .sum works on dense and sparse matrices alike, without densifying
    counts_sum = np.asarray(counts_arr.sum(axis=0)).ravel().astype(int)
    return sizes_arr.repeat(counts_sum)
