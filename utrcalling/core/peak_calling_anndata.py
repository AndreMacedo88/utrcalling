"""Peak calling functions operating on AnnData objects.

Provides secondary peak calling on existing AnnData objects, including KDE-based
peak consolidation, PAS distance scoring, and bandwidth estimation.
"""

from __future__ import annotations

import math

import regex
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema

import pybedtools
import joblib
from tqdm import tqdm

from utrcalling.core import module_log


# ---------------------------------------------------------------------------
# Shared per-UTR peak calling logic (module-level for joblib picklability)
# ---------------------------------------------------------------------------


def _peak_call_single_utr(
    adata_utr: ad.AnnData, method: str, bandwidth: float
) -> tuple[np.ndarray, pd.DataFrame]:
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
# Extracted joblib workers for PAS distance scoring
# ---------------------------------------------------------------------------


def _get_distances_per_utr(
    adata: ad.AnnData, utr: str, hamming: int, genome: str
) -> np.ndarray | None:
    """Compute PAS distances for a single UTR (module-level for joblib).

    Args:
        adata: Full AnnData object.
        utr: UTR name to process.
        hamming: Maximum Hamming distance for PAS matching.
        genome: Path to the genome FASTA file.

    Returns:
        1-D numpy array of distances, or ``None`` if no PAS found.
    """
    adata_utr = adata[:, adata.var.utr_name == utr]

    pas_end, _, seq_strand = _find_pas(adata_utr, hamming=hamming, genome=genome)

    if not pas_end:
        return None

    umi_utr_sizes = _utr_sizes_vector_from_read_counts(
        adata_utr.X, adata_utr.var.utr_size
    )

    distances = _calculate_umi_distance_to_pas(
        umi_utr_sizes=umi_utr_sizes, pas_end=pas_end, seq_strand=seq_strand
    )
    return distances.to_numpy().flatten()


def _get_score_distances_per_utr(
    adata: ad.AnnData, utr: str, hamming: int
) -> list | None:
    """Score PAS distances for a single UTR (module-level for joblib).

    Args:
        adata: Full AnnData object.
        utr: UTR name to process.
        hamming: Maximum Hamming distance for PAS matching.

    Returns:
        A list ``[score, distances_array]``, or ``None`` if no PAS found.
    """
    adata_utr = adata[:, adata.var.utr_name == utr]
    pas_end, _, seq_strand = _find_pas(adata_utr, hamming=hamming)

    if not pas_end:
        return None

    umi_utr_sizes = _utr_sizes_vector_from_read_counts(
        adata_utr.X, adata_utr.var.utr_size
    )

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
# Public API
# ---------------------------------------------------------------------------


def peak_call_anndata_parallel(
    adata: ad.AnnData, method: str, bandwidth: float = 1
) -> ad.AnnData:
    """Call peaks on an AnnData using joblib parallelism across UTRs.

    Each UTR's peaks are consolidated in parallel via
    :func:`_peak_call_single_utr`, then results are merged into a new
    AnnData.

    Args:
        adata: AnnData with UMI counts in ``.X`` and UTR metadata in
            ``.var`` (must contain ``utr_name``, ``gene_name``, ``utr_size``).
        method: Peak calling method (currently only ``"kde"``).
        bandwidth: KDE bandwidth. ``0`` means no peak calling is performed.

    Returns:
        A new AnnData with peaks consolidated.
    """
    logger = module_log.module_logger()

    if bandwidth == 0:
        logger.notice("Using Bandwidth = 0 means no changes were made")
        return adata

    sparse_adata = scipy.sparse.issparse(adata.X)

    utrs = adata.var.utr_name.unique()

    parallel_results = joblib.Parallel(n_jobs=-2)(
        joblib.delayed(_peak_call_single_utr)(
            adata_utr=adata[:, adata.var.utr_name == utr],
            method=method,
            bandwidth=bandwidth,
        )
        for utr in tqdm(utrs, colour="#ffb700", ascii=" ■")
    )

    mtx_list, var_list = list(zip(*parallel_results))

    var_df = pd.concat(var_list, axis=0).drop("pA", axis=1, errors="ignore")
    new_adata = ad.AnnData(X=np.hstack(mtx_list), obs=adata.obs, var=var_df)

    if sparse_adata:
        logger.info("Sparsifying data as initially provided.")
        new_adata.X = scipy.sparse.csr_matrix(new_adata.X)

    new_adata = new_adata[new_adata.obs.sort_index().index, :]
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

    new_adata = new_adata[new_adata.obs.sort_index().index, :]
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
    utr_total_counts = pd.Series(col_sums).groupby(utr_names).sum()
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
) -> list[float] | list[tuple[float, np.ndarray]]:
    """Estimate the best KDE bandwidth by scoring PAS proximity.

    Runs peak calling at each candidate bandwidth, then scores each result
    by how many UMIs fall within 10-30 nt of a PAS hexamer.

    Args:
        adata: AnnData with UMI counts and UTR metadata.
        method: Peak calling method (currently only ``"kde"``).
        hamming: Maximum Hamming distance for PAS matching.
        bandwidths: Candidate bandwidth values to evaluate.
        return_distances_to_hex: If ``True``, also return per-UMI distance
            arrays alongside scores.

    Returns:
        A list of scores (one per bandwidth), or a list of
        ``(score, distances_array)`` tuples if *return_distances_to_hex*
        is ``True``.
    """
    if isinstance(bandwidths, (int, float)):
        bandwidths = [bandwidths]

    peak_data = joblib.Parallel(n_jobs=-2)(
        joblib.delayed(peak_call_anndata)(
            adata=adata, method=method, bandwidth=bandwidth
        )
        for bandwidth in bandwidths
    )

    estimation_results = joblib.Parallel(n_jobs=-2)(
        joblib.delayed(_score_peak_assignment)(
            adata=adata_bandwidth,
            hamming=hamming,
            return_distances_to_hex=return_distances_to_hex,
        )
        for adata_bandwidth in peak_data
    )
    return estimation_results


def pas_distance(adata: ad.AnnData, genome: str, hamming: int = 1) -> np.ndarray:
    """Calculate UMI distances to the nearest polyadenylation signal (PAS).

    For each UTR, finds PAS hexamer sites (AATAAA with up to *hamming*
    mismatches) in the genome sequence and computes the distance of each
    UMI's cleavage site to the nearest PAS.

    Args:
        adata: AnnData with UMI counts per UTR, UTR sizes in
            ``.var.utr_size``, and UTR names in ``.var.utr_name``.
        genome: Path to a genome FASTA file.
        hamming: Maximum Hamming distance for fuzzy PAS matching.

    Returns:
        1-D array of distances (one per UMI across all UTRs with a PAS).
    """
    utrs = adata.var.utr_name.unique()

    distances_list = joblib.Parallel(n_jobs=-2)(
        joblib.delayed(_get_distances_per_utr)(
            adata=adata, utr=utr, hamming=hamming, genome=genome
        )
        for utr in tqdm(utrs, colour="#ffb700", ascii=" ■")
    )

    distances_list = [i for i in distances_list if i is not None]
    distances_array = np.concatenate(distances_list)

    return distances_array


def _score_peak_assignment(
    adata: ad.AnnData,
    hamming: int = 1,
    return_distances_to_hex: bool = False,
) -> float | tuple[float, np.ndarray]:
    """Score peak assignments by proximity to PAS hexamers.

    Args:
        adata: AnnData with peak-called UTR counts.
        hamming: Maximum Hamming distance for PAS matching.
        return_distances_to_hex: If ``True``, also return the raw distance
            array.

    Returns:
        A score (fraction of UMIs within 10-30 nt of a PAS), or a
        ``(score, distances_array)`` tuple.
    """
    logger = module_log.module_logger()
    utrs = adata.var.utr_name.unique()

    logger.info("Scoring peaks for each UTR")
    results_per_utr = joblib.Parallel(n_jobs=-2)(
        joblib.delayed(_get_score_distances_per_utr)(
            adata=adata, utr=utr, hamming=hamming
        )
        for utr in tqdm(utrs, colour="#ffb700", ascii=" ■", desc="UTR: ")
    )

    scores = []
    distances_list = []

    for i in results_per_utr:
        if i is None:
            continue
        scores.append(i[0])
        distances_list.append(i[1])

    score = sum(scores) / len(utrs)
    if return_distances_to_hex:
        distances_array = np.concatenate(distances_list)
        return (score, distances_array)
    else:
        return score


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_pas(
    adata: ad.AnnData, genome: str, hamming: int = 1
) -> tuple[list[int], list[str], str]:
    """Find polyadenylation signal (PAS) sites in a UTR sequence.

    Extracts the genomic sequence for the UTR described in *adata.var* and
    searches for AATAAA motifs (with up to *hamming* mismatches).

    Args:
        adata: AnnData subset for a single UTR.
        genome: Path to a genome FASTA file.
        hamming: Maximum Hamming distance for fuzzy matching.

    Returns:
        A tuple ``(pas_end, pas_seq, strand)`` where *pas_end* is a list
        of 0-based end positions, *pas_seq* the matched sequences, and
        *strand* the genomic strand.
    """
    seq_chrom = adata.var.chrom.iloc[0]
    seq_start = adata.var.start.iloc[0] - 1
    seq_stop = adata.var.end.iloc[0]
    seq_strand = adata.var.strand.iloc[0]

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
    counts_arr: np.ndarray | scipy.sparse.spmatrix,
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
    utr_sizes_vector: np.ndarray, bandwidth: float = 1
) -> np.ndarray:
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
