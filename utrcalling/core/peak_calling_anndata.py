"""
File containg functions relating to peak calling, but on AnnData objects.
"""
# import general tools
import os
from urllib.error import HTTPError
import time
import regex
# import numerical methods
import numpy as np
import math
from sklearn.neighbors import KernelDensity
import scipy
from scipy.signal import argrelextrema
# import data processing tools
import pandas as pd
import anndata as ad
# import biological tools
import pybedtools
# import quality of life tools
import joblib
from tqdm import tqdm
# import internal tools
from utrcalling.core import module_log

# Set up logging
logger = module_log.module_logger(disable=False, store_log=True, level="debug")


def peak_call_anndata_parallel(adata, method, **kwargs):

    def peak_call(adata, method, utr):
        adata_utr = adata[:, adata.var.utr_name == utr]
        gene = adata_utr.var.gene_name[0]
        utr_name = adata_utr.var.utr_name.str.split("_")[0]
        if len(utr_name) > 1:
            utr_number = f"UTR{utr_name[1]}_"
        else:
            utr_number = ""
        utr_var = adata_utr.var.iloc[0]

        # generate a vector of one UTR size per UMI count
        umi_utr_sizes = _utr_sizes_vector_from_read_counts(
            adata_utr.X.astype(int), adata_utr.var.utr_size)

        if str.lower(method) == "kde":
            peaks = _call_peaks_using_kernel_density_estimator(
                umi_utr_sizes.values, bandwidth=kwargs.get("bandwidth", 1)
            )
            peak_sizes = pd.Series(data=peaks, index=umi_utr_sizes.index)
            peak_sizes = peak_sizes[~peak_sizes.index.duplicated()]
            peak_sizes_unique = peak_sizes.unique()

        # create new matrix with the sum of UMIs for each new peak
        mtx_new_t = adata_utr.to_df().T
        mtx_new_t["sizes"] = peak_sizes.tolist()
        mtx_new = mtx_new_t.groupby("sizes").sum().T

        # create new polyA names
        umi_utr_sizes_new = mtx_new.columns
        polya_names = [f"{gene}_{utr_number}pA{x}" for x in range(
            1, len(umi_utr_sizes_new) + 1)]

        # generate new utr_var
        utr_var_new = pd.DataFrame([utr_var] * len(umi_utr_sizes_new))
        utr_var_new.index = polya_names
        utr_var_new["utr_size"] = peak_sizes_unique

        return mtx_new.to_numpy(), utr_var_new

    if kwargs.get("bandwidth", 1) == 0:
        logger.notice("Using Bandwidth = 0 means no changes were made")
        return adata

    utrs = adata.var.utr_name.unique()

    parallel_results = joblib.Parallel(n_jobs=-2)(
        joblib.delayed(peak_call)(adata=adata, method=method, utr=utr) for utr in tqdm(
            utrs, colour="#ffb700", ascii=" ■")
    )

    mtx_list, var_list = list(zip(*parallel_results))

    var_df = pd.concat(var_list, axis=0).drop("pA", axis=1, errors="ignore")
    new_adata = ad.AnnData(X=np.hstack(mtx_list), obs=adata.obs, var=var_df)
    new_adata = new_adata[new_adata.obs.sort_index().index, :]
    return new_adata


def peak_call_anndata(adata, method, **kwargs):
    if kwargs.get("bandwidth", 1) == 0:
        logger.notice("Using Bandwidth = 0 means no changes were made")
        return adata

    sparse_adata = False
    if scipy.sparse.issparse(adata.X):
        sparse_adata = True
        logger.info("Densifying sparse data for further processing")
        adata.X = adata.X.toarray()

    new_obs = adata.obs

    utrs = adata.var.utr_name.unique()

    mtx_list = []
    var_list = []

    # select the UTR
    logger.info("Calling peaks for each UTR")
    for utr in tqdm(utrs, colour="#ffb700", ascii=" ■", desc="UTR: "):
        adata_utr = adata[:, adata.var.utr_name == utr]
        gene = adata_utr.var.gene_name[0]
        utr_name = adata_utr.var.utr_name.str.split("_")[0]
        if len(utr_name) > 1:
            utr_number = f"UTR{utr_name[1]}_"
        else:
            utr_number = ""
        utr_var = adata_utr.var.iloc[0]

        # generate a vector of one UTR size per UMI count
        umi_utr_sizes = _utr_sizes_vector_from_read_counts(
            adata_utr.X.astype(int), adata_utr.var.utr_size)

        if str.lower(method) == "kde":
            peaks = _call_peaks_using_kernel_density_estimator(
                umi_utr_sizes.values, bandwidth=kwargs.get("bandwidth", 1)
            )
            peak_sizes = pd.Series(data=peaks, index=umi_utr_sizes.index)
            peak_sizes = peak_sizes[~peak_sizes.index.duplicated()]
            peak_sizes_unique = peak_sizes.unique()

        # create new matrix with the sum of UMIs for each new peak
        mtx_new_t = adata_utr.to_df().T
        mtx_new_t["sizes"] = peak_sizes.tolist()
        mtx_new = mtx_new_t.groupby("sizes").sum().T
        mtx_list.append(mtx_new.to_numpy())

        # create new polyA names
        umi_utr_sizes_new = mtx_new.columns
        polya_names = [f"{gene}_{utr_number}pA{x}" for x in range(
            1, len(umi_utr_sizes_new) + 1)]

        # generate new utr_var
        utr_var_new = pd.DataFrame([utr_var] * len(umi_utr_sizes_new))
        utr_var_new.index = polya_names
        utr_var_new["utr_size"] = peak_sizes_unique
        var_list.append(utr_var_new)

    del adata

    var_df = pd.concat(var_list, axis=0).drop("pA", axis=1, errors="ignore")
    new_adata = ad.AnnData(X=np.hstack(mtx_list), obs=new_obs, var=var_df)

    if sparse_adata:
        logger.info("Sparsifying data as initially provided.")
        new_adata.X = scipy.sparse.csr_matrix(new_adata.X)

    new_adata = new_adata[new_adata.obs.sort_index().index, :]
    return new_adata


def remove_peaks_below_count(adata, threshold=10):
    """
    Removes the UTR location peaks that have less than `threshold` counts in
    the sum of all samples. This effectively curates the peaks, selecting the most common
    and/or more prone to be true peaks.

    Args:
        adata (ad.AnnData): The AnnData object to filter.
        threshold (int, optional): The minimum counts number each peak must have. 
            Defaults to 10.

    Returns:
        ad.AnnData: A filtered AnnData object with UTR peaks counts above the threshold.
    """

    bigger_than_10 = adata.X.sum(axis=0) >= threshold
    if bigger_than_10.ndim > 1:
        bigger_than_10 = np.asarray(bigger_than_10).reshape(-1)

    x_ = adata.X[:, bigger_than_10]
    var_ = adata.var[bigger_than_10]

    return ad.AnnData(X=x_, var=var_, obs=adata.obs)


def estimate_best_bandwidth(adata, method="kde", hamming=1,
                            bandwidths=(0.5, 1, 1.5, 2, 2.5, 3),
                            return_distances_to_hex=False):

    if isinstance(bandwidths, int):
        bandwidths = [bandwidths]

    peak_data = joblib.Parallel(n_jobs=-2)(
        joblib.delayed(peak_call_anndata)(adata=adata, method=method, bandwidth=bandwidth) for bandwidth in bandwidths
    )

    estimation_results = joblib.Parallel(n_jobs=-2)(
        joblib.delayed(_score_peak_assignment)(adata=adata_bandwidth, hamming=hamming,
                                               return_distances_to_hex=return_distances_to_hex) for adata_bandwidth in peak_data
    )
    return estimation_results


def pas_distance(adata, genome, hamming=1):
    """
    Calculates the distance of each read to the nearest polyadenilation hexamer site
    AATAAA.

    Args:
        adata (_type_): an anndata file containing UMI counts per UTR, with UTR sizes
            in .var.utr_size and UTR names in .var.utr_name.
        genome (_type_): a fasta file of a genome assembly.
        hamming (int, optional): value for the maximum degeneration of the PAS.
            Defaults to 1.
    """

    def get_distances_per_utr(utr):
        adata_utr = adata[:, adata.var.utr_name == utr]

        pas_end, _, seq_strand = _find_pas(adata_utr, hamming=hamming, genome=genome)

        if not pas_end:
            return None

        # generate a vector of one UTR size per UMI count
        umi_utr_sizes = _utr_sizes_vector_from_read_counts(
            adata_utr.X.astype(int), adata_utr.var.utr_size)

        distances = _calculate_umi_distance_to_pas(umi_utr_sizes=umi_utr_sizes,
                                                   pas_end=pas_end,
                                                   seq_strand=seq_strand)
        return distances.to_numpy().flatten()

    utrs = adata.var.utr_name.unique()

    distances_list = joblib.Parallel(n_jobs=-2)(
        joblib.delayed(get_distances_per_utr)(utr=utr) for utr in tqdm(
            utrs, colour="#ffb700", ascii=" ■")
    )

    # remove None from the list
    distances_list = [i for i in distances_list if i is not None]

    distances_array = np.concatenate(distances_list)

    return distances_array


def _score_peak_assignment(adata, hamming=1, return_distances_to_hex=False):

    def get_distances(utr):
        adata_utr = adata[:, adata.var.utr_name == utr]
        pas_end, _, seq_strand = _find_pas(adata_utr, hamming=hamming)

        if not pas_end:
            return None

        # generate a vector of one UTR size per UMI count
        umi_utr_sizes = _utr_sizes_vector_from_read_counts(
            adata_utr.X.astype(int), adata_utr.var.utr_size)

        distances = _calculate_umi_distance_to_pas(umi_utr_sizes=umi_utr_sizes,
                                                   pas_end=pas_end,
                                                   seq_strand=seq_strand)

        distances_ = distances.to_numpy().flatten()

        total_umis = len(distances)
        threshold = int(math.ceil(total_umis * 0.01))
        distances_bigger_ten = distances[distances.groupby(level=0).count() > threshold]
        in_range_of_pas = distances_bigger_ten.between(10, 30)

        try:
            return [in_range_of_pas.value_counts().loc[True] / total_umis, distances_]
        except KeyError:
            return [0, distances_]

    utrs = adata.var.utr_name.unique()

    logger.info("Scoring peaks for each UTR")
    results_per_utr = joblib.Parallel(n_jobs=-2)(
        joblib.delayed(get_distances)(utr=utr) for utr in tqdm(utrs, colour="#ffb700",
                                                               ascii=" ■", desc="UTR: ")
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


def _find_pas(adata, genome, hamming=1):
    seq_chrom = adata.var.chrom[0]
    seq_start = adata.var.start[0] - 1
    seq_stop = adata.var.end[0]
    seq_strand = adata.var.strand[0]

    query = pybedtools.BedTool(
        f"{seq_chrom} {seq_start} {seq_stop} three_prime_utr 0 {seq_strand}", from_string=True)
    query = query.sequence(fi=genome, s=True)

    with open(query.seqfn) as q:
        next(q)
        utr_seq = q.read().strip()

    pas_seq = []
    pas_end = []
    start_cut = 0
    while True:
        if hamming > 0:
            expression = fr"(AATAAA){{s<={hamming}}}"
            hex = regex.compile(expression)
        elif hamming == 0:
            hex = regex.compile(r"(AATAAA)")
        else:
            raise ValueError(f"Hamming distance of {hamming} not supported")
        search = hex.search(utr_seq)
        try:
            span = search.span()
        except AttributeError:
            break
        end = span[1] + start_cut
        start_cut += span[0] + 1
        pas_end.append(end)
        pas_seq.append(search.group(0))
        utr_seq = utr_seq[span[0] + 1:]

    return(pas_end, pas_seq, seq_strand)


def _calculate_umi_distance_to_pas(umi_utr_sizes, pas_end, seq_strand):
    if len(pas_end) == 1:
        if seq_strand == "+":
            return umi_utr_sizes - pas_end
        elif seq_strand == "-":
            return (umi_utr_sizes - pas_end) * -1
        else:
            raise ValueError(f"Strand {seq_strand} not supported")
    else:
        distances = []
        for end in pas_end:
            if seq_strand == "+":
                distances.append((umi_utr_sizes - end))
            elif seq_strand == "-":
                distances.append((umi_utr_sizes - end) * -1)
            else:
                raise ValueError(f"Strand {seq_strand} not supported")
        distances = pd.concat(distances, axis=1, ignore_index=True)

        closest_to_pas = []
        for idx, row in distances.iterrows():
            is_between = np.where(row.between(10, 30))[0]
            if len(is_between) == 1:
                closest_to_pas.append(is_between[0])
            elif len(is_between) > 1:
                row_between = row.iloc[is_between]
                min_index = row_between.abs().idxmin()
                closest_to_pas.append(min_index)
            else:
                closest_to_pas.append(row.abs().idxmin())

        idx, cols = pd.factorize(closest_to_pas)

        return pd.Series(distances.reindex(cols, axis=1).to_numpy()[np.arange(len(distances)), idx], index=distances.index)


def _utr_sizes_vector_from_read_counts(counts_arr, sizes_arr):
    counts_sum = np.sum(counts_arr, axis=0)
    try:
        return np.repeat(sizes_arr, counts_sum)
    except ValueError:
        counts_sum = np.ravel(counts_sum)
        return np.repeat(sizes_arr, counts_sum)


def _call_peaks_using_kernel_density_estimator(utr_sizes_vector, bandwidth=1):
    """
    Fit a kernel density model to predict the peak's ditribution density function

    Args:
        utr_sizes_vector (np.ndarray): column vector with the UTR sizes to call peaks.

    Returns:
        np.ndarray: The vector with the peaks.
    """

    if len(utr_sizes_vector) <= 3:
        peaks = np.repeat(round(np.median(utr_sizes_vector)), len(utr_sizes_vector))
        return peaks

    if np.ndim(utr_sizes_vector) == 1:
        utr_sizes_vector = utr_sizes_vector.reshape(-1, 1)

    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(utr_sizes_vector)

    # calculate an appropriate window to check the model
    minimum_size, maximum_size = utr_sizes_vector.min(), utr_sizes_vector.max()
    padding = (maximum_size - minimum_size) * 0.1
    window = np.arange(round(minimum_size - padding),
                       round(maximum_size + padding) + 1)
    # get the model predictions for that window. Specifically, the extrema.
    evaluation = kde.score_samples(window.reshape(-1, 1))
    minima = argrelextrema(evaluation, np.less)[0]
    maxima = argrelextrema(evaluation, np.greater)[0]

    # create a container to hold the UTR peak sizes
    peaks = np.zeros(len(utr_sizes_vector))

    # attribute a peak to each cell according to their UTR sizes and the kde model:
    # first iterate over the function' minima
    idx_minimum = 0  # for the unbound warning... change it in the future!
    for idx_minimum, val in enumerate(window[minima]):
        # get the maximum upstream that minimum (the peak) and store it
        try:
            max_ = round(window[maxima][idx_minimum])
        except IndexError:
            continue
        for idx_maximum, cond in enumerate(utr_sizes_vector < val):
            if cond and peaks[idx_maximum] == 0:
                peaks[idx_maximum] = max_
    # get the last maximum, just downstream the last minimum.
    try:
        max_ = round(window[maxima][idx_minimum + 1])
    except IndexError:  # for those cases with just one peak, there will be no minimum
        # so we take the median as the peak
        max_ = round(np.median(utr_sizes_vector))

    # place the last maximum in those molecules that haven't yet had their
    # peaks scored
    peaks[peaks == 0] = max_
    return peaks
