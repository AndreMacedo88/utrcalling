import os
import re
from functools import partial, reduce
from tempfile import mkstemp
# import gc
# import psutil
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)  # silence modin UserWarnings

# import joblib
# from joblib.externals.loky import get_reusable_executor
import modin.pandas as mpd
import pandas as pd
from rapidfuzz import process, distance
import anndata
from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)

from utrcalling.core import helpers, module_log, protocols


def merge_samples_utr_sizes(utr_sizes, method='intersection', label_batches=True):
    """
    Before calling peaks from position counts, we should merge the datasets we want to
    compare, so that peaks get called with the pseudo-bulk data from all samples.
    Merging appends both samples information for each UTR. If `method='intersection'`,
    UTRs that only one sample mapped to will be discarded. If `method='union'`, then
    a union merge is performed.
    If you want to call peaks for a single sample, skip this function.

    Args:
        utr_sizes (list): List with the objects containing the position
        counts for each sample.

    Returns:
        pd.Dataframe: An object with the merged datasets.
    """

    logger = module_log.module_logger()

    if label_batches:
        for idx, batch in enumerate(utr_sizes):
            batch.read_barcodes = [
                [f"{barcode}-{idx+1}" for barcode in barcodes]
                for barcodes in batch.read_barcodes
            ]

    logger.info("Merging all batches' UTR sizes")
    if method == "union":
        utr_sizes = pd.concat(utr_sizes)
        utr_sizes = utr_sizes.groupby(level=0).sum()
    elif method == "intersection":
        utr_sizes = reduce(lambda a, b: (a + b).dropna(axis=0), utr_sizes)
    else:
        raise AttributeError(f"Method {method} not supported.")

    return utr_sizes


def subset_reference_data(utr_sizes, utr_reference_regions):
    logger = module_log.module_logger()
    logger.info('Subsetting UTR reference data')

    sub_utr_reference = utr_reference_regions.filter(
        items=utr_sizes.index, axis=0)

    sub_utr_reference_unique = sub_utr_reference.drop_duplicates()

    return sub_utr_reference_unique


def filter_barcodes(utr_sizes, whitelist):
    logger = module_log.module_logger()
    logger.info('Filtering cell barcodes')

    barcodes_to_test = set(helpers.flatten(utr_sizes.read_barcodes))
    barcodes_correct = set()
    barcodes_replace = {}
    for barcode in barcodes_to_test:
        if barcode in whitelist:
            barcodes_correct.add(barcode)
        else:
            try:
                match = process.extract(
                    barcode, whitelist, scorer=distance.Hamming.distance,
                    score_cutoff=1
                )
            except ValueError as err:
                print(
                    f"Error matching sample barcode/ID: {barcode}\nPlease check if "
                    f"the barcodes in the annotation file are correct or, if using "
                    f"different batches with similar barcodes/IDs, if all the batches "
                    f"are correctly described in the annotations file."
                )
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

    utr_sizes = utr_sizes.explode(utr_sizes.columns.to_list())
    utr_sizes.read_barcodes.replace(barcodes_replace, inplace=True)
    utr_sizes = utr_sizes[utr_sizes.read_barcodes.isin(barcodes_correct)]

    return utr_sizes.groupby(level=0).agg(list)


def generate_peak_counts(
    utr_sizes,
    annotations,
    func=protocols.Protocol._call_peaks_using_kernel_density_estimator
):

    logger = module_log.module_logger()
    logger.info('Generating and counting UTR size peaks')

    if not isinstance(utr_sizes, mpd.DataFrame):
        fd, df_path = mkstemp(suffix=".feather", dir=".")
        utr_sizes.index.name = "utr_name"
        utr_sizes = utr_sizes.reset_index()
        utr_sizes.to_feather(df_path)
        utr_sizes = mpd.read_feather(df_path)
        os.remove(df_path)
        utr_sizes = utr_sizes.set_index("utr_name")

    # iterate over each UTR, to call peaks corresponding to the UTR's different sizes.
    logger.info("Calling peaks for each UTR")

    count_matrix_list, peaks_var_list = zip(
        *utr_sizes.apply(call_peaks, axis=1, func_=func))

    logger.info('Finished generating and counting UTR size peaks')

    # Free some RAM by deleting biggest variables, collecting the garbage, and shutting
    # down joblib inactive workers
    del utr_sizes

    # gc.collect()
    # try:
    #     get_reusable_executor().shutdown(wait=True)
    # except PermissionError:
    #     pass
    # if psutil.virtual_memory().percent > 50:
    #     for ct_mtx in count_matrix_list:
    #         ct_mtx.astype("int32", copy=False, errors="ignore")

    count_matrix = pd.concat(count_matrix_list, axis=1,
                             ignore_index=False, copy=False)
    del count_matrix_list  # Free some RAM
    count_matrix = count_matrix.fillna(0).astype("uint16")

    peaks_var = pd.concat(peaks_var_list, axis=0, ignore_index=False)

    logger.info('Adding annotations')
    barcodes_column_name = annotations.columns.str.extract(
        r"(.*barcode.*)", flags=re.IGNORECASE).dropna().values[0]
    peaks_obs = annotations.set_index(barcodes_column_name[0])
    peaks_obs = peaks_obs.reindex(index=count_matrix.index)
    return count_matrix, peaks_var, peaks_obs


def call_peaks(utr, func_):
    # create containers to append the data
    count_matrix = {}
    peaks_var = {}

    info = utr.to_frame().T
    info = info.explode(column=info.columns.tolist(), ignore_index=True)
    info = info.sort_values(by="utr_sizes_from_reads")

    utr_sizes_vector = info["utr_sizes_from_reads"].values.reshape(-1, 1)
    # now we call the utr size peaks based on the distribution of individual UTR sizes
    peaks = func_(utr_sizes_vector)
    # store this info in a new key on the original container
    info["peaks"] = peaks
    # group by each peak size and cell and count the number of each peak each cell has
    info_grouped_by_peaks = info.groupby(["peaks", "read_barcodes"])["peaks"].agg("count")

    # append this information to the containers created above
    count = 1
    for utr_size_at_peak, barcodes in info_grouped_by_peaks.groupby(level=0):
        # generate a new name for each peak. Here, it's UTR name (which is gene name
        # plus a counter for each transcript UTR) plus a counter for each peak
        peak_name = f'{utr.name}-{count}'
        count += 1
        # update the peak observations table
        peaks_var[peak_name] = utr_size_at_peak
        # update the peak count matrix
        peak_counts = barcodes.droplevel(0)
        count_matrix[peak_name] = peak_counts

    peaks_var = pd.DataFrame.from_dict(peaks_var, orient="index", columns=["utr_size"])
    count_matrix = pd.DataFrame(count_matrix)
    return count_matrix, peaks_var


def add_to_peaks_var(peak_var, annotations):
    logger = module_log.module_logger()
    logger.info('Adding reference annotations to the final result')

    annotations['utr_name'] = annotations.index

    peak_var_new = peak_var.copy(deep=True)
    peak_var_new['utr_name'] = peak_var_new.index.str.split(pat=r"-").str[0]

    peak_var_new = peak_var_new.join(
        annotations, on='utr_name', how='left', rsuffix='_right')

    peak_var_new = peak_var_new.drop('utr_name_right', axis=1)

    return peak_var_new


def generate_count_matrix_anndata(count_matrix, var, obs=None):
    logger = module_log.module_logger()
    logger.info('Generating Count Matrix')

    count_matrix_anndata = anndata.AnnData(count_matrix.to_numpy(), var=var, obs=obs,
                                           dtype=count_matrix.values.dtype)
    return count_matrix_anndata


# LEGACY FUNCTIONS
"""
def generate_peak_counts(
    utr_sizes,
    annotations,
    func=protocols.Protocol._call_peaks_using_kernel_density_estimator
):

    logger = module_log.module_logger()
    logger.info('Generating and counting UTR size peaks')

    '''# uncomment this block to generate data in chunks
    n_chunks = 3
    chunk = 3
    logger.debug(f"Dividing the data into {n_chunks} chunks and processing chunk {chunk}")
    a = list(helpers.split_list(range(len(utr_sizes)), n_chunks))[chunk - 1]
    slicer = slice(a.start, a.stop, a.step)
    utr_sizes = utr_sizes[utr_sizes[slicer]]
    '''

    # iterate over each UTR, to call peaks corresponding to the UTR's different sizes.
    logger.info("Calling peaks for each UTR")
    cores = helpers.get_number_cores(left_out=2)
    utr_sizes_iter = utr_sizes.iterrows()
    count_matrix_list, peaks_var_list = zip(*joblib.Parallel(n_jobs=cores)(
        joblib.delayed(call_peaks)(utr=utr, func=func) for utr in tqdm(
            utr_sizes_iter, total=utr_sizes.shape[0], colour="#ffb700",
            ascii=" ■", desc='UTR: ')))

    logger.info('Finished generating and counting UTR size peaks')

    # Free some RAM by deleting biggest variables, collecting the garbage, and shutting
    # down joblib inactive workers
    del utr_sizes
    gc.collect()
    try:
        get_reusable_executor().shutdown(wait=True)
    except PermissionError:
        pass

    if psutil.virtual_memory().percent > 50:
        for ct_mtx in count_matrix_list:
            ct_mtx.astype("int32", copy=False, errors="ignore")

    count_matrix = pd.concat(count_matrix_list, axis=1,
                             ignore_index=False, copy=False)
    del count_matrix_list  # Free some RAM
    count_matrix = count_matrix.fillna(0).astype("int32")

    peaks_var = pd.concat(peaks_var_list, axis=0, ignore_index=False)

    logger.info('Adding annotations')
    barcodes_column_name = annotations.columns.str.extract(
        r"(.*barcode.*)", flags=re.IGNORECASE).dropna().values[0]
    peaks_obs = annotations.set_index(barcodes_column_name[0])
    peaks_obs = peaks_obs.reindex(index=count_matrix.index)
    return count_matrix, peaks_var, peaks_obs


def call_peaks(utr, func):
    # create containers to append the data
    count_matrix = {}
    peaks_var = {}

    info = utr[1].to_frame().T
    info = info.explode(column=info.columns.tolist(), ignore_index=True)
    info = info.sort_values(by="utr_sizes_from_reads")

    '''
    # similar UMIs refer to the same event (molecule), and so we take the mode
    info['utr_sizes_from_reads'] = info.groupby(['read_barcodes', 'read_umis'])[
        'utr_sizes_from_reads'].transform(mode)
    info = info.drop_duplicates()
    '''

    utr_sizes_vector = info["utr_sizes_from_reads"].values.reshape(-1, 1)
    # now we call the utr size peaks based on the distribution of individual UTR sizes
    peaks = func(utr_sizes_vector)
    # store this info in a new key on the original container
    info["peaks"] = peaks
    # group by each peak size and cell and count the number of each peak each cell has
    info_grouped_by_peaks = info.groupby(["peaks", "read_barcodes"])["peaks"].agg("count")

    # append this information to the containers created above
    count = 1
    for utr_size_at_peak, barcodes in info_grouped_by_peaks.groupby(level=0):
        # generate a new name for each peak. Here, it's UTR name (which is gene name
        # plus a counter for each transcript UTR) plus a counter for each peak
        peak_name = f'{utr[0]}-{count}'
        count += 1
        # update the peak observations table
        peaks_var[peak_name] = utr_size_at_peak
        # update the peak count matrix
        peak_counts = barcodes.droplevel(0)
        count_matrix[peak_name] = peak_counts
    # # append this information to the containers created above
    # count = 1
    # for utr_size_at_peak, barcodes in info_grouped_by_peaks.groupby(level=0):
    #     # generate a new name for each peak. Here, it's UTR name (which is gene name
    #     # plus a counter for each transcript UTR) plus a counter for each peak
    #     peak_name = f'{utr[0]}-{count}'
    #     count += 1
    #     # update the peak observations table
    #     peaks_var.loc[peak_name, "utr_size"] = utr_size_at_peak
    #     # update the peak count matrix
    #     peak_counts = barcodes.droplevel(0)
    #     count_matrix[peak_name] = peak_counts

    peaks_var = pd.DataFrame.from_dict(peaks_var, orient="index", columns=["utr_size"])
    count_matrix = pd.DataFrame(count_matrix)
    return count_matrix, peaks_var
"""
