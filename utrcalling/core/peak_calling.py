"""
peak_calling.py - module with functions to deal with calling peaks
"""

import os
import warnings
from tempfile import mkstemp
import pandas as pd
import modin.pandas as mpd
from modin.utils import to_pandas
import anndata
from utrcalling.core import helpers, module_log
warnings.filterwarnings("ignore", category=FutureWarning)
# silence modin UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)


def generate_peak_counts_and_anno(
    utr_sizes,
    protocol,
):
    logger = module_log.module_logger()
    helpers.init_ray()

    annotations = protocol.annotations
    func = protocol.peak_calling

    logger.info('Generating and counting UTR size peaks')

    if not isinstance(utr_sizes, mpd.DataFrame):
        _, df_path = mkstemp(suffix=".feather", dir=".")
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

    # Free some RAM by deleting biggest variables
    del utr_sizes

    count_matrix_list = [
        count_mtx.fillna(0).apply(
            mpd.to_numeric, downcast="unsigned"
        ) for count_mtx in count_matrix_list
    ]
    count_matrix = pd.concat(count_matrix_list, axis=1,
                             ignore_index=False, copy=False)
    del count_matrix_list  # Free some RAM
    count_matrix = count_matrix.fillna(0).apply(
        mpd.to_numeric, downcast="unsigned")

    peaks_var = pd.concat(peaks_var_list, axis=0, ignore_index=False)

    logger.info('Adding annotations')
    if protocol.barcode_column_name:
        peaks_obs = annotations.set_index(protocol.barcode_column_name)
        peaks_obs = peaks_obs.reindex(index=count_matrix.index)
        return count_matrix, peaks_var, peaks_obs
    else:
        return count_matrix, peaks_var


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
    info_grouped_by_peaks = info.groupby(["peaks", "read_barcodes"])[
        "peaks"].agg("count")

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

    peaks_var = pd.DataFrame.from_dict(
        peaks_var, orient="index", columns=["utr_size"])
    count_matrix = pd.DataFrame(count_matrix)

    return count_matrix, peaks_var


def add_to_peaks_var(peak_var, annotations):
    logger = module_log.module_logger()
    logger.info('Adding reference annotations to the final result')

    annotations['utr_name'] = annotations.index

    peak_var_new = peak_var.copy(deep=True)
    peak_var_new['utr_name'] = peak_var_new.index.str.split(pat=r"-").str[0]

    # Modin join does not allow joins not using the index.
    # so unfortunately for now we must revert back to pandas to use all columns
    if isinstance(annotations, mpd.dataframe.DataFrame):
        annotations = to_pandas(annotations)
    if isinstance(peak_var_new, mpd.dataframe.DataFrame):
        annotations = to_pandas(annotations)

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
