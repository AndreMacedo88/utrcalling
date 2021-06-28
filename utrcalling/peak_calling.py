from functools import partial

import joblib
import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema
import anndata
from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)

from .helpers import flatten, get_number_cores
from .module_log import module_logger


def merge_samples_utr_sizes(utr_sizes):
    """
    Before calling peaks from position counts, we should merge the datasets we want to
    compare, so that peaks get called with the pseudo-bulk data from all samples.
    Merging appends both samples information for each UTR, while discarding UTRs that 
    only one sample mapped to.
    If you want to call peaks for a single sample, skip this function.

    Args:
        utr_positions_counts (list): List with the objects containing the position
        counts for each sample.

    Returns:
        UtrMappings: The merged datasets.
    """
    logger = module_logger()

    def add_counter_to_barcode(x, counter):
        x["read_barcodes"] = [
            f'{barcode}-{counter}' for barcode in x["read_barcodes"]]
        return x

    logger.info("Merging all samples' UTR sizes")

    # first we need to change the barcode names of our samples to identify each sample
    sample_counter = 1
    barcode_modified_utr_sizes = []
    for position_counts in tqdm(utr_sizes):
        changed = position_counts.filter(
            lambda x: x.each(add_counter_to_barcode, sample_counter))
        sample_counter += 1
        barcode_modified_utr_sizes.append(changed)

    # Before merging delete some big objects because of memory size issues
    del utr_sizes
    del changed

    # merge the two datasets
    position_counts_merged = \
        barcode_modified_utr_sizes[0].merge(barcode_modified_utr_sizes[1])

    return position_counts_merged


def subset_reference_data_2(utr_sizes, utr_reference_regions):
    logger = module_logger()
    logger.info('Subsetting UTR reference data')
    sub_utr_reference = utr_reference_regions.filter(
        lambda x: x.name in utr_sizes.utr_ids).saveas()
    return sub_utr_reference.to_dataframe()


def subset_reference_data(utr_sizes, utr_reference_regions):
    logger = module_logger()
    logger.info('Subsetting UTR reference data')
    sub_utr_reference = utr_reference_regions.filter(items=utr_sizes.utr_ids, axis=0)
    return sub_utr_reference


def filter_utr_sizes(utr_sizes):
    logger = module_logger()
    logger.info('Filtering samples UTR sizes')

    def filter_func(x, amount):
        condition_1 = [True if i >
                       amount else False for i in x['utr_sizes_from_reads']]
        # condition_2 = [True if i == amount else False for i in x['read_strands']]
        # condition_final = [all(items) for items in zip(condition_1, condition_2)]
        return condition_1

    filtered = utr_sizes.filter(
        lambda x: x.filter(filter_func, 1))
    return filtered


def filter_barcodes(utr_sizes, annotations, parallel=False):

    def filter_and_correct_barcodes(utr, barcodes_dict):
        barcodes = utr['read_barcodes']

        new_barcodes = []
        for barcode in barcodes:
            new_barcodes.append(barcodes_dict.get(barcode, False))

        filtered = utr.copy(deep=True)
        for key, values in filtered.__dict__.items():
            if key == 'utr_name':
                continue
            filtered[key] = [b for a, b in zip(new_barcodes, values) if a]
        filtered['read_barcodes'] = list(filter(None, new_barcodes))
        return filtered

    logger = module_logger()
    logger.info('Filtering cell barcodes')

    if isinstance(annotations, (pd.DataFrame, pd.Series)):
        annotations_df = annotations
    else:
        annotations_df = pd.read_csv(annotations)

    # Note that barcodes must be on the first column of the annotations file:
    barcodes_real = annotations_df.iloc[:, 0].tolist()
    barcodes_to_test = set(flatten(utr_sizes.read_barcodes))

    barcodes_dict = {}
    for barcode in barcodes_to_test:
        if barcode in barcodes_real:
            barcodes_dict[barcode] = barcode
        else:
            match = process.extractOne(
                barcode, barcodes_real, scorer=fuzz.ratio, score_cutoff=90)
            if match is not None:
                barcodes_dict[barcode] = match[0]
            else:
                continue

    utr_positions_filtered = utr_sizes.filter(
        lambda x: x.each(filter_and_correct_barcodes, barcodes_dict), parallel=parallel)
    return utr_positions_filtered


def call_peaks_using_kernel_density_estimator(utr_sizes_vector):
    """
    Fit a kernel density model to predict the peak's ditribution density function

    Args:
        utr_sizes_vector (np.ndarray): column vector with the UTR sizes to call peaks.

    Returns:
        np.ndarray: The vector with the peaks.
    """

    kde = KernelDensity(kernel='gaussian', bandwidth=1).fit(utr_sizes_vector)

    # calculate an appropriate window to check the model
    minimum_size, maximum_size = utr_sizes_vector.min(), utr_sizes_vector.max()
    padding = (maximum_size - minimum_size) * 0.1
    window = np.linspace(minimum_size - padding, maximum_size + padding)
    # get the model predictions for that window. Specifically, the maxima.
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
        max_ = round(window[maxima][idx_minimum])
        for idx_maximum, cond in enumerate(utr_sizes_vector < val):
            if cond and peaks[idx_maximum] == 0:
                peaks[idx_maximum] = max_
    # get the last maximum, just downstream the last minimum.
    try:
        max_ = round(window[maxima][idx_minimum + 1])
    except IndexError:  # for those cases with just one peak, there will be no minimum
        # so we take the median as the peak
        max_ = np.median(utr_sizes_vector)
    # attribute the last maximum to those molecules that haven't yet had their
    # peaks scored
    peaks[peaks == 0] = max_
    return peaks


def generate_peak_counts(utr_sizes, annotations=None,
                         func=call_peaks_using_kernel_density_estimator):

    logger = module_logger()
    logger.info('Generating and counting UTR size peaks')
    # get list of the unique barcodes (cells)
    cell_barcodes = set(flatten(utr_sizes.read_barcodes))
    # open the annotations file
    if annotations is not None:
        if isinstance(annotations, (pd.DataFrame, pd.Series)):
            pass
        else:
            annotations = pd.read_csv(annotations, engine='c')

    # iterate over each UTR, to call peaks corresponding to the UTR's different sizes.
    cores = get_number_cores()
    count_matrix_list, peaks_var_list = zip(*joblib.Parallel(n_jobs=cores)(
        joblib.delayed(call_peaks)(utr=utr, func=func,
                                   cell_barcodes=cell_barcodes) for utr in tqdm(
            utr_sizes, desc='Calling peaks for each UTR')))

    count_matrix = pd.concat(count_matrix_list, axis=0, ignore_index=False)
    peaks_var = pd.concat(peaks_var_list, axis=0, ignore_index=False)

    count_matrix = count_matrix.T
    if annotations is not None:
        peaks_obs = annotations.set_index(annotations.columns[0])
        peaks_obs = peaks_obs.reindex(index=count_matrix.index)
        return count_matrix, peaks_var, peaks_obs

    return count_matrix, peaks_var


def call_peaks(utr, func, cell_barcodes):
    # create containers to append the data
    count_matrix = pd.DataFrame(columns=cell_barcodes)
    peaks_var = pd.DataFrame()

    info = pd.DataFrame.from_dict(utr.__dict__)
    # similar UMIs refer to the same event (molecule), and so we take the median
    info['utr_sizes_from_reads'] = info.groupby(['read_barcodes', 'read_umis'])[
        'utr_sizes_from_reads'].transform('median')
    info = info.drop_duplicates()

    utr_sizes_vector = info['utr_sizes_from_reads'].values.reshape(-1, 1)

    # now we call the utr size peaks based on the distribution of individual UTR sizes
    peaks = func(utr_sizes_vector)

    # store this info in a new key on the original container
    info['peaks'] = peaks

    # group by each peak size and cell and count the number of each peak each cell has
    info_grouped_by_peaks = \
        info.groupby(['peaks', 'read_barcodes'])['peaks'].agg('count')

    # append this information to the containers created above
    count = 1
    for utr_size_at_peak, barcodes in info_grouped_by_peaks.groupby(level=0):
        # generate a new name for each peak. Here, it's UTR name (which is gene name
        # plus a counter for each transcript UTR) plus a counter for each peak
        peak_name = f'{utr.utr_name}-{count}'
        count += 1
        # update the peak observations table
        peaks_var.loc[peak_name, 'utr_size'] = utr_size_at_peak
        # update the peak count matrix
        peak_counts = barcodes.droplevel(0)
        peak_counts.name = peak_name
        count_matrix = count_matrix.append(peak_counts, ignore_index=False)
        count_matrix = count_matrix.fillna(0)
    return count_matrix, peaks_var


def add_to_peaks_var(peak_var, annotations):
    new_peaks_var = pd.DataFrame()
    for index, utr in annotations.iterrows():
        peak_var_2 = peak_var[peak_var.index.str.startswith(index)]
        for index_2, peak in peak_var_2.iterrows():
            new_row = peak.append(utr, ignore_index=False)
            new_row.name = index_2
            new_peaks_var = new_peaks_var.append(new_row, ignore_index=False)
    return new_peaks_var


def generate_count_matrix_anndata(count_matrix, var, obs=None):
    logger = module_logger()
    logger.info('Generating Count Matrix')
    count_matrix_anndata = anndata.AnnData(count_matrix, var=var, obs=obs)
    return count_matrix_anndata

# LEGACY FUNCTIONS


"""
def filter_barcodes_parallel(utr_sizes, annotations):
    
    def filter_and_correct_barcodes(utr, barcodes_dict):
        barcodes = utr['read_barcodes']
        
        new_barcodes = []
        for barcode in barcodes:
            new_barcodes.append(barcodes_dict.get(barcode, False))

        filtered = utr.copy(deep=True)
        for key, values in filtered.__dict__.items():
            if key == 'utr_name':
                continue
            filtered[key] = [b for a, b in zip(new_barcodes, values) if a]
        filtered['read_barcodes'] = list(filter(None, new_barcodes))
        return filtered
    
    logger = module_logger()
    logger.info('Filtering cell barcodes')

    if isinstance(annotations, (pd.DataFrame, pd.Series)):
        annotations_df = annotations
    else:
        annotations_df = pd.read_csv(annotations)
    
    # Note that barcodes must be on the first column of the annotations file:
    barcodes_real = annotations_df.iloc[:,0].tolist()
    barcodes_to_test = set(flatten(utr_sizes.read_barcodes))
    
    cores = get_number_cores()
    barcodes_dict_list = joblib.Parallel(n_jobs=cores)(joblib.delayed(
        create_barcode_dict)(barcode=barcode, barcodes_real=barcodes_real
        ) for barcode in barcodes_to_test)
    
    barcodes_dict = {k: v for d in barcodes_dict_list if d is not None for k, v in d.items()}


    utr_positions_filtered = utr_sizes.filter(
        lambda x: x.each(filter_and_correct_barcodes, barcodes_dict))
    return utr_positions_filtered

def create_barcode_dict(barcode, barcodes_real):
    barcodes_dict = {}
    if barcode in barcodes_real:
        barcodes_dict[barcode] = barcode
    else:
        match = process.extractOne(
            barcode, barcodes_real, scorer=fuzz.ratio, score_cutoff=90)
        if match is not None:
            barcodes_dict[barcode] = match[0]
        else:
            return None
    return barcodes_dict

def filter_barcodes_3(utr_sizes, annotations):
    logger = module_logger()
    logger.info('Filtering cell barcodes')
    def filter_and_correct_barcodes(utr, barcodes_real):
        barcodes = utr['read_barcodes']
        if set(barcodes).issubset(barcodes_real):
            return utr
        else:
            real_names = []
            mask = []
            for barcode in barcodes:
                match = process.extractOne(
                    barcode, barcodes_real, scorer=fuzz.ratio, score_cutoff=90)
                if match is not None:
                    real_names.append(match[0])
                    mask.append(True)
                else:
                    mask.append(False)

            filtered = utr.copy(deep=True)
            for key, values in filtered.__dict__.items():
                if key == 'utr_name':
                    continue
                filtered[key] = [b for a, b in zip(mask, values) if a]
            filtered['read_barcodes'] = real_names
            return filtered
    
    if isinstance(annotations, (pd.DataFrame, pd.Series)):
        annotations_df = annotations
    else:
        annotations_df = pd.read_csv(annotations)
    # Note that barcodes must be on the first column of the annotations file:
    barcodes_real = annotations_df.iloc[:,0].tolist()
    utr_positions_filtered = utr_sizes.filter(
        lambda x: x.each(filter_and_correct_barcodes, barcodes_real))
    return utr_positions_filtered

def generate_peak_counts_not_parallel(utr_sizes, annotations=None, 
    func=call_peaks_using_kernel_density_estimator):
    
    logger = module_logger()
    logger.info('Generating and counting UTR size peaks')
    # get list of the unique barcodes (cells)
    cell_barcodes = set(flatten(utr_sizes.read_barcodes))
    # create containers for the desired data
    count_matrix = pd.DataFrame(columns=cell_barcodes)
    peaks_var = pd.DataFrame()
    # open the annotations file
    if annotations is not None:
        if isinstance(annotations, (pd.DataFrame, pd.Series)):
            pass
        else:
            annotations = pd.read_csv(annotations, engine='c')

    # iterate over each UTR, to call peaks corresponding to the UTR's different sizes.
    for utr in tqdm(utr_sizes, desc='Calling peaks for each UTR'):
        info = pd.DataFrame.from_dict(utr.__dict__)
        # similar UMIs refer to the same event (molecule), and so we take the median
        info['utr_sizes_from_reads'] = info.groupby(['read_barcodes', 'read_umis'])\
            ['utr_sizes_from_reads'].transform('median')
        info = info.drop_duplicates()

        utr_sizes_vector = info['utr_sizes_from_reads'].values.reshape(-1, 1)

        # now we call the utr size peaks based on the distribution of individual UTR sizes
        peaks = func(utr_sizes_vector)

        # store this info in a new key on the original container
        info['peaks'] = peaks

        # group by each peak size and cell and count the number of each peak each cell has
        info_grouped_by_peaks = \
            info.groupby(['peaks', 'read_barcodes'])['peaks'].agg('count')
        
        # append this information to the containers created above
        count = 1
        for utr_size_at_peak, barcodes in info_grouped_by_peaks.groupby(level=0):
            # generate a new name for each peak. Here, it's UTR name (which is gene name
            # plus a counter for each transcript UTR) plus a counter for each peak
            peak_name = f'{utr.utr_name}-{count}'
            count += 1
            # update the peak observations table
            peaks_var.loc[peak_name, 'utr_size'] = utr_size_at_peak
            # update the peak count matrix
            peak_counts = barcodes.droplevel(0)
            peak_counts.name = peak_name
            count_matrix = count_matrix.append(peak_counts, ignore_index=False)
            count_matrix = count_matrix.fillna(0)
    count_matrix = count_matrix.T
    if annotations is not None:
        peaks_obs = annotations.set_index(annotations.columns[0])
        peaks_obs = peaks_obs.reindex(index=count_matrix.index)
        return count_matrix, peaks_var, peaks_obs
    
    return count_matrix, peaks_var
"""
