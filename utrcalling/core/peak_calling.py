import gc
from functools import partial, reduce
import re
import psutil
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import joblib
from joblib.externals.loky import get_reusable_executor
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

    """
    if label_batches:
        for idx, batch in enumerate(utr_sizes):
            batch.read_barcodes = [
                [f"{barcode}-{idx+1}" for barcode in barcodes]
                for barcodes in batch.read_barcodes
            ]
    """

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


def filter_barcodes(utr_sizes, annotations):
    logger = module_log.module_logger()
    logger.info('Filtering cell barcodes')

    if isinstance(annotations, (pd.DataFrame, pd.Series)):
        annotations_df = annotations
    else:
        annotations_df = pd.read_csv(annotations)

    # Note that barcodes must be on a column with barcode somewhere in its name:
    barcodes_column_name = annotations_df.columns.str.extract(
        r"(.*barcode.*)", flags=re.IGNORECASE).dropna().values[0]
    barcodes_real = annotations_df.loc[:, barcodes_column_name[0]].tolist()

    barcodes_to_test = set(helpers.flatten(utr_sizes.read_barcodes))

    barcodes_correct = set()
    barcodes_replace = {}
    for barcode in barcodes_to_test:
        if barcode in barcodes_real:
            barcodes_correct.add(barcode)
        else:
            try:
                match = process.extract(
                    barcode, barcodes_real, scorer=distance.Hamming.distance, score_cutoff=1)
            except ValueError as err:
                print(f"Error matching sample barcode/ID: {barcode}\n Please check if \
the barcodes in the annotation file are correct or, if using different batches with \
similar barcodes/IDs, if all the batches are correctly described in the annotations file."
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
            else:
                continue

    utr_sizes = utr_sizes.explode(utr_sizes.columns.to_list())
    utr_sizes.read_barcodes.replace(barcodes_replace, inplace=True)
    utr_sizes = utr_sizes[utr_sizes.read_barcodes.isin(barcodes_correct)]

    return utr_sizes.groupby(level=0).agg(list)


def generate_peak_counts(utr_sizes, annotations=None,
                         func=protocols.Protocol._call_peaks_using_kernel_density_estimator):

    logger = module_log.module_logger()
    logger.info('Generating and counting UTR size peaks')

    # open the annotations file
    if annotations is not None:
        if isinstance(annotations, (pd.DataFrame, pd.Series)):
            pass
        else:
            annotations = pd.read_csv(annotations, engine='c')

    """# uncomment this block to generate data in chunks
    n_chunks = 3
    chunk = 3
    logger.debug(f"Dividing the data into {n_chunks} chunks and processing chunk {chunk}")
    a = list(helpers.split_list(range(len(utr_sizes)), n_chunks))[chunk - 1]
    slicer = slice(a.start, a.stop, a.step)
    utr_sizes = utr_sizes[utr_sizes[slicer]]
    """

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

    count_matrix = pd.concat(count_matrix_list, axis=1, ignore_index=False, copy=False)
    del count_matrix_list  # Free some RAM
    count_matrix = count_matrix.fillna(0).astype("int32")

    peaks_var = pd.concat(peaks_var_list, axis=0, ignore_index=False)
    if annotations is not None:
        logger.info('Adding annotations')
        barcodes_column_name = annotations.columns.str.extract(
            r"(.*barcode.*)", flags=re.IGNORECASE).dropna().values[0]
        peaks_obs = annotations.set_index(barcodes_column_name[0])
        peaks_obs = peaks_obs.reindex(index=count_matrix.index)
        return count_matrix, peaks_var, peaks_obs

    return count_matrix, peaks_var


def call_peaks(utr, func):

    # create containers to append the data
    count_matrix = {}
    peaks_var = pd.DataFrame()

    info = utr[1].to_frame().T
    info = info.explode(column=info.columns.tolist(), ignore_index=True)
    info = info.sort_values(by="utr_sizes_from_reads")

    """
    # similar UMIs refer to the same event (molecule), and so we take the mode
    info['utr_sizes_from_reads'] = info.groupby(['read_barcodes', 'read_umis'])[
        'utr_sizes_from_reads'].transform(mode)
    info = info.drop_duplicates()
    """

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
        peaks_var.loc[peak_name, "utr_size"] = utr_size_at_peak
        # update the peak count matrix
        peak_counts = barcodes.droplevel(0)
        count_matrix[peak_name] = peak_counts

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

    count_matrix_anndata = anndata.AnnData(count_matrix, var=var, obs=obs,
                                           dtype=count_matrix.values.dtype)
    return count_matrix_anndata


# LEGACY FUNCTIONS

"""

def call_peaks(utr, func, cell_barcodes):

    # create containers to append the data
    count_matrix = pd.DataFrame(columns=cell_barcodes)
    peaks_var = pd.DataFrame()

    info = pd.DataFrame.from_dict(utr.__dict__)

    # similar UMIs refer to the same event (molecule), and so we take the median
    info['utr_sizes_from_reads'] = info.groupby(['read_barcodes', 'read_umis'])[
        'utr_sizes_from_reads'].transform(mode)
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

def filter_barcodes_old(utr_sizes, annotations, parallel=False):

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

    logger = module_log.module_logger()
    logger.info('Filtering cell barcodes')

    if isinstance(annotations, (pd.DataFrame, pd.Series)):
        annotations_df = annotations
    else:
        annotations_df = pd.read_csv(annotations)

    # Note that barcodes must be on the first column of the annotations file:
    barcodes_real = annotations_df.iloc[:, 0].tolist()
    barcodes_to_test = set(helpers.flatten(utr_sizes.read_barcodes))

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

def merge_samples_utr_sizes_old(utr_sizes, method='intersection', label_samples=True,
                                exclude=[]):
    logger = module_log.module_logger()

    def add_counter_to_barcode(x, counter):
        x["read_barcodes"] = [
            f'{barcode}-{counter}' for barcode in x["read_barcodes"]]
        return x

    logger.info("Merging all samples' UTR sizes")

    sample_counter = 1
    position_counts_merged = None
    for position_counts in tqdm(utr_sizes, colour="#ffb700", ascii=" ■"):
        # account for those instances where utr_sizes are a list of paths
        position_counts = helpers.check_if_path_and_open(position_counts, method='pickle')
        if label_samples:
            # add a counter to the barcode to separate samples from different sources
            position_counts_to_merge = position_counts.filter(
                lambda x: x.each(add_counter_to_barcode, sample_counter))
        else:
            position_counts_to_merge = position_counts

        # merge the datasets
        if position_counts_merged is not None:
            position_counts_merged = position_counts_merged.merge(
                position_counts_to_merge, method=method,
                exclude=exclude)
        else:
            position_counts_merged = position_counts_to_merge
        sample_counter += 1

    return position_counts_merged

def add_to_peaks_var_2(peak_var, annotations):
    logger = module_log.module_logger()
    logger.info('Adding reference annotations to the final result')

    new_peaks_var = pd.DataFrame()
    for index, utr in annotations.iterrows():
        peak_var_2 = peak_var[peak_var.index.str.match(f'{index}(?=[$|-])')]
        for index_2, peak in peak_var_2.iterrows():
            new_row = peak.append(utr, ignore_index=False)
            new_row.name = index_2
            new_peaks_var = new_peaks_var.append(new_row, ignore_index=False)
    return new_peaks_var

def filter_utr_sizes(utr_sizes, amount=1):
    logger = module_log.module_logger()
    logger.info('Filtering samples UTR sizes')

    def filter_func(x, amount_):
        condition_1 = [True if i >
                       amount_ else False for i in x['utr_sizes_from_reads']]
        # condition_2 = [True if i == amount_ else False for i in x['read_strands']]
        # condition_final = [all(items) for items in zip(condition_1, condition_2)]
        return condition_1

    filtered = utr_sizes.filter(
        lambda x: x.filter(filter_func, amount))
    return filtered

def merge_samples_utr_sizes_2(utr_sizes):
    "
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
    "
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
