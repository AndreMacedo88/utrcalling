"""
# Goal
Mapping RNA-seq reads (single cell or bulk) to UTRs.

## Secondary goal
Call consensus peaks based on the distribution of molecule sizes.

# Summary
This file contains 3 functions for the 3 main steps in this procedure.

## General steps
1. Extract the UTR regions of transcripts (from ENSEMBL)
2. Map the aligned, filtered reads to the UTR regions from step 1.
3. Clean-up the data and call consensus peaks if required.

## More detailed steps
1.1. Resolve overlapping transcripts so that every entry stores unique, non-overlapping, 
genomic coordinates

# Observations
- Final object is a count matrix in AnnData format.
- Output should inherit single cell annotations, as well as gene annotation like the 
genomic coordinates and optionally UTR sequence information.
"""

import os
import re
import psutil

import numpy as np
import pandas as pd
import pybedtools
import joblib
from tqdm import tqdm

from utrcalling.core import helpers
from utrcalling.core.utr_reference import (
    generate_utr_name, extract_feature, prepare_annotations_index, reduce_regions,
    generate_utr_name, correct_column_names
)
from utrcalling.core.utr_mapping import generate_utr_intervals, map_reads_to_utr, module_logger
from utrcalling.core.peak_calling import (
    subset_reference_data, merge_samples_utr_sizes, filter_barcodes,
    generate_peak_counts, add_to_peaks_var, generate_count_matrix_anndata
)


def prepare_utr_reference_regions(gtf,
                                  feature,
                                  store=False
                                  ):
    # extract 3'UTRs from all the features in the GTF file
    utr = extract_feature(gtf, feature)
    # sort by chr and start for bed merge
    utr = utr.sort_values(by=['seqname', 'start'])
    annotations = ['gene_id', 'score', 'strand', 'transcript_id', 'gene_name']
    ann_cols, ann_idx = prepare_annotations_index(annotations)
    # Create BedTool obj using the GTF dataframe
    utr_bed = pybedtools.BedTool.from_dataframe(utr, columns=ann_cols)
    # find and merge overlapping regions
    utr_reduced = reduce_regions(utr_bed, annotations_index=ann_idx)
    # get the sequences (from the genome file) for each UTR region
    # sequences = get_sequences(genome, utr_reduced)

    utr_reduced = generate_utr_name(utr_reduced)
    utr_reduced = correct_column_names(utr_reduced)

    if store:
        utr_reduced.to_pickle("utr_reference_regions.pkl")

    return utr_reduced


def map_aligned_reads(alignments,
                      utr_regions,
                      protocol,
                      store=False
                      ):
    intervals = generate_utr_intervals(utr_regions)
    sample_names = protocol.sample_names

    if store:
        filename = "UTR_sizes_counts"
        directory = './intermediate'
        helpers.check_if_exists_and_mkdir(directory)

    if isinstance(alignments, (list, tuple)):
        n_cores = -2
        if sample_names is not None:
            samples = []
            for counter in range(len(alignments)):
                samples.append(sample_names[counter])
            read_mappings = joblib.Parallel(n_jobs=n_cores)(
                joblib.delayed(
                    map_reads_to_utr)(alignment, intervals,
                                      read="first", protocol=protocol,
                                      batch=sample
                                      ) for alignment, sample in tqdm(
                    zip(alignments, samples), colour="#ffb700", ascii=" ■"))
        else:
            read_mappings = joblib.Parallel(n_jobs=n_cores)(
                joblib.delayed(
                    map_reads_to_utr)(alignment, intervals,
                                      read="first", protocol=protocol,
                                      batch=None
                                      ) for alignment in tqdm(
                    alignments, colour="#ffb700", ascii=" ■"))

        if store:
            for counter, utr_sizes_counts in enumerate(read_mappings):
                file_path = os.path.join(
                    directory, f'{filename}_{counter + 1}.pkl')
                utr_sizes_counts.to_pickle(file_path)

    else:
        read_mappings = map_reads_to_utr(
            alignments, intervals, read='first', protocol=protocol, batch=sample_names)
        if store:
            read_mappings.to_pickle(os.path.join(directory, f'{filename}.pkl'))
    return read_mappings


def call_peaks(utr_sizes_counts,
               utr_reference_regions,
               protocol,
               annotations=None,
               store=False,
               ):
    logger = module_logger()

    # if more than one counts file is provided, they must be merged before peak calling
    if isinstance(utr_sizes_counts, list) and len(utr_sizes_counts) > 1:
        utr_sizes_counts = [
            helpers.check_if_path_and_open(element, method="pickle_pandas") for element in utr_sizes_counts
        ]

        utr_sizes_counts = merge_samples_utr_sizes(utr_sizes_counts,
                                                   method=protocol.merge_method,
                                                   label_batches=protocol.label_batches,
                                                   )
        if store:
            directory = './intermediate'
            helpers.check_if_exists_and_mkdir(directory)
            utr_sizes_counts.to_pickle(os.path.join(
                directory, f'UTR_sizes_counts_merged_{protocol.merge_method}.pkl'))
    elif isinstance(utr_sizes_counts, list):
        utr_sizes_counts = utr_sizes_counts[0]

    # utr_size_counts can still be a path if merging was not applied
    utr_sizes_counts = helpers.check_if_path_and_open(utr_sizes_counts,
                                                      method='pickle_pandas')

    # if we are to add the reference data to the results later, we must subset the UTRs
    utr_reference_subset = subset_reference_data(utr_sizes_counts, utr_reference_regions)

    del utr_reference_regions  # to save memory

    # get only utr sizes bigger than X bp
    filtered_positions = protocol.filter_utr_sizes(utr_sizes_counts, threshold=1)

    del utr_sizes_counts  # to save memory

    # if we have a table with a specific set of barcodes/IDs, retain only those ones.
    if annotations is not None:
        annotations = helpers.check_if_path_and_open(annotations, method="csv")
        if protocol.label_batches:
            barcodes_column_name = annotations.columns.str.extract(
                r"(.*barcode.*)", flags=re.IGNORECASE).dropna().values[0][0]
            batch_column_name = annotations.columns.str.extract(
                r"(.*batch.*)", flags=re.IGNORECASE).dropna().values[0][0]
            # rename the barcodes with the batch appended
            annotations[barcodes_column_name] = \
                annotations[barcodes_column_name].astype(str) \
                + "-" \
                + annotations[batch_column_name].apply(lambda x: protocol.batch_codes[x]).astype(str)
        """
        if psutil.virtual_memory().percent > 20:  # if there are memory issues
            logger.info(f"Dataset is too large. Splitting in 10 and analysing sequentially")
            num, div = filtered_positions.shape[0], 10
            idx_tenths = np.cumsum([
                num // div + (1 if x < num % div else 0) for x in range(div)
            ])
            idx_mod = np.insert(idx_tenths, 0, 0)

            split_generator = (
                filtered_positions.iloc[idx_mod[n]:idx_mod[n + 1]] for n in range(len(idx_mod) - 1)
            )

            filtered_list = [
                filter_barcodes(pos, annotations=annotations) for pos in split_generator
            ]

            filtered_positions = pd.concat(filtered_list)
        else:
            filtered_positions = filter_barcodes(filtered_positions,
                                                 annotations=annotations)
        """
    # calculate broad peaks based on individual UMI utr sizes
    peaks_count_matrix, *peaks_var_and_obs = generate_peak_counts(
        filtered_positions, annotations=annotations, func=protocol.peak_calling)

    try:
        del filtered_positions  # to save memory
    except NameError:
        pass

    peaks_var = peaks_var_and_obs[0]
    if len(peaks_var_and_obs) == 2:
        peaks_obs = peaks_var_and_obs[1]
    else:
        peaks_obs = None

    # append the reference information to the new UTR peaks
    peaks_var = add_to_peaks_var(peaks_var, utr_reference_subset)

    if store:  # store the final DFs. Useful for debugging or to change annotations.
        logger.info('Storing intermediate results')
        directory = './intermediate'
        helpers.check_if_exists_and_mkdir(directory)
        peaks_count_matrix.to_pickle(os.path.join(
            directory, f'peaks_count_matrix_{protocol.merge_method}.pkl'))
        peaks_var.to_pickle(os.path.join(
            directory, f'peaks_var_{protocol.merge_method}.pkl'))
        utr_reference_subset.to_pickle(os.path.join(
            directory, f'UTR_reference_subset_{protocol.merge_method}.pkl'))
        if annotations is not None:
            peaks_obs.to_pickle(os.path.join(
                directory, f'peaks_obs_{protocol.merge_method}.pkl'))

    # merge results and annotations into
    count_matrix_anndata = generate_count_matrix_anndata(
        peaks_count_matrix, peaks_var, obs=peaks_obs)
    return count_matrix_anndata
