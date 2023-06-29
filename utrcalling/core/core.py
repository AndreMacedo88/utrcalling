"""
# Goal
Mapping RNA-seq reads (single cell or bulk) to UTRs.

# Secondary goal
Call consensus peaks based on the distribution of molecule sizes.

# Summary
This file contains 3 functions for the 3 main steps in this procedure.

# General steps
1. Extract the UTR regions of transcripts (from ENSEMBL)
2. Map the aligned, filtered reads to the UTR regions from step 1.
3. Clean-up the data and call consensus peaks if required.

# More detailed steps
1.1. Resolve overlapping transcripts so that every entry stores unique, non-overlapping,
genomic coordinates

# Observations
- Final object is a count matrix in AnnData format.
- Output should inherit single cell annotations, as well as gene annotation like the
genomic coordinates and optionally UTR sequence information.
"""

import os
import psutil

import numpy as np
import modin.pandas as mpd
import pybedtools
import joblib
from tqdm import tqdm

from utrcalling.core import helpers
from utrcalling.core.module_log import module_logger
from utrcalling.core import utr_reference
from utrcalling.core import utr_mapping
from utrcalling.core import process_mapped_reads
from utrcalling.core import peak_calling


def prepare_utr_reference_regions(gtf,
                                  feature,
                                  store=False
                                  ):
    # open the gtf file
    gtf_df = utr_reference.open_gtf(gtf)
    # extract 3'UTRs from all the features in the GTF file
    utr = utr_reference.extract_feature(gtf_df, feature)
    # sort by chr and start for bed merge
    utr = utr.sort_values(by=['seqname', 'start'])
    annotations = ['gene_id', 'score', 'strand', 'transcript_id', 'gene_name']
    ann_cols, ann_idx = utr_reference.prepare_annotations_index(annotations)
    # Create BedTool obj using the GTF dataframe
    utr_bed = pybedtools.BedTool.from_dataframe(utr, columns=ann_cols)
    # find and merge overlapping regions
    utr_reduced = utr_reference.reduce_regions(utr_bed, annotations_index=ann_idx)
    utr_reduced = utr_reference.turn_utr_bed_to_dataframe(utr_reduced)
    utr_reduced = utr_reference.generate_utr_name(utr_reduced)
    utr_reduced = utr_reference.correct_column_names(utr_reduced)
    utr_reduced.chrom = utr_reduced.chrom.astype(str)

    if store:
        directory = './intermediate'
        helpers.check_if_exists_and_mkdir(directory)
        utr_bed.saveas(os.path.join(directory, "utrs_unmerged.bed"))
        utr_reduced.to_parquet(os.path.join(directory, "utr_reference_regions.parquet"))
    return utr_reduced


def map_aligned_reads(alignments,
                      utr_regions,
                      protocol,
                      store=False
                      ):
    intervals = utr_mapping.generate_utr_intervals(utr_regions)
    sample_names = protocol.sample_names

    if store:
        filename = "UTR_sizes_counts"
        directory = './intermediate'
        helpers.check_if_exists_and_mkdir(directory)

    if isinstance(alignments, (list, tuple)):
        n_cores = -2
        if sample_names is not None:
            samples = []
            for alignment in alignments:
                samples.append(sample_names[alignment])
            read_mappings = joblib.Parallel(n_jobs=n_cores)(
                joblib.delayed(
                    utr_mapping.map_reads_to_utr)(alignment, intervals,
                                                  read="first", protocol=protocol,
                                                  batch=sample
                                                  ) for alignment, sample in tqdm(
                    zip(alignments, samples), colour="#ffb700", ascii=" ■"))
        else:
            read_mappings = joblib.Parallel(n_jobs=n_cores)(
                joblib.delayed(
                    utr_mapping.map_reads_to_utr)(alignment, intervals,
                                                  read="first", protocol=protocol,
                                                  batch=None
                                                  ) for alignment in tqdm(
                    alignments, colour="#ffb700", ascii=" ■"))

        if store:
            for counter, utr_sizes_counts in enumerate(read_mappings):
                file_path = os.path.join(
                    directory, f'{filename}_{counter + 1}.parquet')
                utr_sizes_counts.to_parquet(file_path)

    else:
        read_mappings = utr_mapping.map_reads_to_utr(
            alignments, intervals, read='first', protocol=protocol, batch=sample_names)
        if store:
            read_mappings.to_parquet(os.path.join(directory, f'{filename}.parquet'))

    return read_mappings


def merge_samples_utr_sizes(utr_sizes_counts,
                            protocol,
                            store=False
                            ):
    helpers.init_ray()
    utr_sizes_counts = process_mapped_reads.merge_samples_utr_sizes(
        utr_sizes_counts,
        method=protocol.merge_method,
        label_batches=protocol.label_batches,
    )
    if store:
        directory = './intermediate'
        helpers.check_if_exists_and_mkdir(directory)
        utr_sizes_counts.to_parquet(os.path.join(
            directory, f'UTR_sizes_counts_merged_{protocol.merge_method}.parquet'))
    return utr_sizes_counts


def merge_utrs_per_feature(utr_regions,
                           read_mappings,
                           level="UTR",
                           ):
    if level.lower() == "gene":
        utr_regions, read_mappings = process_mapped_reads.merge_utrs_per_gene(
            utr_regions=utr_regions,
            read_mappings=read_mappings
        )
    return utr_regions, read_mappings


def call_peaks(utr_sizes_counts,
               utr_reference_regions,
               protocol,
               store=False,
               ):
    logger = module_logger()
    helpers.init_ray()

    # if we are to add the reference data to the results later, we must subset the UTRs
    utr_reference_subset = process_mapped_reads.subset_reference_data(
        utr_sizes_counts, utr_reference_regions)

    del utr_reference_regions  # to save memory

    # get only utr sizes bigger than X bp
    filtered_positions = protocol.filter_utr_sizes(utr_sizes_counts, threshold=1)

    del utr_sizes_counts  # to save memory

    if protocol.barcode_whitelist is not None:
        if psutil.virtual_memory().percent > 70:  # if there are memory issues
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
                process_mapped_reads.filter_barcodes(
                    pos, whitelist=protocol.barcode_whitelist
                ) for pos in split_generator
            ]

            filtered_positions = mpd.concat(filtered_list)
        else:
            filtered_positions = process_mapped_reads.filter_barcodes(
                filtered_positions,
                whitelist=protocol.barcode_whitelist
            )

    # calculate broad peaks based on individual UMI utr sizes
    peaks_count_matrix, *peaks_var_and_obs = peak_calling.generate_peak_counts_and_anno(
        filtered_positions, protocol=protocol)

    try:
        del filtered_positions  # to save memory
    except NameError:
        pass

    # retrieve var and obs information
    peaks_var = peaks_var_and_obs[0]
    if len(peaks_var_and_obs) == 2:
        peaks_obs = peaks_var_and_obs[1]
    else:
        peaks_obs = None

    # append the reference information to the new UTR peaks
    peaks_var = peak_calling.add_to_peaks_var(peaks_var, utr_reference_subset)

    # convert to pandas the final objects
    peaks_count_matrix = helpers.check_if_modin_and_convert_to_pandas(peaks_count_matrix)
    peaks_var = helpers.check_if_modin_and_convert_to_pandas(peaks_var)
    peaks_obs = helpers.check_if_modin_and_convert_to_pandas(peaks_obs)

    if store:  # store the final DFs. Useful for debugging or to change annotations.
        logger.info('Storing intermediate results')
        directory = './intermediate'
        helpers.check_if_exists_and_mkdir(directory)
        peaks_count_matrix.to_parquet(os.path.join(
            directory, f'peaks_count_matrix_{protocol.merge_method}.parquet'))
        peaks_var.to_parquet(os.path.join(
            directory, f'peaks_var_{protocol.merge_method}.parquet'))
        utr_reference_subset.to_parquet(os.path.join(
            directory, f'UTR_reference_subset_{protocol.merge_method}.parquet'))
        peaks_obs.to_parquet(os.path.join(
            directory, f'peaks_obs_{protocol.merge_method}.parquet'))

    # merge results and annotations into
    count_matrix_anndata = peak_calling.generate_count_matrix_anndata(
        count_matrix=peaks_count_matrix,
        var=peaks_var,
        obs=peaks_obs
    )
    return count_matrix_anndata
