"""
Goal: Calling single cell 3'UTR peaks
Steps:
1- Extract the 3'UTR regions of transcripts (from ENSEMBL)
1.2 - Resolve overlapping transcripts so that every entry stores unique, non-overlapping, 
genomic coordinates
2-

Observations:
- Final object should be a count matrix in AnnData format.
- Output should inherit single cell annotations, as well as gene annotation like the 
genomic coordinates (GRanges obj) and 3'UTR sequence information.
"""

import os
import pickle

import pandas as pd
import pybedtools

from utrcalling import helpers
from .platforms import Platform
from .utr_reference import (
    generate_utr_name, extract_feature, prepare_annotations_index, reduce_regions,
    generate_utr_name, correct_column_names
)
from .utr_mapping import generate_utr_intervals, map_reads_to_utr, module_logger
from .peak_calling import (
    subset_reference_data, merge_samples_utr_sizes, filter_utr_sizes, filter_barcodes,
    generate_peak_counts, add_to_peaks_var, generate_count_matrix_anndata
)


def prepare_utr_reference_regions(gtf, feature, genome=''):
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
    return utr_reduced


def map_aligned_reads(alignments, utr_regions, protocol_platform="10x",
                      protocol_version=3, store=False):
    intervals = generate_utr_intervals(utr_regions)

    protocol = Platform(protocol_platform, protocol_version)

    if store:
        filename = "UTR_sizes_counts"
        counter = 1
        directory = './intermediate'
        helpers.check_exists_and_store(directory)
    if isinstance(alignments, (list, tuple)):
        read_mappings = []
        for alignment in alignments:
            utr_sizes_counts = map_reads_to_utr(
                alignment, intervals, read='first', platform=protocol)
            if store:
                file_path = os.path.join(
                    directory, f'{filename}_{counter}.pkl')
                utr_sizes_counts.export(file_path)
                counter += 1
                read_mappings.append(file_path)
            else:
                read_mappings.append(utr_sizes_counts)
    else:
        read_mappings = map_reads_to_utr(
            alignments, intervals, read='first', platform=protocol)
        if store:
            read_mappings.export(os.path.join(directory, f'{filename}.pkl'))
    return read_mappings


def call_peaks(utr_sizes_counts, utr_reference_regions, annotations=None, store=False):
    logger = module_logger()

    if isinstance(utr_sizes_counts, list) and len(utr_sizes_counts) > 1:
        utr_sizes_counts = merge_samples_utr_sizes(utr_sizes_counts)
        if store:
            directory = './intermediate'
            helpers.check_exists_and_store(directory)
            utr_sizes_counts.export(os.path.join(
                directory, 'UTR_sizes_counts_merged.pkl'))
    elif isinstance(utr_sizes_counts, list):
        utr_sizes_counts = utr_sizes_counts[0]
    if isinstance(utr_sizes_counts, str):
        with open(utr_sizes_counts, "rb") as input_file:
            utr_sizes_counts = pickle.load(input_file)

    utr_reference_subset = subset_reference_data(
        utr_sizes_counts, utr_reference_regions)

    del utr_reference_regions  # to save memory

    filtered_positions = filter_utr_sizes(utr_sizes_counts)

    del utr_sizes_counts  # to save memory

    if annotations is not None:
        annotations = pd.read_csv(annotations)
        filtered_positions = filter_barcodes(filtered_positions, annotations)

    peaks_count_matrix, peaks_var, peaks_obs = generate_peak_counts(
        filtered_positions, annotations=annotations)

    del filtered_positions  # to save memory

    logger.debug(f'len before adding: {len(peaks_var)}')
    logger.debug(f'Is unique before adding? {peaks_var.index.is_unique}')

    # append some reference information to the new UTR peaks
    peaks_var = add_to_peaks_var(peaks_var, utr_reference_subset)

    logger.debug(f'len after adding: {len(peaks_var)}')
    logger.debug(f'Is unique before adding? {peaks_var.index.is_unique}')
    logger.debug(
        f'Not unique: {peaks_var[peaks_var.index.duplicated(keep=False)].index.tolist()}')

    logger.debug(f'Count matrix shape: {peaks_count_matrix.shape}')

    count_matrix_anndata = generate_count_matrix_anndata(
        peaks_count_matrix, peaks_var, obs=peaks_obs)
    return count_matrix_anndata
