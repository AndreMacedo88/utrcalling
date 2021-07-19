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
genomic coordinates and optionally UTR sequence information.
"""

import os

import pandas as pd
import pybedtools

from utrcalling import helpers
from .utr_reference import (
    generate_utr_name, extract_feature, prepare_annotations_index, reduce_regions,
    generate_utr_name, correct_column_names
)
from .utr_mapping import generate_utr_intervals, map_reads_to_utr, module_logger
from .peak_calling import (
    subset_reference_data, merge_samples_utr_sizes, filter_barcodes,
    generate_peak_counts, add_to_peaks_var, generate_count_matrix_anndata
)


def prepare_utr_reference_regions(gtf, feature, genome='', store=False):
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


def map_aligned_reads(alignments, utr_regions, protocol, store=False, samples=None):

    intervals = generate_utr_intervals(utr_regions)

    counter = 1
    if store:
        filename = "UTR_sizes_counts"
        directory = './intermediate'
        helpers.check_if_exists_and_mkdir(directory)
    if isinstance(alignments, (list, tuple)):
        read_mappings = []
        for alignment in alignments:
            if samples is not None:
                sample = samples[counter - 1]
            else:
                sample = samples
            utr_sizes_counts = map_reads_to_utr(
                alignment, intervals, read='first', protocol=protocol, sample=sample)
            if store:
                file_path = os.path.join(
                    directory, f'{filename}_{counter}.pkl')
                utr_sizes_counts.export(file_path)
                read_mappings.append(file_path)
            else:
                read_mappings.append(utr_sizes_counts)
            counter += 1
    else:
        read_mappings = map_reads_to_utr(
            alignments, intervals, read='first', protocol=protocol, sample=samples)
        if store:
            read_mappings.export(os.path.join(directory, f'{filename}.pkl'))
    return read_mappings


def call_peaks(utr_sizes_counts, utr_reference_regions, protocol,
               annotations=None, label_samples=True, store=False, exclude_from_merge=[]):
    logger = module_logger()

    # if more than one counts file is provided, they must be merged before peak calling
    if isinstance(utr_sizes_counts, list) and len(utr_sizes_counts) > 1:
        utr_sizes_counts = merge_samples_utr_sizes(utr_sizes_counts,
                                                   method=protocol.merge_method,
                                                   label_samples=label_samples,
                                                   exclude=exclude_from_merge)
        if store:
            directory = './intermediate'
            helpers.check_if_exists_and_mkdir(directory)
            utr_sizes_counts.export(os.path.join(
                directory, f'UTR_sizes_counts_merged_{protocol.merge_method}.pkl'))
    elif isinstance(utr_sizes_counts, list):
        utr_sizes_counts = utr_sizes_counts[0]

    # utr_size_counts can still be a path if merging was not applied
    utr_sizes_counts = helpers.check_if_path_and_open(utr_sizes_counts, method='pickle')

    # if we are to add the reference data to the results later, we must subset the UTRs
    utr_reference_subset = subset_reference_data(
        utr_sizes_counts, utr_reference_regions)

    del utr_reference_regions  # to save memory

    # get only utr sizes bigger than X bp
    filtered_positions = protocol.filter_utr_sizes(utr_sizes_counts, amount=1)

    del utr_sizes_counts  # to save memory

    # if we have a table with a specific set of barcodes/sampleIDs, retain only those
    if annotations is not None:
        annotations = pd.read_csv(annotations)
        filtered_positions = filter_barcodes(filtered_positions, annotations)

    # calculate broad peaks based on individual UMI utr sizes
    peaks_count_matrix, *peaks_var_and_obs = generate_peak_counts(
        filtered_positions, annotations=annotations, func=protocol.peak_calling)

    del filtered_positions  # to save memory

    peaks_var = peaks_var_and_obs[0]
    if len(peaks_var_and_obs) == 2:
        peaks_obs = peaks_var_and_obs[1]
    else:
        peaks_obs = None
    
    if store:  # store some data. Useful for debugging or to change annotations.
        directory = './intermediate'
        peaks_count_matrix.to_pickle(os.path.join(
                directory, f'peaks_count_matrix_{protocol.merge_method}.pkl'))
        peaks_obs.to_pickle(os.path.join(
                directory, f'peaks_obs_{protocol.merge_method}.pkl'))
        peaks_var.to_pickle(os.path.join(
                directory, f'peaks_var_{protocol.merge_method}.pkl'))
        utr_reference_subset.to_pickle(os.path.join(
                directory, f'UTR_reference_subset_{protocol.merge_method}.pkl'))

    # append the reference information to the new UTR peaks
    peaks_var = add_to_peaks_var(peaks_var, utr_reference_subset)

    # merge results and annotations into
    count_matrix_anndata = generate_count_matrix_anndata(
        peaks_count_matrix, peaks_var, obs=peaks_obs)
    return count_matrix_anndata
