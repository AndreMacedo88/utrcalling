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

import pandas as pd
import pybedtools

from .platforms import Platform
from .utr_reference import (
    generate_utr_name, extract_feature, prepare_annotations_index, reduce_regions,
    get_sequences, generate_utr_name, correct_column_names
)
from .utr_mapping import generate_utr_intervals, map_reads_to_utr
from .peak_calling import (
    subset_reference_data, merge_samples_utr_sizes, filter_utr_sizes, filter_barcodes,
    generate_peak_counts, add_to_peaks_var, generate_count_matrix_anndata
)


def prepare_utr_reference_regions(gtf, genome, feature):
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
                      protocol_version=3):
    intervals = generate_utr_intervals(utr_regions)

    protocol = Platform(protocol_platform, protocol_version)
    if isinstance(alignments, (list, tuple)):
        read_mappings = [
            map_reads_to_utr(
                alignment, intervals, read='first', platform=protocol)
            for alignment in alignments
        ]
    else:
        read_mappings = map_reads_to_utr(
            alignments, intervals, read='first', platform=protocol)
    return read_mappings


def call_peaks(utr_sizes_counts, utr_reference_regions, annotations=None):
    if isinstance(utr_sizes_counts, list):
        utr_sizes_counts = merge_samples_utr_sizes(utr_sizes_counts)

    utr_reference_subset = subset_reference_data(
        utr_sizes_counts, utr_reference_regions)

    del utr_reference_regions  # to save memory

    filtered_positions = filter_utr_sizes(utr_sizes_counts)

    del utr_sizes_counts  # to save memory

    if annotations is not None:
        annotations = pd.read_csv(annotations)
        filtered_positions = filter_barcodes(filtered_positions, annotations)
    peaks_count_matrix, peaks_obs, peaks_var = generate_peak_counts(
        filtered_positions, annotations=annotations)

    del filtered_positions  # to save memory

    peaks_var = add_to_peaks_var(peaks_var, utr_reference_subset)

    count_matrix_anndata = generate_count_matrix_anndata(
        peaks_count_matrix, peaks_obs, peaks_var)
    return count_matrix_anndata
