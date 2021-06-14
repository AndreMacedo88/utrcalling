"""
Goal: Calling single cell 3'UTR peaks
Steps:
1- Extract the 3'UTR regions of transcripts (from ENSEMBL)
1.2 - Resolve overlapping transcripts so that every entry stores unique, non-overlapping, 
genomic coordinates
2-

Obs:
- Final object should be a gene ordered list in a format like GRanges object- use pyranges 
package. Should be able to reconstruct count matrix, or instead just offer count matrix 
straight away.
- Output should inherit single cell annotations, as well as gene annotation like the 
genomic coordinates (GRanges obj) and 3'UTR sequence information.
"""

import pandas as pd
import pybedtools

from .module_log import module_logger
from .utr_reference import (
    extract_feature, prepare_annotations_index, reduce_regions, get_sequences
    )
from .utr_mapping import generate_utr_intervals, map_reads_to_utr
from .peak_calling import (
    subset_data, merge_position_counts, filter_positions, filter_barcodes, 
    generate_peak_counts, generate_count_matrix_anndata
)

# Set up logging
logger = module_logger()

def standard_run(utr_annotation, genome, feature, alignments, single_cell_annotation=None, 
protocol_version=3):
    utr_reference_regions = prepare_utr_reference_regions(utr_annotation, genome, feature)
    utr_sizes_counts = map_aligned_reads(
        alignments, utr_reference_regions,protocol_version=protocol_version)
    count_matrix_anndata = call_peaks(
        utr_sizes_counts, utr_reference_regions, annotations=single_cell_annotation)
    return count_matrix_anndata

def prepare_utr_reference_regions(gtf, genome, feature):
    # extract 3'UTRs from all the features in the GTF file
    utr = extract_feature(gtf, feature)
    # sort by chr and start for bed merge
    utr = utr.sort_values(by=['seqname', 'start'])
    annotations = ['gene_id', 'score', 'strand', 'transcript_id']
    ann_cols, ann_idx = prepare_annotations_index(annotations)
    # Create BedTool obj using the GTF dataframe
    utr_bed = pybedtools.BedTool.from_dataframe(utr, columns=ann_cols)
    # find and merge overlapping regions
    utr_reduced = reduce_regions(utr_bed, annotations_index=ann_idx)
    # get the sequences (from the genome file) for each UTR region
    sequences = get_sequences(genome, utr_reduced)
    return utr_reduced

def map_aligned_reads(alignments, utr_regions, protocol_version=3):
    intervals = generate_utr_intervals(utr_regions)

    if isinstance(alignments, (list, tuple)):
        read_mappings = [
            map_reads_to_utr(
                alignment, intervals, read='first', version=protocol_version) \
                    for alignment in alignments
            ]
    else:
        read_mappings = map_reads_to_utr(
            alignments, intervals, read='first', version=protocol_version)
    
    return read_mappings

def call_peaks(utr_sizes_counts, utr_reference_regions, annotations=None):
    if isinstance(utr_sizes_counts, list):
        utr_sizes_counts = merge_position_counts(utr_sizes_counts)
        
    if annotations is not None:
        annotations = pd.read_csv(annotations)
    utr_reference_subset = subset_data(utr_sizes_counts, utr_reference_regions)
    filtered_positions = filter_positions(utr_sizes_counts)
    if annotations is not None:
        filtered_positions = filter_barcodes(filtered_positions, annotations)
    
    peaks_count_matrix, peaks_obs, peaks_var = generate_peak_counts(
        filtered_positions, annotations=annotations)
    count_matrix_anndata = generate_count_matrix_anndata(
        peaks_count_matrix, peaks_obs, peaks_var)
    return count_matrix_anndata


if __name__ == "__main__":
    # standard_run("/mnt/o_drive/andre/projects/3utr/youngNSC/Mus_musculus.GRCm39.104.gtf.gz", protocol_version=2)
    pass