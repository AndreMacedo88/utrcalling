#!/usr/bin/env python

import sys
import argparse
import ast
import os

sys.path.append('../')  # allows utrcalling to be imported from `python utrcalling.py run`
sys.path.append('./')  # allows utrcalling to be imported from this script
from main_module import core, helpers, protocols
from main_module.module_log import module_logger


def main(utr_annotation, alignments, feature='three_prime_utr',
         annotations_file=None, protocol_platform="10x", protocol_version=3,
         store_intermediate=False, log_level='info',
         sample_merge_method=None, call_peaks=True, out='UTR_results.h5ad'):
    """
    Maps reads to UTRs and returns UTR length peak counts per cell/sample from 
    input file/s.

    Args:
        utr_annotation (str): [description]
        alignments (str): [description]
        feature (str, optional): [description]. Defaults to 'three_prime_utr'.
        annotations_file (str, optional): [description]. Defaults to None.
        protocol_version (int, optional): [description]. Defaults to 3.
        store_intermediate (bool, optional): [description]. Defaults to False.
        out (str, optional): [description]. Defaults to 'UTR_results.h5ad'.
    """

    # Set up logging
    logger = module_logger(disable=False, store_log=True, level=log_level)
    for arg, value in locals().items():
        logger.info(f'Argument {arg}: {value}\n')

    # Supply the names of your samples, if applicable:
    # This next if clause is specific to my (Andre) workflow due to path. TODO: generalize it.
    if (not isinstance(alignments, list)) and (os.path.isdir(alignments)):
        alignments = helpers.get_paths_from_folders(alignments, suffix='.bam')
        sample_names = [helpers.get_parent_folders_names_from_file(
            file_path) for file_path in alignments]
    elif protocol_platform in ("smartseq", "smart-seq"):
        sample_names = [file_path.split(".")[0].split("/")[1]
                        for file_path in alignments]
    else:
        sample_names = None

    # Set up the protocol for this specific run
    protocol = protocols.Protocol(protocol_platform, int(protocol_version),
                                  merge_method=sample_merge_method,
                                  call_peaks=call_peaks, sample_names=sample_names)

    # Get UTR reference data from the organisms GTF annotation file
    logger.info('Preparing utr reference regions')
    utr_reference_regions = core.prepare_utr_reference_regions(
        utr_annotation, feature, store=store_intermediate)

    # Map to the reference UTR the reads that properly aligned to genome

    logger.info('Mapping aligned reads to UTR reference')
    utr_sizes_counts = core.map_aligned_reads(
        alignments, utr_reference_regions, protocol, store=store_intermediate)

    # Call peaks from the UTR sizes for each molecule
    logger.info('Calling peaks from mapped data')
    count_matrix_anndata = core.call_peaks(
        utr_sizes_counts, utr_reference_regions, protocol,
        annotations=annotations_file, store=store_intermediate)

    helpers.log_metrics_anndata(count_matrix_anndata)

    try:
        count_matrix_anndata.write(filename=out)
    except TypeError:
        count_matrix_anndata.var["chrom"] = count_matrix_anndata.var["chrom"].astype(str)
        count_matrix_anndata.write(filename=out)

    return


keyword_args = {"-a": "--alignments", "-f": "--feature", "-A": "--annotations_file",
                "-p": "--protocol_platform", "-v": "--protocol_version",
                "-m": "--sample_merge_method", "-C": "--call_peaks",
                "-i": "--store_intermediate", "-l": "--log_level", "-o": "--out"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Runs the entire UTR calling pipeline.
        Exports the UTR sizes count matrix to a h5ad file, which you can name with --out.
        """,
        epilog="""
        Example on how to use in command-line:
        $ run.py mus_musculus.gtf 
            -a bowtie_align.bam bowtie_align_ko.bam -f three_prime_utr -p 10x -v 2 
            -i False -A experiment.csv -o results.h5ad
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        
    parser.add_argument("utr_annotation", type=str,
                        help="GTF file with the genome annotations")

    parser.add_argument("--alignments", "-a", nargs='+', type=str, required=True, help="""
        File, or files with the read alignments. One alignment per sample should be 
        supplied if downstream applications involve comparisons between the samples.
        Alignments must be sorted and have an index file .bai in the same directory
        """)

    parser.add_argument("--feature", "-f", type=str,
                        help="Type of UTR that you want to map. three_prime_utr or \
                            five_prime_utr",
                        default='three_prime_utr')

    parser.add_argument("--annotations_file", "-A", type=str,
                        help="""
                        File with your experiment's information, such as cell barcodes.
                        """,
                        default=None)

    parser.add_argument("--protocol_platform", "-p", type=str,
                        help="Sequencing platform used.",
                        default='10x')

    parser.add_argument("--protocol_version", "-v", type=int,
                        help="Version of the protocol used in library preparation.",
                        default=2)

    parser.add_argument("--sample_merge_method", "-m", type=str,
                        help="""
                        Method used to deal with different UTR coverage in different
                        samples. If `union`, then the tool outputs UTRs that are not 
                        represented by all samples. If `intersection` only the reads that
                        map to UTRs in common between all samples will be used. Default
                        depends on the protocol used.
                        """,
                        default=None)

    parser.add_argument("--store_intermediate", "-i", type=ast.literal_eval,
                        help="""
                        If True, stores intermediate files that can be used to 
                        run quicker pipelines later on. Useful if something wrong 
                        is happening in later stages of the pipeline.
                        """,
                        default=False)

    parser.add_argument("--call_peaks", "-C", type=ast.literal_eval,
                        help="""
                        If True, aggregates individual molecules' UTR sizes into
                        consensus peaks.
                        """,
                        default=True)

    parser.add_argument("--log_level", "-l", type=str,
                        help="Level of logging messages to retrieve during the run.",
                        default="info")

    parser.add_argument("--out", "-o", type=str,
                        help="Path or partial path to a folder to store the result files.",
                        default="UTR_results.h5ad")

    args = parser.parse_args()

    main(args.utr_annotation, args.alignments,
         feature=args.three_prime_utr,
         annotations_file=args.annotations_file,
         protocol_platform=args.protocol_platform,
         protocol_version=args.protocol_version,
         sample_merge_method=args.sample_merge_method,
         store_intermediate=args.store_intermediate,
         call_peaks=args.call_peaks,
         log_level=args.log_level,
         out=args.out
         )
