# import general tools
import argparse
import ast
import os
# import internal tools
from utrcalling.core import core, helpers, protocols
from utrcalling.core.module_log import module_logger


def run(utr_annotation,
        alignments,
        annotations_file=None,
        batch_merge_method=None,
        feature='three_prime_utr',
        protocol_platform="10x",
        protocol_version=3,
        label_batches=None,
        log_level='info',
        out='UTR_results.h5ad',
        raw_sizes=False,
        store_intermediate=False,
        ):
    """
    Maps reads to UTRs and returns UTR length peak counts per cell/sample/batch from 
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
    logger = module_logger(disable=False, store_log=False, level=log_level)
    for arg, value in locals().items():
        logger.info(f'║ Argument {arg}: {value}\n')

    # Supply the names of your samples/batches, if applicable:
    # This next if clause is specific to my (Andre) workflow due to path. TODO: generalize it.
    if (not isinstance(alignments, list)) and (os.path.isdir(alignments)):
        alignments = helpers.get_paths_from_folders(alignments, suffix='.bam')
        sample_names = [helpers.get_parent_folders_names_from_file(
            file_path) for file_path in alignments]
    elif protocol_platform in ("smartseq_cap", "smart-seq_cap"):
        sample_names = [file_path.split("/")[-2] for file_path in alignments]
    else:
        sample_names = None

    # Set up the protocol for this specific run
    protocol = protocols.protocol(
        protocol_platform,
        int(protocol_version),
        merge_method=batch_merge_method,
        call_peaks=not raw_sizes,
        label_batches=label_batches,
        sample_names=sample_names,
        annotations=annotations_file)

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
        try:
            count_matrix_anndata.write(filename=out)
        except TypeError as e:
            logger.error(f"""
Some of your annotations could not be written to a h5ad file.\n
This could be because their type is ambiguous
(characters/integers/etc).\n
To deal with this, either make all annotations columns and re-run the tool,
or run the tool with the option to output the intermediate files and 
build your h5ad object from the counts matrix, obs, and var.
The error message was:\n{e}
            """)

    return


def main():
    args_all = parser.parse_args()

    args = vars(args_all).copy()
    del args["func"]

    args_all.func(**args)


parser = argparse.ArgumentParser(
    description="""
Runs the entire UTR calling pipeline.
Exports the UTR sizes count matrix to a h5ad file, which you can name with --out.
""",
    epilog="""
Example on how to use in command-line:
$ run.py mus_musculus.gtf 
    -a bowtie_align.bam bowtie_align_ko.bam -f three_prime_utr -p 10x -v 2 
    -i -A experiment.csv -o results.h5ad
""",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("utr_annotation", type=str,
                    help="GTF file with the genome annotations")

parser.add_argument("--alignments", "-a", nargs='+', type=str, required=True, help="""
    File, or files with the read alignments. One alignment per sample/batch should be 
    supplied if downstream applications involve comparisons between the samples/batches.
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

parser.add_argument("--batch_merge_method", "-m", type=str,
                    help="""
                    Method used to deal with different UTR coverage in different
                    samples/batches. If `union`, then the tool outputs UTRs that are not 
                    represented by all samples/batches. If `intersection` only the reads that
                    map to UTRs in common between all samples/batches will be used. Default
                    depends on the protocol used.
                    """,
                    default=None)

parser.add_argument("--store_intermediate", "-i", action="store_true",
                    help="""
                    Stores intermediate files that can be used to 
                    run quicker pipelines later on. Useful if something wrong 
                    is happening in later stages of the pipeline.
                    """)

parser.add_argument("--raw_sizes", "-r", action="store_true",
                    help="""
                    Produces a count matrix with the individual UMI UTR sizes, without 
                    aggregating individual molecules' UTR sizes into consensus peaks.
                    """)

parser.add_argument("--label_batches", "-L", type=ast.literal_eval,
                    help="""
                    If True, gets the batches from the annotation file and labels each
                    barcode/sample according to the batch. This prevents aggregation
                    of the results if barcodes are recycled between batches, or when 
                    2 samples have the same ID.
                    """,
                    default=None)

parser.add_argument("--log_level", "-l", type=str,
                    help="Level of logging messages to retrieve during the run.",
                    default="info")

parser.add_argument("--out", "-o", type=str,
                    help="Path or partial path to a folder to store the result files.",
                    default="UTR_results.h5ad")

# This allows for assigning different parsers to different functions:
parser.set_defaults(func=run)

if __name__ == "__main__":
    main()
