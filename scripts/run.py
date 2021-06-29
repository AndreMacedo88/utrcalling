import sys
sys.path.append('../')
import argparse
import ast

from utrcalling import core
from utrcalling.module_log import module_logger


def main(utr_annotation, genome, alignments, feature='three_prime_utr',
         single_cell_annotation=None, protocol_platform="10x", protocol_version=3,
         store_intermediate=False, log_level='info', out='UTR_results.h5ad'):
    """
    Gets UTR sizes peak counts per cell/sample from input file/s.

    Args:
        utr_annotation (str): [description]
        genome (str): [description]
        alignments (str): [description]
        feature (str, optional): [description]. Defaults to 'three_prime_utr'.
        single_cell_annotation (str, optional): [description]. Defaults to None.
        protocol_version (int, optional): [description]. Defaults to 3.
        store_intermediate (bool, optional): [description]. Defaults to False.
        out (str, optional): [description]. Defaults to 'UTR_results.h5ad'.
    """

    # Set up logging
    logger = module_logger(disable=False, store_log=True, level=log_level)

    for arg, value in locals().items():
        logger.info(f'Argument {arg}: {value}\n')

    # get UTR reference data from the organisms GTF annotation file
    logger.info('Preparing utr reference regions')
    utr_reference_regions = core.prepare_utr_reference_regions(utr_annotation, genome,
                                                               feature)
    if store_intermediate:
        utr_reference_regions.to_pickle("utr_reference_regions.pkl")

    # Map to the reference UTR the reads that properly aligned to genome
    logger.info('Mapping aligned reads to UTR reference')
    utr_sizes_counts = core.map_aligned_reads(
        alignments, utr_reference_regions, protocol_platform=protocol_platform,
        protocol_version=int(protocol_version))
    if store_intermediate:
        filename = "UTR_sizes_counts"
        counter = 1
        if isinstance(utr_sizes_counts, list):
            for item in utr_sizes_counts:
                item.export(f'{filename}_{counter}.pkl')
                counter += 1
        else:
            utr_sizes_counts.export(f'{filename}.pkl')

    # call peaks from the UTR sizes for each molecule
    logger.info('Calling peaks from mapped data')
    count_matrix_anndata = core.call_peaks(
        utr_sizes_counts, utr_reference_regions, annotations=single_cell_annotation)

    count_matrix_anndata.write(filename=out)

    return


keyword_args = {"-a": "--alignments", "-f": "--feature", "-s": "--single_cell_annotation",
                "-p": "--protocol_platform", "-v": "--protocol_version",
                "-i": "--store_intermediate", "-l": "--log_level", "-o": "--out"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
        Runs the entire UTR calling pipeline.
        Exports the UTR sizes count matrix to a h5ad file, which you can name with --out.
        Example on how to use in command-line:
        $ python run.py mus_musculus.gtf mus_musculus.fasta 
            -a bowtie_align.bam bowtie_align_ko.bam -f three_prime_utr -p 10x -v 2 
            -i False -s experiment.csv -o results.h5ad
        """)
    parser.add_argument("utr_annotation", type=str,
                        help="GTF file with the genome annotations")

    parser.add_argument("genome", type=str,
                        help="File with the genomic sequences")

    parser.add_argument("--alignments", "-a", nargs='+', type=str, required=True, help="""
        File, or files with the read alignments. One alignment per sample should be 
        supplied if downstream applications involve comparisons between the samples.
        Alignments must be sorted and have an index file .bai in the same directory
        """)

    parser.add_argument("--feature", "-f", type=str,
                        help="Type of UTR that you want to map. three_prime_utr or \
                            five_prime_utr",
                        default='three_prime_utr')

    parser.add_argument("--single_cell_annotation", "-s", type=str,
                        help="File with your experiment's information, \
                            such as cell barcodes.",
                        default=None)

    parser.add_argument("--protocol_platform", "-p", type=str,
                        help="Sequencing platform used",
                        default='10x')

    parser.add_argument("--protocol_version", "-v", type=int,
                        help="Version of the protocol used in library preparation.",
                        default=2)

    parser.add_argument("--store_intermediate", "-i", type=ast.literal_eval,
                        help="If True, stores intermediate files that can be used to \
                            run quicker pipelines later on. Useful if something wrong \
                            is happening in later stages of the pipeline",
                        default=False)

    parser.add_argument("--log_level", "-l", type=str,
                        help="Level of logging messages to retrieve during the run.",
                        default="info")

    parser.add_argument("--out", "-o", type=str,
                        help="Path or partial path to a folder to store the result files.",
                        default="UTR_results.h5ad")

    args = parser.parse_args()

    main(args.utr_annotation, args.genome, args.alignments, feature=args.three_prime_utr,
         single_cell_annotation=args.single_cell_annotation,
         protocol_version=args.protocol_version,
         out=args.out
         )
