import sys
sys.path.append('../')
import argparse
import pickle

import pybedtools

from utrcalling import core
from utrcalling.module_log import module_logger


def main(utr_annotation, utr_sizes_counts, single_cell_annotation=None, log_level='info',
         out="UTR_results.h5ad"):
    """
    Gets UTR sizes peak count per cell/sample from UTR sizes counts and UTR annotation 
    files.

    Args:
        utr_annotation (str): Path to a .bed file with the utr annotations.
        utr_sizes_counts (str): Path to a pickle object with a UtrMappings object.
        single_cell_annotation (str, optional): Path to the experiments' annotation file. 
            Defaults to None.
        out (str, optional): Path to write the results. Defaults to "UTR_results.h5ad".
    """

    # Set up logging
    logger = module_logger(disable=False, store_log=True, level=log_level)

    for arg, value in locals().items():
        logger.info(f'Argument {arg}: {value}\n')

    if isinstance(utr_sizes_counts, list):
        utr_sizes_counts_open = []
        for file_ in utr_sizes_counts:
            with open(file_, "rb") as input_file:
                utr_sizes_counts_open.append(pickle.load(input_file))
    else:
        with open(utr_sizes_counts, "rb") as input_file:
            utr_sizes_counts_open = pickle.load(input_file)

    utr_reference_annotations = pybedtools.BedTool(utr_annotation)

    # call peaks from the UTR sizes for each molecule
    logger.info('Calling peaks from mapped data')
    count_matrix_anndata = core.call_peaks(
        utr_sizes_counts_open, utr_reference_annotations, annotations=single_cell_annotation)

    count_matrix_anndata.write(filename=out)

    return


keyword_args = {"-c": "--utr_sizes_counts", "-s": "--single_cell_annotation",
                "-l": "--log_level", "-o": "--out"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
        Gets UTR sizes peak count per cell/sample from UTR sizes counts and UTR 
        annotation files.
        Exports the UTR sizes count matrix to a h5ad file, which you can name with --out.
        Example on how to use in command-line:
        $ python call_peaks.py mus_musculus_UTRs.bed -c sample-1.pkl sample-2.pkl 
            -s experiment.csv -o results.h5ad
        """)
    parser.add_argument("utr_annotation", type=str,
                        help="GTF file with the genome annotations")

    parser.add_argument("--utr_sizes_counts", "-c", nargs='+', type=str, required=True, help="""
        File, or files with the utr and read infromation. One file per sample should be 
        supplied if downstream applications involve comparisons between the samples.
        The files should be pickle files of a UtrMappings object.
        """)

    parser.add_argument("--single_cell_annotation", "-s", type=str,
                        help="File with your experiment's information, such as cell barcodes.",
                        default=None)

    parser.add_argument("--log_level", "-l", type=str,
                        help="Level of logging messages to retrieve during the run.",
                        default="info")

    parser.add_argument("--out", "-o", type=str,
                        help="Path or partial path to a folder to store the result files.",
                        default="UTR_results.h5ad")

    args = parser.parse_args()

    main(args.utr_annotation, args.genome, args.alignments, feature=args.three_prime_utr,
         single_cell_annotation=args.single_cell_annotation,
         protocol_version=args.protocol_version, out=args.out
         )
