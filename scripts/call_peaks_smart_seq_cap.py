import sys
import argparse
import ast

sys.path.append('../')
from utrcalling import core, helpers, platforms
from utrcalling.module_log import module_logger


def main(utr_annotation, utr_sizes_counts, sample_annotation=None, log_level='info',
         store_intermediate=False, protocol_platform="smart-seq_cap", protocol_version=3,
         out='UTR_results_cap_phospho.h5ad'):
    """
    Gets UTR sizes peak count per cell/sample from UTR sizes counts and UTR annotation 
    files.

    Args:
        utr_annotation (str): Path to a .bed file with the utr annotations.
        utr_sizes_counts (str): Path to a pickle object with a UtrMappings object.
        sample_annotation (str, optional): Path to the experiments' annotation file. 
            Defaults to None.
        store_intermediate (bool, optional): [description]. Defaults to False.
        protocol_platform (str, optional): [description]. Defaults to "smart-seq_cap".
        protocol_version (int, optional): [description]. Defaults to 3.
        out (str, optional): Path to write the results. Defaults to 
            'UTR_results_cap_phospho.h5ad'.
    """

    # Set up logging
    logger = module_logger(disable=False, store_log=True, level=log_level)
    for arg, value in locals().items():
        logger.info(f'Argument {arg}: {value}\n')

    # set up the protocol for this specific run
    protocol = platforms.Platform(protocol_platform, int(protocol_version),
                                  merge_method='union')

    # load UTR reference file
    utr_reference_annotations = helpers.load_annotations(utr_annotation)

    # call peaks from the UTR sizes for each molecule
    logger.info('Calling peaks from mapped data')
    utr_sizes_counts = helpers.check_if_dir_and_list_files(utr_sizes_counts,
                                                           suffix='.pkl')
    count_matrix_anndata = core.call_peaks(
        utr_sizes_counts, utr_reference_annotations, protocol,
        annotations=sample_annotation, store=store_intermediate, label_samples=False)

    helpers.log_metrics_anndata(count_matrix_anndata)

    count_matrix_anndata.write(filename=out)

    return


keyword_args = {"-c": "--utr_sizes_counts", "-s": "--sample_annotation",
                "-p": "--protocol_platform", "-v": "--protocol_version",
                "-i": "--store_intermediate", "-l": "--log_level", "-o": "--out"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
        Gets UTR sizes peak count per cell/sample from UTR sizes counts and UTR 
        annotation files.
        Exports the UTR sizes count matrix to a h5ad file, which you can name with --out.
        Example on how to use in command-line:
        $ python call_peaks_smart_seq_cap.py mus_musculus.pkl 
            -c CAPSeq_counts_pickled/ -s experiment.csv -p smart-seq_cap -v 3 -i True
            -o results.h5ad
        """)
    parser.add_argument("utr_annotation", type=str,
                        help="BED or pickle file with the UTR annotations")

    parser.add_argument("--utr_sizes_counts", "-c", nargs='+', type=str, required=True, help="""
        File, or files with the utr and read infromation. One file per sample should be 
        supplied if downstream applications involve comparisons between the samples.
        The files should be pickle files of a UtrMappings object.
        """)

    parser.add_argument("--sample_annotation", "-s", type=str,
                        help="File with your experiment's information, \
                            such as cell barcodes.",
                        default=None)

    parser.add_argument("--protocol_platform", "-p", type=str,
                        help="Sequencing platform used.",
                        default='smart-seq_cap')

    parser.add_argument("--protocol_version", "-v", type=int,
                        help="Version of the protocol used in library preparation.",
                        default=3)

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
                        default="UTR_results_cap_phospho.h5ad")

    args = parser.parse_args()

    main(args.utr_annotation, args.utr_sizes_counts,
         sample_annotation=args.sample_annotation,
         protocol_platform=args.protocol_platform,
         protocol_version=args.protocol_version,
         store_intermediate=args.store_intermediate,
         log_level=args.log_level,
         out=args.out
         )
