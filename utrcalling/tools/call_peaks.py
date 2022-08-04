import argparse
import ast

from utrcalling.main_module import core, helpers, protocols
from utrcalling.main_module.module_log import module_logger


def main(utr_annotation, utr_sizes_counts, annotations_file=None, log_level='info',
         store_intermediate=False, protocol_platform="10x", protocol_version=2,
         sample_merge_method=None, call_peaks=True, out="UTR_results.h5ad"):
    """
    Returns UTR length peak counts per cell/sample from UTR length counts and UTR 
    annotation files.

    Args:
        utr_annotation (str): Path to a .bed file with the utr annotations.
        utr_sizes_counts (str): Path to a pickle object with a UtrMappings object.
        single_cell_annotation (str, optional): Path to the experiments' annotation file. 
            Defaults to None.
        store_intermediate (bool, optional): [description]. Defaults to False.
        protocol_platform (str, optional): [description]. Defaults to "10x".
        protocol_version (int, optional): [description]. Defaults to 3.
        out (str, optional): Path to write the results. Defaults to "UTR_results.h5ad".
    """

    # Set up logging
    logger = module_logger(disable=False, store_log=True, level=log_level)
    for arg, value in locals().items():
        logger.info(f'Argument {arg}: {value}\n')

    # set up the protocol for this specific run
    protocol = protocols.protocol(protocol_platform, int(protocol_version),
                                  merge_method=sample_merge_method,
                                  call_peaks=call_peaks)

    # load UTR reference file
    utr_reference_annotations = helpers.load_annotations(utr_annotation)

    # call peaks from the UTR sizes for each molecule
    logger.info('Calling peaks from mapped data')
    utr_sizes_counts = helpers.check_if_dir_and_list_files(utr_sizes_counts,
                                                           suffix='.pkl')
    count_matrix_anndata = core.call_peaks(
        utr_sizes_counts, utr_reference_annotations, protocol,
        annotations=annotations_file, store=store_intermediate)

    helpers.log_metrics_anndata(count_matrix_anndata)

    try:
        count_matrix_anndata.write(filename=out)
    except TypeError:
        count_matrix_anndata.var["chrom"] = count_matrix_anndata.var["chrom"].astype(str)
        count_matrix_anndata.obs["Sample_Name"] = count_matrix_anndata.obs["Sample_Name"].astype(
            str)
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


keyword_args = {"-c": "--utr_sizes_counts", "-A": "--annotations_file",
                "-p": "--protocol_platform", "-v": "--protocol_version",
                "-m": "--sample_merge_method", "-C": "--call_peaks",
                "-i": "--store_intermediate", "-l": "--log_level", "-o": "--out"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Returns UTR length peak counts per cell/sample from UTR length counts and UTR 
        annotation files.
        Exports the UTR sizes count matrix to a h5ad file, which you can name with --out.
        """,
        epilog="""
        Example on how to use in command-line:
        $ call_peaks.py mus_musculus_UTRs.bed -c sample-1.pkl sample-2.pkl 
            -A experiment.csv -p 10x -v 2 -i True -o results.h5ad
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("utr_annotation", type=str,
                        help="BED or pickle file with the UTR annotations")

    parser.add_argument("--utr_sizes_counts", "-c", nargs='+', type=str, required=True,
                        help="""
        File, or files with the utr and read information. One file per sample should be 
        supplied if downstream applications involve comparisons between the samples.
        The files should be pickle files of a UtrMappings object.
        """)

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

    main(args.utr_annotation, args.utr_sizes_counts,
         annotations_file=args.annotations_file,
         protocol_platform=args.protocol_platform,
         protocol_version=args.protocol_version,
         sample_merge_method=args.sample_merge_method,
         store_intermediate=args.store_intermediate,
         call_peaks=args.call_peaks,
         log_level=args.log_level,
         out=args.out
         )
