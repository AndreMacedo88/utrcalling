# import general tools
import argparse
import ast
# import internal tools
from utrcalling.core import core, helpers, protocols
from utrcalling.core.module_log import module_logger


def call_peaks(utr_annotation,
               utr_sizes_counts,
               annotations_file=None,
               batch_merge_method=None,
               protocol_platform="10x",
               protocol_version=2,
               label_batches=None,
               log_level='info',
               out="UTR_results.h5ad",
               raw_sizes=False,
               store_intermediate=False,
               ):
    """
    Returns UTR length peak counts per cell/sample/batch from UTR length counts and UTR 
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
    logger = module_logger(disable=False, store_log=False, level=log_level)
    for arg, value in locals().items():
        logger.info(f'║ Argument {arg}: {value}\n')

    # set up the protocol for this specific run
    protocol = protocols.protocol(
        protocol_platform,
        int(protocol_version),
        merge_method=batch_merge_method,
        label_batches=label_batches,
        call_peaks=not raw_sizes,
        annotations=annotations_file)

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


def main():
    args_all = parser.parse_args()

    args = vars(args_all).copy()
    del args["func"]

    args_all.func(**args)


parser = argparse.ArgumentParser(
    description="""
Returns UTR length peak counts per cell/sample/batch from UTR length counts and UTR 
annotation files.
Exports the UTR sizes count matrix to a h5ad file, which you can name with --out.
""",
    epilog="""
Example on how to use in command-line:
$ call_peaks.py mus_musculus_UTRs.bed -c sample-1.pkl sample-2.pkl 
    -A experiment.csv -p 10x -v 2 -i -o results.h5ad
""",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument("utr_annotation", type=str,
                    help="BED or pickle file with the UTR annotations")

parser.add_argument("--utr_sizes_counts", "-c", nargs='+', type=str, required=True,
                    help="""
    File, or files with the utr and read information. One file per sample/batch should be 
    supplied if downstream applications involve comparisons between the samples/batches.
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

parser.add_argument("--batch_merge_method", "-m", type=str,
                    help="""
                    Method used to deal with different UTR coverage in different
                    batches. If `union`, then the tool outputs UTRs that are not 
                    represented by all batches. If `intersection` only the reads that
                    map to UTRs in common between all batches will be used. Default
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
parser.set_defaults(func=call_peaks)


if __name__ == "__main__":
    main()
