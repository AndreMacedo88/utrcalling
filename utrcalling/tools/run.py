# import general tools
import argparse
import ast
import os


def _write_anndata(adata, filename, logger):
    """Write an AnnData object to h5ad, handling dtype issues."""
    try:
        adata.write(filename=filename)
    except TypeError:
        adata.var["chrom"] = adata.var["chrom"].astype(str)
        adata.obs["Sample_Name"] = adata.obs["Sample_Name"].astype(str)
        try:
            adata.write(filename=filename)
        except TypeError as e:
            logger.error(
                f"""
Some of your annotations could not be written to a h5ad file.\n
This could be because their type is ambiguous
(characters/integers/etc).\n
To deal with this, either make all annotations columns and re-run the tool,
or run the tool with the option to output the intermediate files and
build your h5ad object from the count matrix, obs, and var.
The error message was:\n{e}
            """
            )


def run(
    utr_annotation,
    alignments,
    annotations_file=None,
    batch_merge_method=None,
    feature="three_prime_utr",
    protocol_platform="10x",
    protocol_version=3,
    label_batches=None,
    summarization_level="UTR",
    raw_sizes=False,
    kde_bandwidth=12,
    store_intermediate=False,
    log_level="info",
    out="UTR_counts.h5ad",
    store_log=True,
    no_log_file=False,
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

    from utrcalling.core import core, helpers, protocol
    from utrcalling.core.protocols import peak_calling as pc
    from utrcalling.core.module_log import module_logger

    # Set up logging
    store_log = store_log and not no_log_file
    log_dir = os.path.dirname(os.path.abspath(out)) if store_log else None
    logger = module_logger(
        disable=False, store_log=store_log, log_dir=log_dir, level=log_level
    )
    for arg, value in locals().items():
        logger.info(f"║ Argument {arg}: {value}")

    # If a directory is provided instead of individual BAM files, discover them
    if (not isinstance(alignments, list)) and (os.path.isdir(alignments)):
        alignments = helpers.get_paths_from_folders(alignments, suffix=".bam")

    # Set up the protocol for this specific run
    protocol_ = protocol.create(
        protocol_platform,
        int(protocol_version),
        merge_method=batch_merge_method,
        call_peaks=not raw_sizes,
        label_batches=label_batches,
        alignments=alignments,
        annotations=annotations_file,
    )

    # Get UTR reference data from the organisms GTF annotation file
    logger.info("Preparing utr reference regions")
    utr_reference_regions = core.prepare_utr_reference_regions(
        utr_annotation, feature, store=store_intermediate
    )

    # Map to the reference UTR the reads that properly aligned to genome
    logger.info("Mapping aligned reads to UTR reference")
    utr_sizes_counts = core.map_aligned_reads(
        alignments=alignments,
        utr_regions=utr_reference_regions,
        protocol=protocol_,
        store=store_intermediate,
    )

    if isinstance(utr_sizes_counts, list) and len(utr_sizes_counts) > 1:
        utr_sizes_counts = core.merge_samples_utr_sizes(
            utr_sizes_counts=utr_sizes_counts,
            protocol=protocol_,
            store=store_intermediate,
        )

    elif isinstance(utr_sizes_counts, list):
        utr_sizes_counts = utr_sizes_counts[0]

    # Collapse UTR size information to gene-level
    if summarization_level.lower() != "utr":
        logger.info(f"Collapsing UTR size information to {summarization_level}-level")
        utr_reference_regions, utr_sizes_counts = core.merge_utrs_per_feature(
            utr_regions=utr_reference_regions,
            read_mappings=utr_sizes_counts,
            level=summarization_level,
        )

    # Filter mapped reads before peak calling
    logger.info("Filtering mapped reads")
    filtered_reads, utr_reference_subset = core.filter_mapped_reads(
        utr_sizes_counts=utr_sizes_counts,
        utr_reference_regions=utr_reference_regions,
        protocol=protocol_,
    )

    if raw_sizes:
        # Pass 1: raw sizes (no transform — use sizes directly)
        logger.info("Building raw sizes count matrix")
        raw_anndata = core.build_count_matrix(
            filtered_reads=filtered_reads,
            utr_reference_subset=utr_reference_subset,
            protocol=protocol_,
            store=store_intermediate,
        )
        helpers.log_metrics_anndata(raw_anndata)
        _write_anndata(raw_anndata, out, logger)
        del raw_anndata  # free memory before KDE pass

        # Pass 2: KDE peaks (transform sizes via KDE)
        logger.info("Building KDE peaks count matrix")
        out_stem, _ = os.path.splitext(out)
        kde_out = f"{out_stem}_kde.h5ad"
        kde_anndata = core.build_count_matrix(
            filtered_reads=filtered_reads,
            utr_reference_subset=utr_reference_subset,
            protocol=protocol_,
            size_transform=pc.get_caller("kde", bandwidth=kde_bandwidth),
        )
        helpers.log_metrics_anndata(kde_anndata)
        _write_anndata(kde_anndata, kde_out, logger)
        logger.info(f"Wrote KDE peaks to {kde_out}")
    else:
        # Standard: call peaks using protocol's configured KDE
        logger.info("Calling peaks from filtered data")
        count_matrix_anndata = core.call_peaks(
            filtered_reads=filtered_reads,
            utr_reference_subset=utr_reference_subset,
            protocol=protocol_,
            store=store_intermediate,
        )
        helpers.log_metrics_anndata(count_matrix_anndata)
        _write_anndata(count_matrix_anndata, out, logger)

    return


def create_parser(subparsers):

    help = "Runs the entire UTR calling pipeline."

    description = f"{help} " "Exports the UTR sizes count matrix to a h5ad file."

    parser = subparsers.add_parser(
        name="run",
        help=help,
        description=f"DESCRIPTION: {description}",
        epilog="""
    Example on how to use in command-line:
    $ utrcalling run mus_musculus.gtf
    -a star_align.bam star_align_ko.bam -f three_prime_utr -p 10x -v 2
    -i -A experiment.csv -o results.h5ad
    """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "utr_annotation", type=str, help="GTF file with the genome annotations"
    )

    parser.add_argument(
        "--alignments",
        "-a",
        nargs="+",
        type=str,
        required=True,
        help="""
        File, or files with the read alignments. One alignment per sample/batch should be
        supplied if downstream applications involve comparisons between the samples/batches.
        Alignments must be sorted and have an index file .bai in the same directory
        """,
    )

    parser.add_argument(
        "--feature",
        "-f",
        type=str,
        help="Type of UTR that you want to map. three_prime_utr or \
                            five_prime_utr",
        default="three_prime_utr",
    )

    parser.add_argument(
        "--annotations_file",
        "-A",
        type=str,
        help="""
                        File with your experiment's information, such as cell barcodes,
                        sample IDs, and batch information.
                        """,
        default=None,
        required=True,
    )

    parser.add_argument(
        "--protocol_platform",
        "-p",
        type=str,
        help="Sequencing platform used.",
        default="10x",
    )

    parser.add_argument(
        "--protocol_version",
        "-v",
        type=int,
        help="Version of the protocol used in library preparation.",
        default=2,
    )

    parser.add_argument(
        "--batch_merge_method",
        "-m",
        type=str,
        help="""
                        Method used to deal with different UTR coverage in different
                        samples/batches. If 'union', then the tool outputs UTRs that are not
                        represented by all samples/batches. If 'intersection' only the reads that
                        map to UTRs in common between all samples/batches will be used. Default
                        depends on the protocol used.
                        """,
        default=None,
    )

    parser.add_argument(
        "--summarization_level",
        "-s",
        type=str,
        help="""
                        Perform UTR size calling at the level of (merged) 'UTRs',
                        'transcript', or 'gene' level.
                        """,
        default="UTR",
    )

    parser.add_argument(
        "--store_intermediate",
        "-i",
        action="store_true",
        help="""
                        Stores intermediate files that can be used to
                        run quicker pipelines later on. Useful if something wrong
                        is happening in later stages of the pipeline.
                        """,
    )

    parser.add_argument(
        "--raw_sizes",
        "-r",
        action="store_true",
        help="""
                        Produces a count matrix with the individual UMI UTR sizes, without
                        aggregating individual molecules' UTR sizes into consensus peaks.
                        Also produces a second file with KDE-estimated peaks (suffix _kde).
                        """,
    )

    parser.add_argument(
        "--kde_bandwidth",
        type=int,
        default=12,
        help="Bandwidth for the Gaussian KDE when --raw_sizes produces the KDE peaks output.",
    )

    parser.add_argument(
        "--label_batches",
        "-L",
        type=ast.literal_eval,
        help="""
                        If True, gets the batches from the annotation file and labels each
                        barcode/sample according to the batch. This prevents aggregation
                        of the results if barcodes are recycled between batches, or when
                        2 samples have the same ID.
                        """,
        default=None,
    )

    parser.add_argument(
        "--log_level",
        "-l",
        type=str,
        help="Level of logging messages to retrieve during the run.",
        default="info",
    )

    parser.add_argument(
        "--no_log_file",
        action="store_true",
        help="Disable writing a log file alongside the output.",
    )

    parser.add_argument(
        "--out",
        "-o",
        type=str,
        help="Path or partial path to a folder to store the result files.",
        default="UTR_results.h5ad",
    )

    parser.set_defaults(func=run)
