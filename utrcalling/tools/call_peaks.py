# import general tools
from __future__ import annotations

import argparse
import ast
import os


def call_peaks(
    utr_annotation: str,
    utr_sizes_counts: list[str],
    annotations_file: str | None = None,
    batch_merge_method: str | None = None,
    protocol_platform: str = "10x",
    protocol_version: int = 3,
    label_batches: bool | str | None = None,
    summarization_level: str = "UTR",
    raw_sizes: bool = False,
    kde_bandwidth: int = 12,
    store_intermediate: bool = False,
    log_level: str = "info",
    out: str = "UTR_results.h5ad",
    store_log: bool = True,
    no_log_file: bool = False,
    cores: int | None = None,
) -> None:
    """
    Returns UTR length peak counts per cell/sample/batch from UTR length counts and UTR
    annotation files.

    Args:
        utr_annotation (str): Path to the UTR reference (.parquet, or .pkl).
        utr_sizes_counts (str): Path to a parquet object with a UtrMappings object.
        single_cell_annotation (str, optional): Path to the experiments' annotation file.
            Defaults to None.
        store_intermediate (bool, optional): [description]. Defaults to False.
        protocol_platform (str, optional): [description]. Defaults to "10x".
        protocol_version (int, optional): [description]. Defaults to 3.
        out (str, optional): Path to write the results. Defaults to "UTR_results.h5ad".
    """

    from utrcalling.core import core, helpers, protocol, storage
    from utrcalling.core.module_log import module_logger
    from utrcalling.core.protocols import peak_calling as pc

    # Set up logging
    store_log = store_log and not no_log_file
    log_dir = os.path.dirname(os.path.abspath(out)) if store_log else None
    logger = module_logger(disable=False, store_log=store_log, log_dir=log_dir, level=log_level)
    for arg, value in locals().items():
        logger.info(f"║ Argument {arg}: {value}")

    # set up the protocol for this specific run
    protocol_ = protocol.create(
        protocol_platform,
        int(protocol_version),
        merge_method=batch_merge_method,
        label_batches=label_batches,
        call_peaks=not raw_sizes,
        annotations=annotations_file,
        kde_bandwidth=kde_bandwidth,
    )

    # load UTR reference file
    utr_reference_regions = helpers.load_annotations(utr_annotation)

    # if more than one counts file is provided, they must be merged before peak calling
    if len(utr_sizes_counts) > 1:
        utr_sizes_counts = [
            helpers.check_if_path_and_open(element, method="parquet")
            for element in utr_sizes_counts
        ]

        utr_sizes_counts = core.merge_samples_utr_sizes(
            utr_sizes_counts=utr_sizes_counts,
            protocol=protocol_,
            store=store_intermediate,
        )

    else:
        utr_sizes_counts = helpers.check_if_path_and_open(utr_sizes_counts[0], method="parquet")

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
            n_jobs=cores,
        )
        helpers.log_metrics_anndata(raw_anndata)
        storage.write_anndata(raw_anndata, out)
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
            n_jobs=cores,
        )
        helpers.log_metrics_anndata(kde_anndata)
        storage.write_anndata(kde_anndata, kde_out)
        logger.info(f"Wrote KDE peaks to {kde_out}")
    else:
        # Standard: call peaks using protocol's configured KDE
        logger.info("Calling peaks from filtered data")
        count_matrix_anndata = core.call_peaks(
            filtered_reads=filtered_reads,
            utr_reference_subset=utr_reference_subset,
            protocol=protocol_,
            store=store_intermediate,
            n_jobs=cores,
        )
        helpers.log_metrics_anndata(count_matrix_anndata)
        storage.write_anndata(count_matrix_anndata, out)

    return


def create_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:

    help = "Calls consensus peaks based on raw UTR size counts."

    description = (
        f"{help} "
        "Returns UTR size consensus peak counts per cell/sample/batch "
        "from UTR size counts and UTR annotation files. "
        "Exports the UTR sizes count matrix to a h5ad file. "
        "This is the third part of the UTR calling pipeline."
    )

    parser = subparsers.add_parser(
        name="call-peaks",
        help=help,
        description=f"DESCRIPTION: {description}",
        epilog="""
    Example on how to use in command-line:
    $ utrcalling call-peaks mus_musculus_UTRs.bed -c sample-1.parquet sample-2.parquet
    -A experiment.csv -p 10x -v 2 -i -o results.h5ad
    """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "utr_annotation",
        type=str,
        help=(
            "UTR reference: the utr_reference_regions.parquet that run -i or "
            "prepare-reference-and-map-reads write (or a .pkl from older versions)."
        ),
    )

    parser.add_argument(
        "--utr_sizes_counts",
        "-c",
        nargs="+",
        type=str,
        required=True,
        help="""
        File, or files with the utr and read information. One file per sample/batch should be
        supplied if downstream applications involve comparisons between the samples/batches.
        The files should be parquet files of a UtrMappings object.
        """,
    )

    parser.add_argument(
        "--annotations_file",
        "-A",
        type=str,
        help="""
                        File with your experiment's information, such as cell barcodes.
                        """,
        default=None,
    )

    parser.add_argument(
        "--protocol_platform",
        "-p",
        type=str,
        help=(
            "Sequencing platform used. Options: 10x, smartseq, "
            "smart-seq_cap, smart-seq_cap-readname, 5prime10x, "
            "standard_10x (Cell Ranger BAM), standard_smartseq, "
            "truseq (stranded bulk RNA-seq, e.g. TruSeq Stranded mRNA), "
            "bulk_pe (unstranded paired-end bulk RNA-seq; one observation "
            "per fragment, positioned at the fragment's polyA-distal base)."
        ),
        default="10x",
    )

    parser.add_argument(
        "--protocol_version",
        "-v",
        type=int,
        help="Version of the protocol used in library preparation.",
        default=3,
    )

    parser.add_argument(
        "--batch_merge_method",
        "-m",
        type=str,
        help="""
                        Method used to deal with different UTR coverage in different
                        batches. If `union`, then the tool outputs UTRs that are not
                        represented by all batches. If `intersection` only the reads that
                        map to UTRs in common between all batches will be used. Default
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
        help="""
                        Bandwidth of the Gaussian KDE that collapses UTR sizes into
                        consensus peaks. Applies to the standard peak-calling path and
                        to the KDE output of --raw_sizes. The default of 12 is tuned for
                        3'-end-enriched assays where reads pile within a few bp of a true
                        polyadenylation site; random-fragmentation libraries have a much
                        broader per-site spread and need a wider kernel, otherwise the
                        smear is carved into many closely-spaced pseudo-peaks.
                        """,
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
        default="UTR_counts.h5ad",
    )

    parser.add_argument(
        "--cores",
        type=int,
        default=None,
        help=(
            "Max worker processes for the per-UTR parallel loop. Defaults to "
            "JOBLIB_DEFAULT_N_JOBS (12) when omitted. Pass a non-positive "
            "value to use all-but-one core."
        ),
    )

    parser.set_defaults(func=call_peaks)
