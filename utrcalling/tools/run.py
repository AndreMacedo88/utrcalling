# import general tools
from __future__ import annotations

import argparse
import ast
import os


def run(
    utr_annotation: str,
    alignments: list[str],
    annotations_file: str | None = None,
    batch_merge_method: str | None = None,
    feature: str = "three_prime_utr",
    protocol_platform: str = "10x",
    protocol_version: int = 3,
    label_batches: bool | str | None = None,
    strandedness: str | None = None,
    summarization_level: str = "UTR",
    raw_sizes: bool = False,
    kde_bandwidth: int = 12,
    store_intermediate: bool = False,
    log_level: str = "info",
    out: str = "UTR_counts.h5ad",
    store_log: bool = True,
    no_log_file: bool = False,
    cores: int | None = None,
) -> None:
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

    from utrcalling.core import core, helpers, protocol, storage
    from utrcalling.core.module_log import module_logger
    from utrcalling.core.protocols import peak_calling as pc

    # Set up logging
    store_log = store_log and not no_log_file
    log_dir = os.path.dirname(os.path.abspath(out)) if store_log else None
    logger = module_logger(disable=False, store_log=store_log, log_dir=log_dir, level=log_level)
    for arg, value in locals().items():
        logger.info(f"║ Argument {arg}: {value}")

    # Set up the protocol for this specific run
    protocol_ = protocol.create(
        protocol_platform,
        int(protocol_version),
        merge_method=batch_merge_method,
        call_peaks=not raw_sizes,
        label_batches=label_batches,
        alignments=alignments,
        annotations=annotations_file,
        strandedness=strandedness,
        kde_bandwidth=kde_bandwidth,
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
        n_jobs=cores,
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

    help = "Runs the entire UTR calling pipeline."

    description = f"{help} Exports the UTR sizes count matrix to a h5ad file."

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

    parser.add_argument("utr_annotation", type=str, help="GTF file with the genome annotations")

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
        "--strandedness",
        type=str,
        choices=["reverse", "forward", "unstranded"],
        default=None,
        help=(
            "Library strandedness for stranded bulk protocols (e.g. truseq). "
            "'reverse' (default for truseq; correct for TruSeq Stranded mRNA), "
            "'forward', or 'unstranded'. Ignored by protocols with a fixed "
            "strand convention."
        ),
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

    parser.add_argument(
        "--cores",
        type=int,
        default=None,
        help=(
            "Max worker processes for parallel sections. Defaults to "
            "JOBLIB_DEFAULT_N_JOBS (12) when omitted. Use this to lift the "
            "default cap on multi-core machines, or to constrain inside "
            "Snakemake. Pass a non-positive value to use all-but-one core."
        ),
    )

    parser.set_defaults(func=run)
