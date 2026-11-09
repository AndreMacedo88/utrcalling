"""`utrcalling count`: the second half of the pipeline, from read mappings to UTR site counts."""

from __future__ import annotations

import os
from typing import Annotated

from cyclopts import Parameter

from utrcalling.commands.options import (
    INPUTS,
    OUTPUT,
    AnnotationsFile,
    BatchMergeMethod,
    Cores,
    KdeBandwidth,
    LabelBatches,
    LogLevel,
    NoLogFile,
    Out,
    ProtocolPlatform,
    ProtocolVersion,
    RawSizes,
    StoreIntermediate,
    SummarizationLevel,
)


def count(
    utr_annotation: Annotated[str, Parameter(group=INPUTS)],
    *,
    utr_sizes_counts: Annotated[
        list[str], Parameter(name=("--utr-sizes-counts", "-c"), consume_multiple=1, group=INPUTS)
    ],
    annotations_file: AnnotationsFile | None = None,
    protocol_platform: ProtocolPlatform = "10x",
    protocol_version: ProtocolVersion = 3,
    label_batches: LabelBatches = None,
    batch_merge_method: BatchMergeMethod = None,
    summarization_level: SummarizationLevel = "utr",
    raw_sizes: RawSizes = False,
    kde_bandwidth: KdeBandwidth = 12,
    out: Out = "UTR_counts.h5ad",
    store_intermediate: StoreIntermediate = False,
    log_level: LogLevel = "info",
    no_log_file: NoLogFile = False,
    cores: Annotated[Cores, Parameter(group=OUTPUT)] = None,
) -> None:
    """Count the molecules at each UTR site from the read mappings that map-reads stores.

    Does the last steps of `run` (merging samples, filtering and peak calling)
    from the files that `run -i` or `map-reads` store in
    ./intermediate/. Give it the same annotation file, protocol and
    --label-batches as the command that stored them.

    For example:

        utrcalling count intermediate/utr_reference_regions.parquet \\
            -c intermediate/UTR_sizes_counts_*.parquet -A experiment.csv

    Args:
        utr_annotation: UTR reference: the utr_reference_regions.parquet that run -i
            or map-reads stores (or a .pkl from older versions).
        utr_sizes_counts: Read mappings (.parquet): one file per BAM, or the merged
            file that run -i stores for several BAMs.
    """

    from utrcalling.core import core, helpers, protocol, storage
    from utrcalling.core.module_log import module_logger
    from utrcalling.core.protocols import peak_calling as pc

    # Set up logging
    store_log = not no_log_file
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
