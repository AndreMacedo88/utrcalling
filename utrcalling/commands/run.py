"""`utrcalling run`: the whole pipeline, from aligned reads to UTR site counts."""

from __future__ import annotations

import os
from typing import Annotated

from cyclopts import Parameter

from utrcalling.commands.options import (
    INPUTS,
    OUTPUT,
    Alignments,
    AnnotationsFile,
    BatchMergeMethod,
    Cores,
    Feature,
    KdeBandwidth,
    LabelBatches,
    LogLevel,
    NoLogFile,
    Out,
    ProtocolPlatform,
    ProtocolVersion,
    RawSizes,
    StoreIntermediate,
    Strandedness,
    SummarizationLevel,
)


def run(
    utr_annotation: Annotated[str, Parameter(group=INPUTS)],
    *,
    alignments: Alignments,
    annotations_file: AnnotationsFile,
    feature: Feature = "three_prime_utr",
    protocol_platform: ProtocolPlatform = "10x",
    protocol_version: ProtocolVersion = 3,
    strandedness: Strandedness = None,
    label_batches: LabelBatches = None,
    batch_merge_method: BatchMergeMethod = None,
    summarization_level: SummarizationLevel = "utr",
    raw_sizes: RawSizes = False,
    kde_bandwidth: KdeBandwidth = 12,
    out: Out = "UTR_results.h5ad",
    store_intermediate: StoreIntermediate = False,
    log_level: LogLevel = "info",
    no_log_file: NoLogFile = False,
    cores: Annotated[Cores, Parameter(group=OUTPUT)] = None,
) -> None:
    """Run the whole pipeline: count the molecules at each UTR site, per cell or sample.

    Maps the reads of the BAM files to the UTRs of the GTF annotation, measures
    each molecule's UTR size, and collapses the sizes of each UTR into sites
    (peaks). Writes a cells (or samples) x sites count matrix.

    For example:

        utrcalling run mus_musculus.gtf -a wt.bam ko.bam -A experiment.csv -o counts.h5ad

    Args:
        utr_annotation: GTF file with the genome annotation (plain or gzipped).
    """

    from utrcalling.commands._count_sites import count_sites
    from utrcalling.core import core, protocol
    from utrcalling.core.module_log import module_logger

    # Set up logging
    store_log = not no_log_file
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
    read_mappings = core.map_aligned_reads(
        alignments=alignments,
        utr_regions=utr_reference_regions,
        protocol=protocol_,
        store=store_intermediate,
        n_jobs=cores,
    )

    count_sites(
        read_mappings,
        utr_reference_regions,
        protocol_,
        summarization_level=summarization_level,
        raw_sizes=raw_sizes,
        kde_bandwidth=kde_bandwidth,
        out=out,
        store_intermediate=store_intermediate,
        cores=cores,
        settings={"protocol_platform": protocol_platform, "protocol_version": protocol_version},
    )
