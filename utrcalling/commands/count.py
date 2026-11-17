"""`utrcalling count`: the second half of the pipeline, from read mappings to UTR site counts."""

from __future__ import annotations

import os
from typing import Annotated

from cyclopts import Parameter

from utrcalling.commands.options import (
    INPUTS,
    OUTPUT,
    PEAK_CALLING,
    AnnotationsFile,
    BatchMergeMethod,
    CandidateBandwidths,
    Cores,
    Genome,
    KdeBandwidth,
    KdeKernels,
    KnownSites,
    LabelBatches,
    LogLevel,
    NoLogFile,
    Out,
    PeakCaller,
    ProtocolPlatform,
    ProtocolVersion,
    RawSizes,
    Scores,
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
    kde_kernel: Annotated[KdeKernels, Parameter(group=PEAK_CALLING)] = ("gaussian",),
    peak_caller: Annotated[PeakCaller, Parameter(group=PEAK_CALLING)] = "kde",
    candidates: Annotated[CandidateBandwidths, Parameter(group=PEAK_CALLING)] = None,
    scores: Annotated[Scores, Parameter(group=PEAK_CALLING)] = None,
    genome: Annotated[Genome, Parameter(group=PEAK_CALLING)] = None,
    known_sites: Annotated[KnownSites, Parameter(group=PEAK_CALLING)] = None,
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

    from utrcalling.commands import _optimize
    from utrcalling.commands._count_sites import count_sites
    from utrcalling.core import helpers, protocol
    from utrcalling.core.module_log import module_logger
    from utrcalling.core.peak_calling.settings import OPTIMIZE

    optimize = kde_bandwidth == OPTIMIZE
    _optimize.check_kernels(kde_kernel, optimize)
    if optimize:
        _optimize.check(
            utr_end=protocol.protocol_class(protocol_platform, int(protocol_version)).utr_end,
            scores=scores,
            genome=genome,
            known_sites=known_sites,
            summarization_level=summarization_level,
        )

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
        call_peaks=not (raw_sizes or optimize),  # optimize calls them itself
        annotations=annotations_file,
        kde_bandwidth=kde_bandwidth,
        kde_kernel=kde_kernel[0],
        call_peaks_method=peak_caller,
    )

    utr_reference_regions = helpers.load_annotations(utr_annotation)
    read_mappings = [
        helpers.check_if_path_and_open(path, method="parquet") for path in utr_sizes_counts
    ]

    count_sites(
        read_mappings,
        utr_reference_regions,
        protocol_,
        inputs=utr_sizes_counts,
        summarization_level=summarization_level,
        raw_sizes=raw_sizes,
        kde_bandwidth=kde_bandwidth,
        kde_kernels=kde_kernel,
        peak_caller=peak_caller,
        candidates=candidates,
        scores=scores,
        genome=genome,
        known_sites=known_sites,
        out=out,
        store_intermediate=store_intermediate,
        cores=cores,
        settings={"protocol_platform": protocol_platform, "protocol_version": protocol_version},
    )
