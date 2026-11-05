"""`utrcalling prepare-reference-and-map-reads`: the first half of the pipeline."""

from __future__ import annotations

import os
from typing import Annotated

from cyclopts import Parameter

from utrcalling.tools.options import (
    INPUTS,
    Alignments,
    AnnotationsFile,
    Feature,
    LabelBatches,
    LogLevel,
    NoLogFile,
    ProtocolPlatform,
    ProtocolVersion,
    Strandedness,
)


def generate_utr_reference_and_map_reads(
    utr_annotation: Annotated[str, Parameter(group=INPUTS)],
    *,
    alignments: Alignments,
    annotations_file: AnnotationsFile,
    feature: Feature = "three_prime_utr",
    protocol_platform: ProtocolPlatform = "10x",
    protocol_version: ProtocolVersion = 3,
    strandedness: Strandedness = None,
    label_batches: LabelBatches = None,
    log_level: LogLevel = "info",
    no_log_file: NoLogFile = False,
) -> None:
    """Build the UTR reference and map the reads to it, for call-peaks to continue.

    Stores the UTR reference and each BAM's read mappings in ./intermediate/.
    Give call-peaks the same annotation file and --label-batches: they decide
    the observation names stored with the mappings.

    For example:

        utrcalling prepare-reference-and-map-reads mus_musculus.gtf -a wt.bam ko.bam -A experiment.csv

    Args:
        utr_annotation: GTF file with the genome annotation (plain or gzipped).
    """

    from utrcalling.core import core, protocol
    from utrcalling.core.module_log import module_logger

    # Set up logging
    store_log = not no_log_file
    log_dir = os.getcwd() if store_log else None
    logger = module_logger(disable=False, store_log=store_log, log_dir=log_dir, level=log_level)
    for arg, value in locals().items():
        logger.info(f"║ Argument {arg}: {value}\n")

    # Set up the protocol for this specific run
    protocol_ = protocol.create(
        protocol_platform,
        int(protocol_version),
        label_batches=label_batches,
        alignments=alignments,
        annotations=annotations_file,
        strandedness=strandedness,
    )

    # Get UTR reference data from the organisms GTF annotation file
    logger.info("Preparing utr reference regions")
    utr_reference_regions = core.prepare_utr_reference_regions(utr_annotation, feature, store=True)

    # Map to the reference UTR the reads that properly aligned to genome
    logger.info("Mapping aligned reads to UTR reference")
    core.map_aligned_reads(alignments, utr_reference_regions, protocol_, store=True)
