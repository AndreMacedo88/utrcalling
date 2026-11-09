"""The second half of the pipeline, from read mappings to count matrices: `run` and `count`."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import anndata
    import pandas as pd

    from utrcalling.core.protocol import Protocol


def count_sites(
    read_mappings: pd.DataFrame | list[pd.DataFrame],
    utr_reference_regions: pd.DataFrame,
    protocol_: Protocol,
    *,
    inputs: Sequence[str],
    summarization_level: str,
    raw_sizes: bool,
    kde_bandwidth: int,
    out: str,
    store_intermediate: bool,
    cores: int | None,
    settings: dict[str, Any],
) -> None:
    """Merge the samples' read mappings, filter them and write the count matrices.

    Writes the sites (KDE peaks) to *out*. With *raw_sizes*, writes the raw UTR
    sizes to *out* and the peaks to ``<out stem>_kde.h5ad``. Each matrix records
    how it was made (see core.provenance), with *settings* and the mapping
    summary (see core.qc.mapping_summary), which also goes to
    ``<out stem>_mapping_summary.tsv``.

    Args:
        read_mappings: One table of read mappings per input, or one table.
        utr_reference_regions: The UTR reference the reads were mapped to.
        protocol_: The protocol the reads were mapped with.
        inputs: The input files of the read mappings, for the mapping summary.
        summarization_level: ``utr``, or ``gene`` to join the UTRs of each gene.
        raw_sizes: Whether to write the raw sizes too.
        kde_bandwidth: Bandwidth of the KDE that calls the peaks.
        out: Path of the count matrix (.h5ad).
        store_intermediate: Whether to store intermediate files in ./intermediate/.
        cores: Maximum number of worker processes.
        settings: The command's settings to record, such as its protocol.
    """
    from utrcalling.core import core, qc, storage
    from utrcalling.core.module_log import module_logger
    from utrcalling.core.protocol import ProtocolFivePrime
    from utrcalling.core.protocols import peak_calling as pc

    logger = module_logger()
    frames = read_mappings if isinstance(read_mappings, list) else [read_mappings]
    # Stored read mappings carry their mapping summary too (pandas 2.1 or later).
    mapped = [frame.attrs.get("mapping_summary", {"molecules": len(frame)}) for frame in frames]
    if len(frames) > 1:
        read_mappings = core.merge_samples_utr_sizes(
            utr_sizes_counts=frames, protocol=protocol_, store=store_intermediate
        )
    else:
        read_mappings = frames[0]

    if summarization_level.lower() != "utr":
        logger.info(f"Collapsing UTR size information to {summarization_level}-level")
        utr_reference_regions, read_mappings = core.merge_utrs_per_feature(
            utr_regions=utr_reference_regions,
            read_mappings=read_mappings,
            level=summarization_level,
        )
    counting = {"molecules_after_merge": len(read_mappings)}

    logger.info("Filtering mapped reads")
    filtered_reads, utr_reference_subset = core.filter_mapped_reads(
        utr_sizes_counts=read_mappings,
        utr_reference_regions=utr_reference_regions,
        protocol=protocol_,
        summary=counting,
    )
    out_stem, _ = os.path.splitext(out)
    settings = settings | {
        "summarization_level": summarization_level.lower(),
        # Which end of the UTR its sizes count from: the 5' end for 3' UTRs.
        "utr_end": "five_prime" if isinstance(protocol_, ProtocolFivePrime) else "three_prime",
    }

    def add_mapping_summary(adata: anndata.AnnData) -> None:
        """Count the molecules of the first matrix: raw sizes and peaks have the same."""
        counting["molecules_counted"] = int(adata.X.sum())
        summary = qc.mapping_summary(inputs, mapped, counting)
        storage.write_table(summary, f"{out_stem}_mapping_summary.tsv")
        logger.info("Mapping summary: %s", summary.loc["total"].to_dict())
        settings["mapping_summary"] = summary

    if raw_sizes:
        logger.info("Building raw sizes count matrix")
        raw_anndata = core.build_count_matrix(
            filtered_reads=filtered_reads,
            utr_reference_subset=utr_reference_subset,
            protocol=protocol_,
            store=store_intermediate,
            n_jobs=cores,
        )
        add_mapping_summary(raw_anndata)
        _write(raw_anndata, out, matrix="raw_sizes", **settings)
        del raw_anndata  # free memory before the KDE pass

        logger.info("Building KDE peaks count matrix")
        kde_out = f"{out_stem}_kde.h5ad"
        kde_anndata = core.build_count_matrix(
            filtered_reads=filtered_reads,
            utr_reference_subset=utr_reference_subset,
            protocol=protocol_,
            size_transform=pc.get_caller("kde", bandwidth=kde_bandwidth),
            n_jobs=cores,
        )
        _write(kde_anndata, kde_out, matrix="peaks", bandwidth=kde_bandwidth, **settings)
        logger.info(f"Wrote KDE peaks to {kde_out}")
    else:
        # The protocol's peak caller, which uses kde_bandwidth
        logger.info("Calling peaks from filtered data")
        count_matrix_anndata = core.call_peaks(
            filtered_reads=filtered_reads,
            utr_reference_subset=utr_reference_subset,
            protocol=protocol_,
            store=store_intermediate,
            n_jobs=cores,
        )
        add_mapping_summary(count_matrix_anndata)
        _write(count_matrix_anndata, out, matrix="peaks", bandwidth=kde_bandwidth, **settings)


def _write(adata: anndata.AnnData, path: str, **record: Any) -> None:
    """Log the matrix's size, record how it was made, and write it."""
    from utrcalling.core import helpers, provenance, storage

    helpers.log_metrics_anndata(adata)
    provenance.record(adata, **record)
    storage.write_anndata(adata, path)
