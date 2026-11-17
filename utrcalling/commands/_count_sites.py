"""The second half of the pipeline, from read mappings to count matrices: `run` and `count`."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    import anndata
    import pandas as pd

    from utrcalling.core.peak_calling.settings import Bandwidth
    from utrcalling.core.protocol import Protocol


def count_sites(
    read_mappings: pd.DataFrame | list[pd.DataFrame],
    utr_reference_regions: pd.DataFrame,
    protocol_: Protocol,
    *,
    inputs: Sequence[str],
    summarization_level: str,
    raw_sizes: bool,
    kde_bandwidth: Bandwidth,
    kde_kernels: Sequence[str] = ("gaussian",),
    peak_caller: str = "kde",
    candidates: Sequence[Bandwidth] | None = None,
    scores: Sequence[str] | None = None,
    genome: str | None = None,
    known_sites: str | None = None,
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
        kde_bandwidth: Bandwidth of the KDE that calls the peaks: a number of
            nt, or a rule, whose value for each UTR goes to the sites'
            ``bandwidth`` column; or ``optimize``, to choose the best of
            *candidates* with each of *kde_kernels* by their *scores* (see
            commands._optimize), written to ``<out stem>_bandwidth_scores.tsv``.
        kde_kernels: Kernel of the KDE that calls the peaks; several only with
            ``optimize``, which chooses among them.
        peak_caller: The peak caller's name (see peak_calling.callers).
        candidates: The bandwidths that ``optimize`` chooses from; by default
            those of the UTR end (settings.CANDIDATE_BANDWIDTHS).
        scores: The scores of ``optimize``; the first chooses. By default, that
            of the UTR end.
        genome: Genome FASTA, for ``optimize`` with the pas or initiator score.
        known_sites: BED file of known sites, for the known-sites score.
        out: Path of the count matrix (.h5ad).
        store_intermediate: Whether to store intermediate files in ./intermediate/.
        cores: Maximum number of worker processes.
        settings: The command's settings to record, such as its protocol.
    """
    import numpy as np

    from utrcalling.commands import _optimize
    from utrcalling.core import core, count_matrix, qc, storage
    from utrcalling.core.module_log import module_logger
    from utrcalling.core.peak_calling.callers import get_caller
    from utrcalling.core.peak_calling.settings import OPTIMIZE

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
    out_stem = str(Path(out).with_suffix(""))
    settings = settings | {
        "summarization_level": summarization_level.lower(),
        # Which end of the UTR its sizes count from: the 5' end for 3' UTRs.
        "utr_end": protocol_.utr_end,
    }

    def add_mapping_summary(adata: anndata.AnnData) -> None:
        """Count the molecules of the first matrix: raw sizes and peaks have the same."""
        counting["molecules_counted"] = int(adata.X.sum())
        summary = qc.mapping_summary(inputs, mapped, counting)
        storage.write_table(summary, f"{out_stem}_mapping_summary.tsv")
        logger.info("Mapping summary: %s", summary.loc["total"].to_dict())
        settings["mapping_summary"] = summary

    raw_anndata = None
    if raw_sizes or kde_bandwidth == OPTIMIZE:
        logger.info("Building raw sizes count matrix")
        raw_anndata = core.build_count_matrix(
            filtered_reads=filtered_reads,
            utr_reference_subset=utr_reference_subset,
            protocol=protocol_,
            store=store_intermediate and raw_sizes,
            n_jobs=cores,
        )
        add_mapping_summary(raw_anndata)

    kde_kernel = kde_kernels[0]
    peak_calling: dict[str, Any] = {}
    if kde_bandwidth == OPTIMIZE:
        # run and count check the scores' files first; the raw sizes are built above.
        kde_kernel, kde_bandwidth, peak_calling["bandwidth_scores"] = _optimize.choose(
            cast("anndata.AnnData", raw_anndata),
            candidates=candidates,
            kernels=kde_kernels,
            scores=scores,
            genome=genome,
            known_sites=known_sites,
            utr_end=settings["utr_end"],
            method=peak_caller,
            cores=cores,
            scores_path=f"{out_stem}_bandwidth_scores.tsv",
        )
    peak_calling |= {"peak_caller": peak_caller, "kernel": kde_kernel, "bandwidth": kde_bandwidth}
    if raw_anndata is not None and raw_sizes:
        _write(raw_anndata, out, matrix="raw_sizes", **settings)
    del raw_anndata  # free memory before the KDE pass

    logger.info("Calling peaks from filtered data")
    peaks = core.build_count_matrix(
        filtered_reads=filtered_reads,
        utr_reference_subset=utr_reference_subset,
        protocol=protocol_,
        store=store_intermediate and not raw_sizes,
        size_transform=get_caller(peak_caller, bandwidth=kde_bandwidth, kernel=kde_kernel),
        n_jobs=cores,
    )
    if "molecules_counted" not in counting:  # no raw sizes counted them
        add_mapping_summary(peaks)
    if isinstance(kde_bandwidth, str):  # a rule: each site gets its UTR's bandwidth
        per_utr = count_matrix.utr_bandwidths(filtered_reads, kde_bandwidth)
        peaks.var["bandwidth"] = peaks.var["utr_name"].map(per_utr).to_numpy(np.float64)
    peaks_out = f"{out_stem}_kde.h5ad" if raw_sizes else out
    _write(peaks, peaks_out, matrix="peaks", **peak_calling, **settings)
    logger.info(f"Wrote KDE peaks to {peaks_out}")


def _write(adata: anndata.AnnData, path: str, **record: Any) -> None:
    """Log the matrix's size, record how it was made, and write it."""
    from utrcalling.core import helpers, provenance, storage

    helpers.log_metrics_anndata(adata)
    provenance.record(adata, **record)
    storage.write_anndata(adata, path)
