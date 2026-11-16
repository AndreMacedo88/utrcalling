"""`-b optimize` of `run`, `count` and `call-peaks`: choose the KDE bandwidth by its score."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import anndata
    import pandas as pd

    from utrcalling.core.protocols.kde_options import Bandwidth


def check(genome: str | None, summarization_level: str | None) -> None:
    """Refuse -b optimize without what it needs, before any work.

    Raises:
        ValueError: Without a genome, or for gene-level sites, which have no
            genome positions.
        FileNotFoundError: If the genome file does not exist.
    """
    if genome is None:
        raise ValueError(
            "-b optimize scores the bandwidths by the genome around the sites: give "
            "the genome (FASTA) with --genome."
        )
    if not os.path.exists(genome):
        raise FileNotFoundError(f"--genome {genome}: no such file")
    if summarization_level == "gene":
        raise ValueError(
            "-b optimize scores sites by the genome around them, which gene-level sites "
            "have not (their sizes are positions along a gene's joined UTRs): use -s utr."
        )


def choose(
    raw_sizes: anndata.AnnData,
    genome: str,
    candidates: Sequence[Bandwidth] | None,
    kernel: str,
    utr_end: str,
    cores: int | None,
    scores_path: str,
) -> tuple[Bandwidth, pd.DataFrame]:
    """Score the candidate bandwidths on raw sizes, write their scores, and pick the best.

    Args:
        raw_sizes: Raw-sizes count matrix.
        genome: Genome FASTA of the annotation.
        candidates: The bandwidths to score; by default those of the UTR end.
        kernel: The KDE's kernel.
        utr_end: Which end of the UTR the sizes count from.
        cores: Maximum number of worker processes.
        scores_path: Where to write the scores (.tsv).

    Returns:
        The best bandwidth, and the scores of all (see
        peak_calling_anndata.score_bandwidths), indexed by the bandwidths as text.
    """
    from utrcalling.core import peak_calling_anndata, storage
    from utrcalling.core.module_log import module_logger

    logger = module_logger()
    logger.info("Scoring the candidate KDE bandwidths against the genome")
    scores = peak_calling_anndata.score_bandwidths(
        raw_sizes, genome, candidates, kernel=kernel, utr_end=utr_end, n_jobs=cores
    )
    chosen = peak_calling_anndata.best_bandwidth(scores)
    storage.write_table(scores, scores_path)
    logger.info(f"Bandwidth scores:\n{scores.to_string()}\nCalling peaks with bandwidth {chosen}")
    # h5ad files store tables with text indexes.
    return chosen, scores.set_axis(scores.index.map(str))
