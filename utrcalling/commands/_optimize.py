"""`-b optimize` of `run`, `count` and `call-peaks`: choose the peak-calling setting by its score."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import anndata
    import pandas as pd

    from utrcalling.core.peak_calling.settings import Bandwidth


def check_kernels(kernels: Sequence[str], optimize: bool) -> None:
    """Refuse several kernels, unless -b optimize chooses among them.

    Raises:
        ValueError: If there are several kernels without -b optimize.
    """
    if len(kernels) > 1 and not optimize:
        raise ValueError(
            f"--kde-kernel {' '.join(kernels)}: give one kernel, or several with -b "
            "optimize to choose among them."
        )


def check(
    *,
    utr_end: str,
    scores: Sequence[str] | None,
    genome: str | None,
    known_sites: str | None,
    summarization_level: str | None,
) -> None:
    """Refuse -b optimize without what its scores need, before any work.

    Raises:
        ValueError: For gene-level sites, which have no genome positions, or if
            a score does not score this UTR end or lacks a file it needs (see
            peak_calling.scores.score_class).
        FileNotFoundError: If a file that a score needs does not exist.
    """
    from utrcalling.core.peak_calling.scores import ScoreContext, score_class
    from utrcalling.core.peak_calling.settings import DEFAULT_SCORES

    if summarization_level == "gene":
        raise ValueError(
            "-b optimize scores sites by where they lie in the genome, which gene-level "
            "sites have not (their sizes are positions along a gene's joined UTRs): use "
            "-s utr."
        )
    context = ScoreContext(utr_end=utr_end, genome=genome, known_sites=known_sites)
    for name in scores or [DEFAULT_SCORES[utr_end]]:
        score_class(name, context)


def choose(
    raw_sizes: anndata.AnnData,
    *,
    candidates: Sequence[Bandwidth] | None,
    kernels: Sequence[str],
    scores: Sequence[str] | None,
    genome: str | None,
    known_sites: str | None,
    utr_end: str,
    method: str,
    cores: int | None,
    scores_path: str,
) -> tuple[str, Bandwidth, pd.DataFrame]:
    """Score the candidate settings on raw sizes, write their scores, and pick the best.

    Args:
        raw_sizes: Raw-sizes count matrix.
        candidates: The bandwidths to score; by default those of the UTR end.
        kernels: The KDE kernels to score each bandwidth with.
        scores: The scores' names; the first chooses. By default, that of the
            UTR end.
        genome: Genome FASTA of the annotation, for the pas and initiator scores.
        known_sites: BED file of known sites, for the known-sites score.
        utr_end: Which end of the UTR the sizes count from.
        method: The peak caller's name.
        cores: Maximum number of worker processes.
        scores_path: Where to write the scores (.tsv).

    Returns:
        The best kernel and bandwidth, and the scores of all (see
        peak_calling_anndata.score_bandwidths) as the record keeps them: a row
        per setting, with the bandwidths as text.
    """
    from utrcalling.core import peak_calling_anndata, storage
    from utrcalling.core.module_log import module_logger

    logger = module_logger()
    logger.info("Scoring the candidate peak-calling settings")
    table = peak_calling_anndata.score_bandwidths(
        raw_sizes,
        genome,
        candidates,
        kernel=kernels,
        utr_end=utr_end,
        scores=scores,
        known_sites=known_sites,
        method=method,
        n_jobs=cores,
    )
    kernel, bandwidth = peak_calling_anndata.best_setting(table)
    storage.write_table(table, scores_path)
    logger.info(
        f"Scores of the settings:\n{table.to_string()}\n"
        f"Calling peaks with the {kernel} kernel and bandwidth {bandwidth}"
    )
    # h5ad files store tables of text, numbers and booleans, not mixed columns.
    record = table.reset_index()
    return kernel, bandwidth, record.astype({"bandwidth": str})
