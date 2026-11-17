"""The ``pas`` score of 3' sites: the polyadenylation signal 10-30 nt before them."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from utrcalling.core import genome as genome_
from utrcalling.core import helpers, module_log
from utrcalling.core import peak_calling_anndata as pca
from utrcalling.core.helpers import JOBLIB_DEFAULT_BACKEND
from utrcalling.core.peak_calling.scores import (
    ScoreContext,
    SiteScore,
    share_at_frequent_sites,
    site_molecules,
)

if TYPE_CHECKING:
    import anndata
    import numpy.typing as npt

# Where the cleavage site lies after the end of its polyadenylation signal.
PAS_DISTANCES = (10, 30)


class PasScore(SiteScore):
    """Per UTR, the share of molecules at sites 10-30 nt after the nearest PAS hexamer.

    The polyadenylation signal AATAAA (with up to ``context.hamming``
    mismatches) lies 10-30 nt before the cleavage site; a smoothing that moves
    a 3' site off it loses it. The score of estimate_best_bandwidth (Macedo et
    al. 2026): UTRs without a hexamer score 0, and those on chromosomes that
    the genome lacks are left out. Each UTR's hexamers are found once, in
    ``__init__``.
    """

    utr_ends = ("three_prime",)
    needs = ("genome",)

    def __init__(self, raw_sizes: anndata.AnnData, context: ScoreContext) -> None:
        super().__init__(raw_sizes, context)
        genome_.positioned_utr_end(raw_sizes, context.utr_end)  # refuses gene-level matrices
        self.pas_ends = pas_ends_per_utr(
            raw_sizes, cast(str, context.genome), hamming=context.hamming, n_jobs=context.n_jobs
        )

    def utr_scores(self, sites: anndata.AnnData) -> pd.Series:
        var = sites.var
        sizes = var["utr_size"].to_numpy(dtype=np.int64)
        hits = np.zeros(sites.n_vars, dtype=bool)
        for utr, columns in var.groupby("utr_name", observed=True).indices.items():
            ends = self.pas_ends.get(str(utr))
            if ends is None or len(ends) == 0:
                continue
            distances = sizes[columns, np.newaxis] - ends[np.newaxis, :]
            nearest = distances[np.arange(len(columns)), np.abs(distances).argmin(axis=1)]
            hits[columns] = (nearest >= PAS_DISTANCES[0]) & (nearest <= PAS_DISTANCES[1])
        scored = var["utr_name"].astype(str).isin(self.pas_ends.keys()).to_numpy()
        return share_at_frequent_sites(
            var["utr_name"].to_numpy()[scored], site_molecules(sites)[scored], hits[scored]
        )


def pas_ends_per_utr(
    adata: anndata.AnnData, genome: str, hamming: int, n_jobs: int | None
) -> dict[str, npt.NDArray[np.int64]]:
    """The end of each PAS hexamer of each UTR (see peak_calling_anndata._find_pas).

    Chromosomes match the genome's with or without "chr". UTRs on chromosomes
    that the genome lacks have no entry, with a warning.

    Raises:
        ValueError: If the genome has no chromosome of the matrix.
    """
    names = genome_.chromosome_names(genome, adata.var["chrom"].astype(str).unique())
    per_utr_inputs = []
    lacking = 0
    for orig_idx, utr_name, X_sub, var_sub in pca._extract_per_utr_inputs(adata):
        name = names[str(var_sub["chrom"].iloc[0])]
        if name is None:
            lacking += 1
            continue
        per_utr_inputs.append((orig_idx, utr_name, X_sub, var_sub.assign(chrom=name)))
    if lacking:
        module_log.module_logger().warning(
            f"The PAS score leaves out {lacking} UTRs on chromosomes that the genome lacks"
        )
    n_workers = helpers.resolve_n_jobs(n_jobs)
    batches = pca._build_compact_batches(per_utr_inputs, n_workers)
    try:
        with joblib.parallel_config(backend=JOBLIB_DEFAULT_BACKEND, inner_max_num_threads=1):
            batch_results = list(
                joblib.Parallel(n_jobs=n_workers, return_as="generator")(
                    joblib.delayed(_pas_ends_utr_batch)(batch, hamming, genome)
                    for batch in tqdm(batches, colour="#ffb700", ascii=" ■", desc="UTR: ")
                )
            )
    finally:
        helpers.shutdown_joblib_pool()
    return dict(item for result in batch_results for item in result)


def _pas_ends_utr_batch(
    batch: list[pca.CompactUtrItem], hamming: int, genome: str
) -> list[tuple[str, npt.NDArray[np.int64]]]:
    with genome_.open_genome(genome) as fasta:  # once per batch
        return [
            (
                str(utr_name),
                np.asarray(
                    pca._find_pas(var_sub, hamming=hamming, genome=fasta)[0], dtype=np.int64
                ),
            )
            for _orig_idx, utr_name, _X_sub, var_sub in batch
        ]
