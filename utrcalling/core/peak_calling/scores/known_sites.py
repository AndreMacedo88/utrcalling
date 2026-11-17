"""The ``known-sites`` score: how many molecules lie at sites that a database knows."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from utrcalling.core import genome as genome_
from utrcalling.core import qc
from utrcalling.core.peak_calling.scores import (
    ScoreContext,
    SiteScore,
    share_at_frequent_sites,
)

if TYPE_CHECKING:
    import anndata
    import pandas as pd


class KnownSitesScore(SiteScore):
    """Per UTR, the share of molecules at sites within ``context.window`` nt of a known site.

    The known sites come from a database of the same genome assembly, as a BED
    file: for 3' sites, polyadenylation sites such as the clusters of
    PolyASite 2.0 or PolyA_DB; for 5' sites, transcription start sites such as
    refTSS. Sites match known sites on their chromosome (with or without
    "chr") and strand, as `qc known-sites` matches them. UTRs whose chromosome
    and strand have no known site are left out.

    A database knows the sites of the cells it was built from: it favours
    settings that call the sites it knows, and misses sites new to it. And it
    knows clusters of sites, not each nucleotide: refTSS's clusters span 14 nt
    in the median. Smoothing that joins a minor site outside a cluster to a
    major one inside raises the score, also when it moves the peak off the
    start site. On 5' Cap-seq data of neural stem cells (Eugenia Guerrero
    Aruffo's thesis, 2026), the score by refTSS rose from 0.77 to 0.83 between
    linear bandwidths of 1 and 12 nt, while the initiator score fell from 0.54
    to 0.47. So it compares settings at the scale of the clusters, and checks
    that sites lie where the database knows sites; for 5' data, the initiator
    chooses the bandwidth to the nucleotide.
    """

    needs = ("known_sites",)

    def __init__(self, raw_sizes: anndata.AnnData, context: ScoreContext) -> None:
        super().__init__(raw_sizes, context)
        genome_.positioned_utr_end(raw_sizes, context.utr_end)  # refuses gene-level matrices
        self.known_sites = qc.read_known_sites(cast(str, context.known_sites))

    def utr_scores(self, sites: anndata.AnnData) -> pd.Series:
        table = qc.known_site_distances(
            sites, self.known_sites, utr_end=self.context.utr_end, window=self.context.window
        )
        shares = share_at_frequent_sites(
            table["utr_name"], table["molecules"], table["near_known_site"]
        )
        has_known_sites = (
            table["known_site_distance"]
            .notna()
            .groupby(table["utr_name"].astype(str).to_numpy())
            .any()
        )
        return shares.where(has_known_sites.reindex(shares.index).to_numpy())
