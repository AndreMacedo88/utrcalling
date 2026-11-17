"""The ``initiator`` score of 5' sites: the start-site dinucleotide of RNA polymerase II."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

from utrcalling.core import genome as genome_
from utrcalling.core.peak_calling.scores import (
    ScoreContext,
    SiteScore,
    share_at_frequent_sites,
    site_molecules,
)

if TYPE_CHECKING:
    import anndata

# The initiator: a pyrimidine at -1 and a purine at +1, the start site.
PYRIMIDINES = "CT"
PURINES = "AG"


class InitiatorScore(SiteScore):
    """Per UTR, the share of molecules at sites with C or T at -1 and A or G at +1.

    RNA polymerase II starts most transcripts at such a pyrimidine-purine (YR)
    dinucleotide, mostly CA (Carninci et al. 2006, Nat Genet 38:626; Vo ngoc et
    al. 2017, Genes Dev 31:6); a smoothing that moves a peak off the start site
    loses it. On 5' Cap-seq data (Eugenia Guerrero Aruffo's master's thesis,
    2026), 42% of the raw molecules start at a YR, 22% of the Gaussian peaks of
    12 nt, and 60% of the linear peaks of about 2 nt. UTRs on chromosomes that
    the genome lacks are left out.
    """

    utr_ends = ("five_prime",)
    needs = ("genome",)

    def __init__(self, raw_sizes: anndata.AnnData, context: ScoreContext) -> None:
        super().__init__(raw_sizes, context)
        self.genome = cast(str, context.genome)
        genome_.positioned_utr_end(raw_sizes, context.utr_end)  # refuses gene-level matrices

    def utr_scores(self, sites: anndata.AnnData) -> pd.Series:
        var = sites.var
        table = pd.DataFrame(
            {
                "chrom": var["chrom"].astype(str).to_numpy(),
                "position": genome_.site_positions(var, "five_prime"),
                "strand": var["strand"].astype(str).to_numpy(),
            }
        )
        dinucleotides, found = genome_.read_sequences(self.genome, table, -1, 1)
        initiator = np.array(
            [pair[0] in PYRIMIDINES and pair[1] in PURINES for pair in dinucleotides], dtype=bool
        )
        return share_at_frequent_sites(
            var["utr_name"].to_numpy()[found], site_molecules(sites)[found], initiator[found]
        )
