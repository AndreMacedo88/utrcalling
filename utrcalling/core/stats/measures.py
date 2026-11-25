"""How a UTR uses its sites: the length index and the entropy of site use.

The *length index* of a UTR in a cell or sample is the mean rank of its
molecules' sites: each site's rank by UTR size, scaled from 0 (the shortest
UTR isoform) to 1 (the longest), weighted by its share of the molecules. It
is the Weighted UTR Index of scUTRquant (Fansler, Mitschka and Mayr 2024,
Nat Commun 15:4050), after the psi of LABRAT (Goering et al. 2021, BMC
Genomics 22:476). For 3' data a UTR size counts from the stop codon, for 5'
data from the start codon, so in both a larger index means longer UTRs: more
distal polyadenylation sites, or more upstream transcription start sites.

The *entropy* of site use is the Shannon entropy of the sites' shares divided
by its largest possible value, log K for K sites: 0 when one site takes all
molecules, 1 when all sites take the same share.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import numpy.typing as npt


def site_ranks(utr_names: npt.ArrayLike, utr_sizes: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Each site's rank among its UTR's sites by size, scaled from 0 to 1.

    The shortest site of a UTR gets 0 and the longest 1; equal sizes share
    their mean rank. Sites of single-site UTRs get NaN: they have no rank.

    Args:
        utr_names: The UTR of each site.
        utr_sizes: The UTR size of each site.

    Returns:
        The scaled ranks, in the order of the sites.
    """
    sites = pd.DataFrame({"utr": np.asarray(utr_names), "size": np.asarray(utr_sizes, float)})
    grouped = sites.groupby("utr", sort=False, observed=True)["size"]
    ranks = grouped.rank(method="average").to_numpy(float) - 1
    sites_per_utr = grouped.transform("size").to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sites_per_utr > 1, ranks / (sites_per_utr - 1), np.nan)


def length_index(proportions: npt.ArrayLike, ranks: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """The length index of each row of site shares: their mean scaled rank.

    Args:
        proportions: Shares of a UTR's sites, a row per cell, sample or
            condition (rows add up to 1).
        ranks: The sites' scaled ranks (see :func:`site_ranks`).

    Returns:
        The length index of each row.
    """
    return np.asarray(proportions, float) @ np.asarray(ranks, float)


def normalized_entropy(proportions: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """The entropy of each row of site shares, divided by log K for K sites.

    Args:
        proportions: Shares of a UTR's K sites (K of 2 or more), a row per
            cell, sample or condition (rows add up to 1).

    Returns:
        The normalised entropy of each row, from 0 (one site) to 1 (equal shares).
    """
    shares = np.asarray(proportions, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(shares > 0, -shares * np.log(shares), 0.0)
    return terms.sum(axis=1) / np.log(shares.shape[1])
