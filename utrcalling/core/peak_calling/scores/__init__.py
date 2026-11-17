"""Scores of peak-calling settings: how well their sites fall where true sites are.

`qc score-bandwidths` calls the peaks of a raw-sizes count matrix with each
setting (a kernel and a bandwidth) and scores the sites of each; `-b optimize`
calls the peaks with the setting of the highest score. A score gives each UTR
a score of its sites, higher for better sites, and the setting's score is the
mean over UTRs. utrcalling's scores (settings.SCORES, `--score`):

- ``pas`` (3' sites, the default for 3' data): the share of molecules at sites
  10-30 nt after a polyadenylation signal (see pas).
- ``initiator`` (5' sites, the default for 5' data): the share of molecules at
  sites with the initiator of RNA polymerase II (see initiator).
- ``known-sites`` (either): the share of molecules at sites near the known
  sites of a database, such as PolyASite for 3' sites or refTSS for 5' sites
  (see known_sites).

All three count only the sites with more than 1% of their UTR's molecules
(see :func:`share_at_frequent_sites`), as estimate_best_bandwidth always has.

A new score is a subclass of :class:`SiteScore`, registered by name; then
`--score <name>` uses it (see docs/extending.md)::

    from utrcalling.core.peak_calling.scores import SiteScore
    from utrcalling.core.peak_calling.settings import SCORES

    @SCORES.register("my-score")
    class MyScore(SiteScore):
        utr_ends = ("three_prime",)  # the sites it can score

        needs = ("genome",)  # what it needs of the context: see NEEDS

        def __init__(self, raw_sizes, context):
            super().__init__(raw_sizes, context)
            ...  # once: what does not depend on the setting

        def utr_scores(self, sites):
            ...  # per setting: a pandas Series, a score per UTR
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pandas as pd

from utrcalling.core.genome import UTR_ENDS
from utrcalling.core.peak_calling.settings import SCORES

if TYPE_CHECKING:
    import anndata
    import numpy.typing as npt

# The scores count only the sites with more than this share of their UTR's
# molecules: a smoothing that splits a site into a large one and crumbs keeps
# the score of the large one.
FREQUENT_SITE_SHARE = 0.01

# How the messages name the sites of each UTR end.
SITES = {"three_prime": "3'", "five_prime": "5'"}
# The files of the context that a score may need (SiteScore.needs): what each
# is, and the option that gives it.
NEEDS = {
    "genome": ("the genome sequence (FASTA) of the annotation", "--genome"),
    "known_sites": ("a BED file of known sites", "--known-sites"),
}


@dataclass(frozen=True)
class ScoreContext:
    """What a score can use besides the sites, as the command line gives it.

    Attributes:
        utr_end: Which end of the UTR the sizes count from: ``three_prime``
            for 3' sites, ``five_prime`` for 5' sites.
        genome: Genome FASTA of the annotation (`--genome`).
        known_sites: BED file of known sites (`--known-sites`).
        hamming: Mismatches to AATAAA that a PAS hexamer may have (pas).
        window: Distance (nt) up to which a site is near a known site
            (known-sites).
        n_jobs: Maximum number of worker processes.
    """

    utr_end: str
    genome: str | None = None
    known_sites: str | None = None
    hamming: int = 1
    window: int = 10
    n_jobs: int | None = None


class SiteScore(ABC):
    """A score of the sites that peak-calling settings call from one raw-sizes matrix.

    utrcalling creates the score once per matrix, then asks it for the UTRs'
    scores of each setting's sites. Do the work that does not depend on the
    setting, such as reading the genome around the UTRs, in ``__init__``.

    Args:
        raw_sizes: The raw-sizes count matrix whose peaks are called (cells x
            sizes), with ``utr_name``, ``utr_size``, ``chrom``, ``start``,
            ``end`` and ``strand`` in ``.var``.
        context: What the score can use besides the sites.
    """

    #: The UTR ends whose sites it scores.
    utr_ends: ClassVar[tuple[str, ...]] = UTR_ENDS
    #: The files of the context it needs (see NEEDS), which :func:`score_class`
    #: checks before any work.
    needs: ClassVar[tuple[str, ...]] = ()

    def __init__(self, raw_sizes: anndata.AnnData, context: ScoreContext) -> None:
        self.context = context

    @abstractmethod
    def utr_scores(self, sites: anndata.AnnData) -> pd.Series:
        """Each UTR's score of the sites of one setting: higher is better.

        Args:
            sites: The sites (peaks) of one setting, a count matrix like the
                raw sizes, with a site per column.

        Returns:
            A score per UTR, indexed by UTR name. NaN leaves a UTR out of the
            setting's score, as for UTRs that cannot be scored.
        """

    def score(self, sites: anndata.AnnData) -> float:
        """The setting's score: the mean of its UTRs' scores (NaN if none has one)."""
        return float(self.utr_scores(sites).mean())


def score_class(name: str, context: ScoreContext) -> type[SiteScore]:
    """The score called *name* (see settings.SCORES), checked against *context*.

    Checks, before any work, that the score scores the sites of the context's
    UTR end and that the context has the files it needs.

    Raises:
        ValueError: If no score has that name, it does not score the sites of
            the context's UTR end, or the context lacks a file it needs.
        FileNotFoundError: If a file it needs does not exist.
        TypeError: If what the name gives is not a SiteScore.
    """
    score_class_ = SCORES.get(name)
    if not (isinstance(score_class_, type) and issubclass(score_class_, SiteScore)):
        raise TypeError(f"score {name!r} must be a subclass of SiteScore, not {score_class_!r}")
    if context.utr_end not in score_class_.utr_ends:
        scored = " or ".join(SITES[end] for end in score_class_.utr_ends)
        raise ValueError(
            f"The {name} score scores {scored} sites, not the {SITES[context.utr_end]} "
            "sites of this matrix."
        )
    for need in score_class_.needs:
        what, option = NEEDS[need]
        path = getattr(context, need)
        if path is None:
            raise ValueError(f"The {name} score needs {what}: give it with {option}.")
        if not Path(path).exists():
            raise FileNotFoundError(f"{option} {path}: no such file")
    return score_class_


def make_score(name: str, raw_sizes: anndata.AnnData, context: ScoreContext) -> SiteScore:
    """The score called *name*, for the peaks of *raw_sizes* (see :func:`score_class`)."""
    return score_class(name, context)(raw_sizes, context)


def site_molecules(sites: anndata.AnnData) -> npt.NDArray[np.int64]:
    """Each site's molecules, in all cells."""
    return np.asarray(sites.X.sum(axis=0)).ravel().astype(np.int64)


def share_at_frequent_sites(
    utrs: npt.ArrayLike, molecules: npt.ArrayLike, hits: npt.ArrayLike
) -> pd.Series:
    """Per UTR, the share of its molecules at frequent sites that hit.

    A site is frequent with more than 1% of its UTR's molecules (more than
    ceil(1% of them)).

    Args:
        utrs: Each site's UTR.
        molecules: Each site's molecules.
        hits: Whether each site has what the score looks for.

    Returns:
        The share of each UTR, indexed by UTR name, in the order the UTRs
        first appear.
    """
    table = pd.DataFrame(
        {
            "utr": np.asarray(utrs).astype(str),
            "molecules": np.asarray(molecules, dtype=np.int64),
            "hit": np.asarray(hits, dtype=bool),
        }
    )
    by_utr = table.groupby("utr", sort=False)["molecules"]
    frequent = table["molecules"] > np.ceil(by_utr.transform("sum") * FREQUENT_SITE_SHARE)
    hit_molecules = table["molecules"].where(frequent & table["hit"], 0)
    shares = hit_molecules.groupby(table["utr"], sort=False).sum() / by_utr.sum()
    shares.index.name = "utr_name"
    return shares
