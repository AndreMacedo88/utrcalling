"""`utrcalling qc score-bandwidths`: how well each KDE setting places peaks at true sites."""

from __future__ import annotations

from typing import Annotated, Literal

from cyclopts import Parameter

from utrcalling.commands.options import (
    SCORES_HELP,
    Cores,
    KdeKernels,
    KnownSites,
    PeakCaller,
    ScoredBandwidths,
    Scores,
)


def score_bandwidths(
    input_path: str,
    out_path: str,
    genome_path: str | None = None,
    *,
    bandwidths: ScoredBandwidths = None,
    kde_kernel: KdeKernels = ("gaussian",),
    scores: Annotated[Scores, Parameter(help=f"The scores of the settings: {SCORES_HELP}")] = None,
    known_sites: KnownSites = None,
    window: Annotated[int, Parameter(name=("--window", "-w"))] = 10,
    hamming: Annotated[int, Parameter(name=("--hamming", "-H"))] = 1,
    peak_caller: PeakCaller = "kde",
    utr_end: Literal["three_prime", "five_prime"] | None = None,
    cores: Cores = None,
) -> None:
    """Score KDE settings by how well their peaks fall where true sites are.

    Calls peaks from the raw-sizes count matrix of `run -r` with each bandwidth
    and each kernel, then scores the sites of each setting by what true sites
    have. Only the sites with more than 1% of their UTR's molecules count; a
    setting's score averages the UTRs' shares of molecules at sites that have
    it. Higher scores are better.

    The scores (--score):

    - pas, the default for 3' data: the polyadenylation signal. The share of
      molecules at peaks 10 to 30 nt after the nearest hexamer AATAAA (with up
      to --hamming mismatches), in the genome.
    - initiator, the default for 5' data: the share of molecules at peaks with
      a pyrimidine (C or T) at -1 and a purine (A or G) at +1, the start site,
      in the genome. RNA polymerase II starts most transcripts at such a pair,
      mostly CA; smoothing that moves a peak off the start site loses it.
    - known-sites: the share of molecules at peaks within --window nt of the
      --known-sites of a database (PolyASite for 3' data, refTSS for 5' data).
      A database knows clusters of sites, not each nucleotide: smoothing that
      joins sites within a cluster raises this score, even when it moves the
      peak off the site. It compares settings at the scale of the clusters;
      for 5' data, the initiator chooses to the nucleotide.

    Writes a tab-separated table with a row per kernel and bandwidth: a column
    per score, the number of sites (peaks) and the sites per UTR. A setting
    that splits sites gives many sites. `run`, `count` and `call-peaks` choose
    the best setting themselves with `-b optimize`. Scores that a plugin adds
    work too (see docs/extending.md).

    For example:

        utrcalling qc score-bandwidths raw_sizes.h5ad scores.tsv GRCh38.fa -b 4 8 12 16
        utrcalling qc score-bandwidths capseq_raw.h5ad scores.tsv GRCm39.fa --kde-kernel gaussian linear
        utrcalling qc score-bandwidths capseq_raw.h5ad scores.tsv GRCm39.fa --score initiator known-sites --known-sites refTSS_v4.1_mouse_coordinate.mm39.bed.gz

    Args:
        input_path: Count matrix of raw UTR sizes (.h5ad), from run -r.
        out_path: Where to write the table (.tsv).
        genome_path: Genome sequence (FASTA) of the annotation the UTRs come
            from, for the pas and initiator scores.
        kde_kernel: Kernels of the KDE: each bandwidth is scored with each.
        window: Distance (nt) up to which a site is near a known site
            (known-sites).
        hamming: How many mismatches to AATAAA a hexamer may have (pas).
        utr_end: Which end of the UTR the sizes count from: three_prime for 3'
            protocols, five_prime for 5' ones. Only for matrices of utrcalling
            before 5.0.0, which are taken for 3' data: later ones record it.
    """
    from utrcalling.core import helpers, storage
    from utrcalling.core import peak_calling_anndata as pc

    adata = helpers.check_if_path_and_open(input_path, method="h5ad")
    table = pc.score_bandwidths(
        adata=adata,
        genome=genome_path,
        bandwidths=bandwidths,
        hamming=hamming,
        n_jobs=cores,
        kernel=kde_kernel,
        utr_end=utr_end,
        scores=scores,
        known_sites=known_sites,
        window=window,
        method=peak_caller,
    )
    storage.write_table(table, out_path)
