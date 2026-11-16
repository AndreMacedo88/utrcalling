"""`utrcalling qc score-bandwidths`: how well each KDE bandwidth places peaks at true sites."""

from __future__ import annotations

from typing import Annotated, Literal

from cyclopts import Parameter

from utrcalling.commands.options import CandidateBandwidths, Cores, KdeKernel


def score_bandwidths(
    input_path: str,
    out_path: str,
    genome_path: str,
    *,
    bandwidths: Annotated[CandidateBandwidths, Parameter(name=("--bandwidths", "-b"))] = None,
    hamming: Annotated[int, Parameter(name=("--hamming", "-H"))] = 1,
    kde_kernel: KdeKernel = "gaussian",
    utr_end: Literal["three_prime", "five_prime"] | None = None,
    cores: Cores = None,
) -> None:
    """Score KDE bandwidths by how well their peaks fall where the genome expects sites.

    Calls peaks from the raw-sizes count matrix of `run -r` with each bandwidth,
    then scores them by a sequence that true sites have. Only the sites with
    more than 1% of their UTR's molecules count, and the score averages the
    UTRs' shares of molecules at sites that have it. Higher scores are better.

    3' data: the polyadenylation signal. The share of molecules at peaks 10 to
    30 nt after the nearest hexamer AATAAA (with up to --hamming mismatches).

    5' data: the initiator. The share of molecules at peaks with a pyrimidine
    (C or T) at -1 and a purine (A or G) at +1, the start site: RNA polymerase
    II starts most transcripts at such a pair, mostly CA. Smoothing that moves
    a peak off the start site loses it. Compare kernels by scoring each: the
    linear kernel keeps peaks on the sizes where the molecules are.

    Writes a tab-separated table with a row per bandwidth: the kernel, the
    score, the number of sites (peaks) and the sites per UTR. A bandwidth that
    splits sites gives many sites.

    For example:

        utrcalling qc score-bandwidths raw_sizes.h5ad scores.tsv GRCh38.fa -b 4 8 12 16
        utrcalling qc score-bandwidths capseq_raw.h5ad scores.tsv GRCm39.fa --kde-kernel linear

    Args:
        input_path: Count matrix of raw UTR sizes (.h5ad), from run -r.
        out_path: Where to write the table (.tsv).
        genome_path: Genome sequence (FASTA) of the annotation the UTRs come from.
        hamming: How many mismatches to AATAAA a hexamer may have (3' data).
        utr_end: Which end of the UTR the sizes count from: three_prime for 3'
            protocols, five_prime for 5' ones. Only for matrices of utrcalling
            before 5.0.0, which are taken for 3' data: later ones record it.
    """
    import pybedtools

    from utrcalling.core import helpers, storage
    from utrcalling.core import peak_calling_anndata as pc

    adata = helpers.check_if_path_and_open(input_path, method="h5ad")
    genome = pybedtools.BedTool(genome_path)
    table = pc.score_bandwidths(
        adata=adata,
        genome=genome,
        bandwidths=bandwidths,
        hamming=hamming,
        n_jobs=cores,
        kernel=kde_kernel,
        utr_end=utr_end,
    )
    storage.write_table(table, out_path)
