"""`utrcalling qc score-bandwidths`: how well each KDE bandwidth places peaks at PAS hexamers."""

from __future__ import annotations

from typing import Annotated

from cyclopts import Parameter

from utrcalling.commands.options import Bandwidths, Cores


def score_bandwidths(
    input_path: str,
    out_path: str,
    genome_path: str,
    *,
    bandwidths: Bandwidths = (1, 7, 12, 15),
    hamming: Annotated[int, Parameter(name=("--hamming", "-H"))] = 1,
    cores: Cores = None,
) -> None:
    """Score KDE bandwidths by how many peaks they place 10-30 nt after a PAS hexamer.

    Calls peaks from the raw-sizes count matrix of `run -r` with each bandwidth,
    then scores the bandwidth: per UTR, the fraction of molecules whose peak lies
    10 to 30 nt after the nearest polyadenylation signal hexamer (AATAAA, with
    up to --hamming mismatches), averaged over all UTRs. Writes a tab-separated table with
    a row per bandwidth: its score, its number of sites (peaks) and the sites per
    UTR. Higher scores are better; a bandwidth that splits peaks gives many sites.

    For example:

        utrcalling qc score-bandwidths raw_sizes.h5ad scores.tsv GRCh38.fa -b 1 7 12

    Args:
        input_path: Count matrix of raw UTR sizes (.h5ad), from run -r.
        out_path: Where to write the table (.tsv).
        genome_path: Genome sequence (FASTA) of the annotation the UTRs come from.
        hamming: How many mismatches to AATAAA a hexamer may have.
    """
    import pybedtools

    from utrcalling.core import helpers, storage
    from utrcalling.core import peak_calling_anndata as pc

    adata = helpers.check_if_path_and_open(input_path, method="h5ad")
    genome = pybedtools.BedTool(genome_path)
    table = pc.score_bandwidths(
        adata=adata, genome=genome, bandwidths=bandwidths, hamming=hamming, n_jobs=cores
    )
    storage.write_table(table, out_path)
