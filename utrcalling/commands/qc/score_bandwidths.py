"""`utrcalling qc score-bandwidths`: how well each KDE bandwidth places peaks at PAS hexamers."""

from __future__ import annotations

from utrcalling.commands.options import Bandwidths, Cores


def score_bandwidths(
    input_path: str,
    out_path: str,
    genome_path: str,
    *,
    bandwidths: Bandwidths = (1, 7, 12, 15),
    cores: Cores = None,
) -> None:
    """Score KDE bandwidths by how many peaks they place 10-30 nt after a PAS hexamer.

    Calls peaks from the raw-sizes count matrix of `run -r` with each bandwidth,
    then scores the bandwidth: per UTR, the fraction of molecules whose peak lies
    10 to 30 nt after the nearest polyadenylation signal hexamer (AATAAA, or one
    mismatch from it), averaged over all UTRs. Saves a {bandwidth: score}
    dictionary with joblib.

    For example:

        utrcalling qc score-bandwidths raw_sizes.h5ad scores.joblib GRCh38.fa -b 1 7 12

    Args:
        input_path: Count matrix of raw UTR sizes (.h5ad), from run -r.
        out_path: Where to write the scores (joblib file).
        genome_path: Genome sequence (FASTA) of the annotation the UTRs come from.
    """
    import joblib
    import pybedtools

    from utrcalling.core import helpers
    from utrcalling.core import peak_calling_anndata as pc

    adata = helpers.check_if_path_and_open(input_path, method="h5ad")
    genome = pybedtools.BedTool(genome_path)

    fit_score = pc.estimate_best_bandwidth(
        adata=adata,
        genome=genome,
        hamming=1,
        bandwidths=bandwidths,
        return_distances_to_hex=False,
        n_jobs=cores,
    )

    scores = {bandwidth: score for bandwidth, score in zip(bandwidths, fit_score, strict=True)}

    joblib.dump(scores, out_path)
