"""`utrcalling qc pas-distance`: distances of molecules to polyadenylation signals."""

from __future__ import annotations

from typing import Annotated

from cyclopts import Parameter

from utrcalling.commands.options import Cores


def pas_distance(
    input_path: str,
    out_path: str,
    genome_path: str,
    *,
    hamming: Annotated[int, Parameter(name=("--hamming", "-H"))] = 0,
    cores: Cores = None,
) -> None:
    """Measure each molecule's distance to the nearest polyadenylation signal (PAS).

    Finds the PAS hexamer AATAAA in each UTR. For every molecule, the distance
    is its UTR size minus the position where the nearest hexamer ends, so
    molecules cleaved 10-30 nt after a hexamer, as expected, give 10 to 30.
    Molecules of UTRs without a hexamer give none. Saves the distances as a
    numpy array with joblib.

    For example:

        utrcalling qc pas-distance sites.h5ad distances.joblib GRCh38.fa -H 1

    Args:
        input_path: Count matrix (.h5ad) of sites or raw UTR sizes.
        out_path: Where to write the distances (joblib file).
        genome_path: Genome sequence (FASTA) of the annotation the UTRs come from.
        hamming: How many mismatches to AATAAA a hexamer may have.
    """
    import joblib
    import pybedtools

    from utrcalling.core import helpers
    from utrcalling.core import peak_calling_anndata as pc

    adata = helpers.check_if_path_and_open(input_path, method="h5ad")
    genome = pybedtools.BedTool(genome_path)
    distances_array = pc.pas_distance(adata=adata, genome=genome, hamming=hamming, n_jobs=cores)

    joblib.dump(distances_array, out_path)
