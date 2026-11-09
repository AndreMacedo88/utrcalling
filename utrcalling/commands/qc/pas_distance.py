"""`utrcalling qc pas-distance`: distances of sites to polyadenylation signals."""

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
    """Measure each site's distance to the nearest polyadenylation signal (PAS).

    Finds the PAS hexamer AATAAA in each UTR. A site's distance is its UTR size
    minus the position where the nearest hexamer ends, so sites cleaved 10-30 nt
    after a hexamer, as expected, give 10 to 30. Writes a tab-separated table
    with a row per site: gene, UTR, UTR size, molecules, the nearest hexamer and
    the distance (empty for UTRs without a hexamer). Weight the distances by the
    molecules for their distribution over molecules.

    For example:

        utrcalling qc pas-distance sites.h5ad pas_distances.tsv GRCh38.fa -H 1

    Args:
        input_path: Count matrix (.h5ad) of sites or raw UTR sizes.
        out_path: Where to write the table (.tsv).
        genome_path: Genome sequence (FASTA) of the annotation the UTRs come from.
        hamming: How many mismatches to AATAAA a hexamer may have.
    """
    import pybedtools

    from utrcalling.core import helpers, storage
    from utrcalling.core import peak_calling_anndata as pc

    adata = helpers.check_if_path_and_open(input_path, method="h5ad")
    genome = pybedtools.BedTool(genome_path)
    table = pc.pas_distance_table(adata=adata, genome=genome, hamming=hamming, n_jobs=cores)
    storage.write_table(table, out_path)
