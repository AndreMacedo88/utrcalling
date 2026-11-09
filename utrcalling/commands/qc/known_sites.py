"""`utrcalling qc known-sites`: how far each site lies from known sites."""

from __future__ import annotations

from typing import Annotated, Literal

from cyclopts import Parameter


def known_sites(
    input_path: str,
    known_sites_path: str,
    out_path: str,
    *,
    window: Annotated[int, Parameter(name=("--window", "-w"))] = 20,
    utr_end: Literal["three_prime", "five_prime"] | None = None,
) -> None:
    """Measure each site's distance to the nearest known site of a database.

    For 3' data, known polyadenylation sites, such as the clusters of
    PolyASite 2.0 or PolyA_DB; for 5' data, known transcription start sites,
    such as refTSS. Sites are matched on chromosome (with or without "chr")
    and strand. Writes a tab-separated table with a row per site: its gene,
    UTR, UTR size, genome position, molecules, distance to the nearest known
    site, and whether that is within --window. Also logs the share of sites
    and of molecules near a known site.

    Sites far from known sites are not necessarily wrong: they may be new, or
    artefacts. Check them, for example their PAS hexamer (qc pas-distance).

    For example:

        utrcalling qc known-sites peaks.h5ad atlas.clusters.2.0.GRCh38.96.bed.gz known.tsv

    Args:
        input_path: Count matrix (.h5ad) at UTR level.
        known_sites_path: BED file (plain or gzipped) of known sites, from the
            same genome assembly; with their strand in the sixth column, or
            they count on both strands.
        out_path: Where to write the table (.tsv).
        window: Distance (nt) up to which a site is near a known site.
        utr_end: Which end of the UTR the sizes count from: three_prime for 3'
            protocols, five_prime for 5' ones. Only for matrices of utrcalling
            before 5.0.0: later ones record it.
    """
    from utrcalling.core import helpers, qc, storage
    from utrcalling.core.module_log import module_logger

    adata = helpers.check_if_path_and_open(input_path, method="h5ad")
    known = qc.read_known_sites(known_sites_path)
    table = qc.known_site_distances(adata, known, utr_end=utr_end, window=window)
    storage.write_table(table, out_path)
    near = table.near_known_site
    module_logger().info(
        f"Within {window} nt of a known site: {near.mean():.1%} of the sites, "
        f"{table.molecules[near].sum() / table.molecules.sum():.1%} of the molecules"
    )
