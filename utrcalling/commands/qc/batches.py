"""`utrcalling qc batches`: how alike the site usage of batches or samples is."""

from __future__ import annotations

from typing import Annotated

from cyclopts import Parameter


def batches(
    input_path: str,
    out_prefix: str,
    *,
    groupby: Annotated[str, Parameter(name=("--groupby", "-g"))],
    min_molecules: Annotated[int, Parameter(name=("--min-molecules", "-m"))] = 10,
) -> None:
    """Compare the site usage of groups of cells, such as batches or replicates.

    Sums the molecules of each group's cells and compares each pair of groups
    on the UTRs with two or more sites and at least --min-molecules in both.
    Writes two tab-separated tables:

    - `<out_prefix>_groups.tsv`, a row per group: its cells, molecules and
      sites, and the sites (and their molecules) that no other group has.
    - `<out_prefix>_pairs.tsv`, a row per pair of groups: the UTRs compared,
      the correlation of their sites' usage (each site's share of its UTR's
      molecules), and the usage difference: the share of a UTR's molecules at
      other sites in one group than in the other, on average.

    Replicates should correlate closely; a batch with many sites of its own,
    or far from the others, deserves a look before comparing conditions.

    For example:

        utrcalling qc batches peaks.h5ad batches -g sample

    Args:
        input_path: Count matrix (.h5ad) of raw sizes or peaks.
        out_prefix: Start of the two table names (a .tsv ending is dropped).
        groupby: Column of the cell annotations with each cell's group.
        min_molecules: Molecules a UTR needs in both groups to compare them.
    """
    from utrcalling.core import helpers, qc, storage

    adata = helpers.check_if_path_and_open(input_path, method="h5ad")
    groups, pairs = qc.batch_consistency(adata, groupby=groupby, min_molecules=min_molecules)
    out_prefix = out_prefix.removesuffix(".tsv")
    storage.write_table(groups, f"{out_prefix}_groups.tsv")
    storage.write_table(pairs, f"{out_prefix}_pairs.tsv")
