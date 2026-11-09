"""`utrcalling qc cells`: how much each cell tells about its UTR sizes."""

from __future__ import annotations

from typing import Annotated

from cyclopts import Parameter


def cells(
    input_path: str,
    out_path: str,
    *,
    min_molecules: Annotated[int, Parameter(name=("--min-molecules", "-m"))] = 5,
) -> None:
    """Describe each cell (or sample): its molecules, sites and measurable UTRs.

    Writes a tab-separated table with a row per cell: its molecules, the sites
    and UTRs with molecules, the UTRs measured (with at least --min-molecules),
    the UTRs with molecules at two or more sites, and its relative UTR size: the
    mean over its molecules of each one's UTR size divided by its UTR's mean
    size in all cells (above 1: longer UTRs than usual). Join it to the cells'
    annotations to filter or colour them. Also logs the Spearman correlation of
    the relative UTR size with the molecules: a strong one means UTR sizes
    differ with sequencing depth.

    For example:

        utrcalling qc cells peaks.h5ad cells.tsv -m 5

    Args:
        input_path: Count matrix (.h5ad) of raw sizes or peaks.
        out_path: Where to write the table (.tsv).
        min_molecules: Molecules a cell needs in a UTR to measure the UTR's size
            in that cell alone.
    """
    from utrcalling.core import helpers, qc, storage
    from utrcalling.core.module_log import module_logger

    adata = helpers.check_if_path_and_open(input_path, method="h5ad")
    table = qc.cell_metrics(adata, min_molecules=min_molecules)
    storage.write_table(table, out_path)
    rho, n_cells = qc.depth_correlation(table)
    module_logger().info(
        f"Correlation (Spearman) of the relative UTR size with the molecules of "
        f"{n_cells} cells: {rho:.2f}"
    )
