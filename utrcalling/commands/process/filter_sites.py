"""`utrcalling process filter-sites`: remove the sites with few counts."""

from __future__ import annotations

from typing import Annotated

from cyclopts import Parameter


def filter_sites(
    input_path: str,
    out_path: str,
    *,
    threshold: Annotated[float, Parameter(name=("--threshold", "-t"))] = 0.05,
    minimum_counts: Annotated[int, Parameter(name=("--minimum-counts", "-m"))] = 10,
    groupby: Annotated[str | None, Parameter(name=("--groupby", "-g"))] = None,
) -> None:
    """Remove the sites with few counts, keeping the most likely true sites.

    Keeps a site if its total count over all cells reaches the larger of
    threshold x its UTR's total count, and minimum_counts.

    With --groupby, counts per group of cells (a column of the cell or sample
    annotations, such as a treatment), and keeps a site that reaches its
    group's threshold in at least one group: a site that a condition raises
    then stays, though it is a small share of all counts. The minimum count
    holds in each group, so a site spread thinly over the groups can go,
    though all cells together would keep it. Cells without a group count in
    none.

    For example:

        utrcalling process filter-sites sites.h5ad filtered.h5ad -t 0.05 -m 10
        utrcalling process filter-sites sites.h5ad filtered.h5ad -t 0.1 -m 10 -g Treatment

    Args:
        input_path: Count matrix (.h5ad).
        out_path: Where to write the filtered count matrix (.h5ad).
        threshold: Minimum share of its UTR's counts that a site needs (0.05 is 5%).
        minimum_counts: Minimum count that a site needs.
        groupby: Column of the cell annotations (.obs) whose groups of cells
            each count on their own.
    """
    from utrcalling.core import helpers, provenance
    from utrcalling.core import peak_calling_anndata as pc
    from utrcalling.core.module_log import module_logger

    adata = helpers.check_if_path_and_open(input_path, method="h5ad")

    adata_new = pc.remove_peaks_below_count(
        adata=adata, threshold=threshold, minimum_counts=minimum_counts, group_by=groupby
    )
    module_logger().info(f"Kept {adata_new.n_vars} of {adata.n_vars} sites")

    provenance.record(adata_new, source=adata)
    adata_new.write_h5ad(out_path)
