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
) -> None:
    """Remove the sites with few counts, keeping the most likely true sites.

    Keeps a site if its total count over all cells reaches the larger of
    threshold x its UTR's total count, and minimum_counts.

    For example:

        utrcalling process filter-sites sites.h5ad filtered.h5ad -t 0.05 -m 10

    Args:
        input_path: Count matrix (.h5ad).
        out_path: Where to write the filtered count matrix (.h5ad).
        threshold: Minimum share of its UTR's counts that a site needs (0.05 is 5%).
        minimum_counts: Minimum count that a site needs.
    """
    from utrcalling.core import helpers
    from utrcalling.core import peak_calling_anndata as pc

    adata = helpers.check_if_path_and_open(input_path, method="h5ad")

    adata_new = pc.remove_peaks_below_count(
        adata=adata, threshold=threshold, minimum_counts=minimum_counts
    )

    adata_new.write_h5ad(out_path)
