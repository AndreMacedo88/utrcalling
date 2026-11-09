"""`utrcalling tools mean-sizes`: each cell's mean UTR size per UTR."""

from __future__ import annotations

from utrcalling.commands.options import UtrNameColumn


def mean_sizes(
    input_path: str, out_path: str, *, utr_name_column: UtrNameColumn = "utr_name"
) -> None:
    """Compute each cell's (or sample's) mean UTR size per UTR.

    Replaces the sites of each UTR with a single column: the mean size of the
    UTR's sites, weighted by their counts in that cell.

    Args:
        input_path: Count matrix (.h5ad).
        out_path: Where to write the mean sizes (.h5ad).
    """
    from utrcalling.core import anndata_processing as ap
    from utrcalling.core import helpers

    adata = helpers.check_if_path_and_open(input_path, method="h5ad")
    adata = ap.compute_utr_size_means(adata=adata, utr_name_column=utr_name_column)
    adata.write_h5ad(out_path)
