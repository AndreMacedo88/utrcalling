"""`utrcalling process concatenate`: the cells of several count matrices in one."""

from __future__ import annotations

from typing import Annotated

from cyclopts import Parameter


def concatenate_experiments(
    *,
    input_paths: Annotated[list[str], Parameter(name=("--input-paths", "-i"), consume_multiple=1)],
    out_path: Annotated[str, Parameter(name=("--out-path", "-o"))],
) -> None:
    """Join the cells of several experiments into one count matrix.

    Sites are matched by UTR and size. A site that only some experiments have
    counts 0 in the cells of the others. The sites are then named again.

    For example:

        utrcalling process concatenate -i batch1.h5ad batch2.h5ad -o both.h5ad

    Args:
        input_paths: Count matrices (.h5ad) to join.
        out_path: Where to write the joined count matrix (.h5ad).
    """
    from utrcalling.core import anndata_processing as ap
    from utrcalling.core import helpers

    adata_list = [helpers.check_if_path_and_open(path_, method="h5ad") for path_ in input_paths]
    adata_result = ap.concatenate_adata(adata_list=adata_list)
    adata_result.write_h5ad(out_path)
