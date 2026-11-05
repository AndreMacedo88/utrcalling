"""`utrcalling process rename`: number each gene's sites again, after filtering."""

from __future__ import annotations

from typing import Annotated

from cyclopts import Parameter

from utrcalling.tools.options import UtrNameColumn


def generate_names(
    input_path: str,
    out_path: str,
    *,
    utr_name_column: UtrNameColumn = "utr_name",
    gene_name_column: Annotated[str, Parameter(name=("--gene-name-column", "-g"))] = "gene_name",
) -> None:
    """Number each gene's sites 1, 2, ... again, for example after filtering.

    utrcalling names sites `<gene>_<n>-<UTR size in bp>` when it writes them,
    where n numbers the sites of each gene. Filtering removes sites and leaves gaps in
    n: this numbers each gene's remaining sites again. The previous names are
    kept in .var["UTR_site"].

    Args:
        input_path: Count matrix (.h5ad).
        out_path: Where to write the renamed count matrix (.h5ad).
        gene_name_column: Column of the count matrix's .var table with the gene
            names. Genes without a name use the Ensembl ID that starts their UTR name.
    """
    from utrcalling.core import anndata_processing as ap
    from utrcalling.core import helpers

    adata = helpers.check_if_path_and_open(input_path, method="h5ad")
    adata = ap.generate_utr_site_names_from_gene_name(
        adata=adata, utr_name_column=utr_name_column, gene_name_column=gene_name_column
    )

    adata.write_h5ad(out_path)
