"""`utrcalling process filter-utrs`: remove the UTRs with few molecules."""

from __future__ import annotations

from typing import Annotated

from cyclopts import Parameter


def filter_utrs(
    input_path: str,
    out_path: str,
    *,
    minimum_molecules: Annotated[int, Parameter(name=("--minimum-molecules", "-m"))],
) -> None:
    """Remove the UTRs with few molecules, whose sites are unsure.

    Keeps a UTR if its molecules, over all its sites and cells, reach
    minimum_molecules. Peak calling keeps every molecule of a UTR, so filtering
    the raw sizes before call-peaks keeps the same UTRs as filtering the sites
    after it. To count only some cells or samples, subset the matrix first.

    For example, as Eugenia Guerrero Aruffo filtered 5' Cap-seq data:

        utrcalling process filter-utrs sites.h5ad deep_utrs.h5ad -m 500

    Args:
        input_path: Count matrix (.h5ad): raw sizes or sites.
        out_path: Where to write the sites of the UTRs kept (.h5ad).
        minimum_molecules: The molecules a UTR needs.
    """
    from utrcalling.core import anndata_processing as ap
    from utrcalling.core import helpers, provenance, qc
    from utrcalling.core.module_log import module_logger

    adata = helpers.check_if_path_and_open(input_path, method="h5ad")
    qc.require_counts(adata)
    adata_new = ap.select_utrs_with_molecules(adata, minimum_molecules)
    utrs = adata.var["utr_name"].nunique()
    module_logger().info(
        f"Kept {adata_new.var['utr_name'].nunique()} of {utrs} UTRs, with "
        f"{adata_new.n_vars} of {adata.n_vars} sites"
    )
    provenance.record(adata_new, source=adata)
    adata_new.write_h5ad(out_path)
