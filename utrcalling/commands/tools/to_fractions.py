"""`utrcalling tools to-fractions`: each site's share of its UTR's counts."""

from __future__ import annotations

from utrcalling.commands.options import UtrNameColumn


def to_fractions(
    input_path: str, out_path: str, *, utr_name_column: UtrNameColumn = "utr_name"
) -> None:
    """Convert counts into each site's share of its UTR's counts, per cell.

    In each cell (or sample), divides the count of every site by the total count
    of its UTR, so a UTR's sites add up to 1 (0 where the cell has no counts in
    that UTR).

    Args:
        input_path: Count matrix (.h5ad).
        out_path: Where to write the fractions (.h5ad).
    """
    import scipy.sparse

    from utrcalling.core import anndata_processing as ap
    from utrcalling.core import helpers, provenance

    counts = helpers.check_if_path_and_open(input_path, method="h5ad")
    adata = ap.counts_to_percentages(adata=counts, utr_name_column=utr_name_column)

    # Sparsify if it was sparse before
    if scipy.sparse.issparse(counts.X):
        adata.X = scipy.sparse.csr_matrix(adata.X)
    provenance.record(adata, source=counts, matrix="fractions")
    adata.write_h5ad(out_path)
