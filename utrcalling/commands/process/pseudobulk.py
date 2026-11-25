"""`utrcalling process pseudobulk`: sum the counts of cells into units, per sample."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated

from cyclopts import Parameter, Token


def bins_per_column(type_: object, tokens: Sequence[Token]) -> tuple[str, ...]:
    """Read COLUMN:N pairs, as in `--bins pseudotime:3`."""
    for token in tokens:
        column, colon, number = token.value.rpartition(":")
        if not colon or not column or not number.isdigit():
            raise ValueError(f"expected COLUMN:N, as in pseudotime:3, got {token.value!r}")
    return tuple(token.value for token in tokens)


def pseudobulk(
    input_path: str,
    out_path: str,
    *,
    by: Annotated[list[str], Parameter(consume_multiple=1, metavar="COLUMN")],
    bins: Annotated[
        tuple[str, ...] | None,
        Parameter(consume_multiple=1, converter=bins_per_column, metavar="COLUMN:N"),
    ] = None,
    min_cells: Annotated[int, Parameter(metavar="N")] = 10,
) -> None:
    """Sum the counts of cells into units: per sample, or per sample and bin.

    Statistics need replicates: cells of one sample are not independent
    replicates of anything that differs between samples (a condition, a
    genotype), so test those on one unit per sample. A unit is the cells that
    share their values of the --by columns of the cell annotations (.obs),
    such as the sample, and with --bins their bin of a numeric column, such
    as pseudotime cut into equal-frequency bins over all cells. Always
    include the sample (or the column that names the biological replicate)
    in --by: units must not mix samples.

    Each unit keeps the annotation columns whose value is the same for all its
    cells (the sample's condition or batch), and the mean of the other
    numeric columns (the mean pseudotime of a bin); others are left out. It
    also gets n_cells, and with --bins the bin number in `<column>_bin`.

    For example:

        utrcalling process pseudobulk sites.h5ad samples.h5ad --by sample_id
        utrcalling process pseudobulk sites.h5ad units.h5ad --by sample_id --bins pseudotime:3
        utrcalling process pseudobulk sites.h5ad units.h5ad --by sample_id cell_type --min-cells 20

    Args:
        input_path: Count matrix (.h5ad), cells by sites or raw sizes.
        out_path: Where to write the units' counts (.h5ad).
        by: Columns of the cell annotations whose values make the units.
        bins: Numeric columns to cut into N bins of equal numbers of cells,
            each unit then being one bin, as in pseudotime:3 (terciles).
        min_cells: Units with fewer cells are left out.
    """
    from utrcalling.core import anndata_processing as ap
    from utrcalling.core import helpers, provenance, qc, storage
    from utrcalling.core.module_log import module_logger

    bins_by_column = {
        column: int(number) for column, _, number in (pair.rpartition(":") for pair in bins or ())
    }
    counts = helpers.check_if_path_and_open(input_path, method="h5ad")
    qc.require_counts(counts)
    units = ap.pseudobulk(counts, by=by, bins=bins_by_column, min_cells=min_cells)
    module_logger().info(f"Summed {int(units.obs['n_cells'].sum())} cells into {units.n_obs} units")
    provenance.record(
        units,
        source=counts,
        pseudobulk_by=list(by),
        pseudobulk_bins=list(bins) if bins else None,
        pseudobulk_min_cells=min_cells,
    )
    storage.write_anndata(units, out_path)
