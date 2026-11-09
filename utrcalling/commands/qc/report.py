"""`utrcalling qc report`: a summary of count matrices, from any step."""

from __future__ import annotations

from typing import Annotated

from cyclopts import Parameter

# MultiQC shows a file whose name ends with _mqc.tsv as a table, with this header.
MULTIQC_HEADER = """\
# id: 'utrcalling'
# section_name: 'utrcalling'
# description: 'Count matrices of UTR sites: cells, sites and molecules (utrcalling qc report).'
# plot_type: 'table'
"""


def report(
    input_paths: Annotated[list[str], Parameter(consume_multiple=1)],
    *,
    out_path: Annotated[str, Parameter(name=("--out-path", "-o"))],
    min_molecules: Annotated[int, Parameter(name=("--min-molecules", "-m"))] = 5,
    multiqc: bool = False,
) -> None:
    """Summarise count matrices of any step in a table, a row per file.

    For each matrix: what it holds and the steps that made it; its cells,
    sites, UTRs and genes. For counts (raw sizes or peaks): the molecules, the
    median molecules, sites and measured UTRs per cell, the UTRs with two or
    more sites and their share of the molecules, and the correlation of the
    cells' relative UTR size with their molecules (see qc cells). For fractions
    and mean sizes: the median values per cell. With the mapping summary of
    run or count: its alignments, reads and molecules.

    For example:

        utrcalling qc report raw.h5ad peaks.h5ad filtered.h5ad -o report_mqc.tsv --multiqc

    Args:
        input_paths: Matrices (.h5ad) that utrcalling wrote.
        out_path: Where to write the table (.tsv).
        min_molecules: Molecules a cell needs in a UTR to measure its size there.
        multiqc: Start the table with the header MultiQC reads: name it
            `<name>_mqc.tsv` for MultiQC to find it.
    """
    from pathlib import Path

    from utrcalling.core import helpers, qc

    rows = {
        Path(path).name: qc.matrix_report(
            helpers.check_if_path_and_open(path, method="h5ad"), min_molecules=min_molecules
        )
        for path in input_paths
    }
    table = qc.report_table(rows)
    with open(out_path, "w") as out:
        if multiqc:
            out.write(MULTIQC_HEADER)
        table.to_csv(out, sep="\t")
