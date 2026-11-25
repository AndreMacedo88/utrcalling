"""`utrcalling tools length-index`: each UTR's length index and entropy of site use."""

from __future__ import annotations

from typing import Annotated

from cyclopts import Parameter

from utrcalling.commands.options import UtrNameColumn


def length_index(
    input_path: str,
    out_path: str,
    *,
    utr_name_column: UtrNameColumn = "utr_name",
    smooth_graph: Annotated[str | None, Parameter(metavar="KEY")] = None,
    smooth_rounds: Annotated[int, Parameter(metavar="N")] = 2,
) -> None:
    """Compute each cell's (or sample's) length index and entropy of site use, per UTR.

    The length index is the mean rank of a UTR's molecules' sites by UTR size,
    scaled from 0 (all at the shortest site) to 1 (all at the longest): a
    larger index means longer UTRs, for 3' data (distal polyadenylation sites)
    and 5' data (upstream start sites) alike. It is scUTRquant's Weighted UTR
    Index. The entropy of site use goes from 0 (one site takes all molecules)
    to 1 (all sites take the same share). Only UTRs with two or more sites
    have them; cells without molecules in a UTR get none (NaN).

    Writes a matrix of cells by UTRs with the length index in .X and the
    entropy in .layers['entropy']. Its .obs adds each cell's mean over its
    UTRs, length_index and entropy (a global length index), and the number
    of UTRs it has molecules in, utrs_measured.

    Single cells have few molecules per UTR, so their indices jump between 0
    and 1. --smooth-graph averages each cell's counts over its neighbours
    first, as in a graph that scanpy's neighbors computed on gene expression
    (.obsp['connectivities']). Smoothed cells share molecules: describe them,
    but don't give p-values to correlations over them (stats site-usage
    tests on the counts).

    For example:

        utrcalling tools length-index sites.h5ad length_index.h5ad
        utrcalling tools length-index sites.h5ad smoothed.h5ad --smooth-graph connectivities

    Args:
        input_path: Count matrix of sites (.h5ad), cells or samples by sites.
        out_path: Where to write the length indices (.h5ad).
        smooth_graph: Key of a graph of neighbouring cells in the matrix's
            .obsp, to average the counts over first.
        smooth_rounds: How many times to average over --smooth-graph.
    """
    from utrcalling.core import anndata_processing as ap
    from utrcalling.core import helpers, provenance, qc, storage

    counts = helpers.check_if_path_and_open(input_path, method="h5ad")
    qc.require_counts(counts)
    adata = ap.length_index(
        counts,
        utr_name_column=utr_name_column,
        smooth_graph=smooth_graph,
        smooth_rounds=smooth_rounds,
    )
    smoothing = None if smooth_graph is None else f"{smooth_graph} x {smooth_rounds}"
    provenance.record(adata, source=counts, matrix="length_index", smoothing=smoothing)
    storage.write_anndata(adata, out_path)
