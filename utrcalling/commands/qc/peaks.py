"""`utrcalling qc peaks`: the peaks a KDE bandwidth calls, and how wide each one is."""

from __future__ import annotations

from typing import Annotated

from cyclopts import Parameter

from utrcalling.commands.options import Cores


def peaks(
    input_path: str,
    out_path: str,
    *,
    bandwidth: Annotated[int, Parameter(name=("--bandwidth", "-b"))] = 12,
    cores: Cores = None,
) -> None:
    """Describe the peaks that a KDE bandwidth calls from raw UTR sizes.

    Calls the peaks of each UTR as `call-peaks` does, and writes a tab-separated
    table with a row per peak (site): its molecules and share of its UTR's, the
    number of peaks of its UTR, and the mean, standard deviation, minimum and
    maximum of the raw sizes it collects. A peak much wider than the bandwidth
    may join neighbouring sites; many narrow peaks with few molecules each
    suggest a bandwidth too small for the data.

    For example:

        utrcalling qc peaks raw_sizes.h5ad peaks_bw12.tsv -b 12

    Args:
        input_path: Count matrix of raw UTR sizes (.h5ad), from run -r.
        out_path: Where to write the table (.tsv).
        bandwidth: KDE bandwidth of the peak calling.
    """
    from utrcalling.core import helpers, provenance, storage
    from utrcalling.core import peak_calling_anndata as pc

    adata = helpers.check_if_path_and_open(input_path, method="h5ad")
    matrix = provenance.recorded(adata).get("matrix", "raw_sizes")
    if matrix != "raw_sizes":
        raise ValueError(
            f"This matrix holds {matrix.replace('_', ' ')}: give the raw sizes of run -r."
        )
    table = pc.peak_table(adata, bandwidth=bandwidth, n_jobs=cores)
    storage.write_table(table, out_path)
