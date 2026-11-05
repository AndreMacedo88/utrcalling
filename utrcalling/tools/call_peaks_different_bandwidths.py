"""`utrcalling bandwidth call-peaks`: peaks of a raw-sizes count matrix, per bandwidth."""

from __future__ import annotations

from utrcalling.tools.options import Bandwidths, Cores


def call_peaks_different_bandwidths(
    input_path: str,
    out_path: str,
    *,
    bandwidths: Bandwidths = (12,),
    cores: Cores = None,
) -> None:
    """Collapse each molecule's UTR size into peaks, with one or more KDE bandwidths.

    Takes the raw-sizes count matrix of `run -r` and collapses the sizes of each
    UTR into sites (peaks), as `run` does, once per bandwidth. Writes one file per
    bandwidth: `<out_path>_bw<bandwidth>.h5ad`.

    For example:

        utrcalling bandwidth call-peaks raw_sizes.h5ad peaks -b 7 12 15

    Args:
        input_path: Count matrix of raw UTR sizes (.h5ad), from run -r.
        out_path: Start of the output file names (a .h5ad ending is dropped).
    """
    from utrcalling.core import helpers
    from utrcalling.core import peak_calling_anndata as pc

    adata = helpers.check_if_path_and_open(input_path, method="h5ad")
    adata = adata[:, adata.var.sort_values(by=["utr_name", "utr_size"], ascending=True).index]
    out_path = out_path.split(".h5ad")[0]
    for bandwidth in bandwidths:
        adata_peaks = pc.peak_call_anndata_parallel(
            adata, method="kde", bandwidth=bandwidth, n_jobs=cores
        )
        adata_peaks.write_h5ad(f"{out_path}_bw{bandwidth}.h5ad")
