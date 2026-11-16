"""`utrcalling call-peaks`: the peaks of a raw-sizes count matrix, per KDE bandwidth."""

from __future__ import annotations

from utrcalling.commands.options import Bandwidths, Cores, KdeKernel


def call_peaks(
    input_path: str,
    out_path: str,
    *,
    bandwidths: Bandwidths = (12,),
    kde_kernel: KdeKernel = "gaussian",
    cores: Cores = None,
) -> None:
    """Call the peaks of a raw-sizes count matrix, with one or more KDE bandwidths.

    Takes the raw-sizes count matrix of `run -r` and collapses the sizes of each
    UTR into sites (peaks), as `run` does, once per bandwidth. Writes one file per
    bandwidth: `<out_path>_bw<bandwidth>.h5ad`, such as peaks_bw12.h5ad, or
    peaks_bwscott0.5.h5ad for scott:0.5. With a rule, each site's .var has the
    bandwidth of its UTR.

    For example:

        utrcalling call-peaks raw_sizes.h5ad peaks -b 7 12 15

    Args:
        input_path: Count matrix of raw UTR sizes (.h5ad), from run -r.
        out_path: Start of the output file names (a .h5ad ending is dropped).
    """
    from utrcalling.core import helpers, provenance
    from utrcalling.core import peak_calling_anndata as pc
    from utrcalling.core.protocols.kde_options import bandwidth_label

    adata = helpers.check_if_path_and_open(input_path, method="h5ad")
    adata = adata[:, adata.var.sort_values(by=["utr_name", "utr_size"], ascending=True).index]
    out_path = out_path.split(".h5ad")[0]
    for bandwidth in bandwidths:
        adata_peaks = pc.peak_call_anndata_parallel(
            adata, method="kde", bandwidth=bandwidth, n_jobs=cores, kernel=kde_kernel
        )
        if adata_peaks is adata:  # bandwidth 0 leaves the sizes as they are
            adata_peaks = adata.copy()
        kde = {"matrix": "peaks", "kernel": kde_kernel} if bandwidth else {}
        provenance.record(adata_peaks, source=adata, bandwidth=bandwidth, **kde)
        adata_peaks.write_h5ad(f"{out_path}_bw{bandwidth_label(bandwidth)}.h5ad")
