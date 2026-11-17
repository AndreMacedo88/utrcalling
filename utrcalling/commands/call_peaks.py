"""`utrcalling call-peaks`: the peaks of a raw-sizes count matrix, per KDE bandwidth and kernel."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from utrcalling.commands.options import (
    Bandwidths,
    CandidateBandwidths,
    Cores,
    Genome,
    KdeKernels,
    KnownSites,
    PeakCaller,
    Scores,
)

if TYPE_CHECKING:
    from utrcalling.core.peak_calling.settings import Bandwidth


def call_peaks(
    input_path: str,
    out_path: str,
    *,
    bandwidths: Bandwidths = (12,),
    kde_kernel: KdeKernels = ("gaussian",),
    peak_caller: PeakCaller = "kde",
    candidates: CandidateBandwidths = None,
    scores: Scores = None,
    genome: Genome = None,
    known_sites: KnownSites = None,
    utr_end: Literal["three_prime", "five_prime"] | None = None,
    cores: Cores = None,
) -> None:
    """Call the peaks of a raw-sizes count matrix, with one or more KDE bandwidths and kernels.

    Takes the raw-sizes count matrix of `run -r` and collapses the sizes of each
    UTR into sites (peaks), as `run` does, once per bandwidth and kernel. Writes
    one file per setting, `<out_path>_<kernel>_bw<bandwidth>.h5ad`, without the
    kernel for the Gaussian: such as peaks_bw12.h5ad, or
    peaks_linear_bwscott0.5.h5ad for the linear kernel and scott:0.5. With a
    rule, each site's .var has the bandwidth of its UTR.

    With -b optimize, scores the --candidates with each kernel (as qc
    score-bandwidths does), writes their scores to
    `<out_path>_bandwidth_scores.tsv`, and calls the peaks with the best.

    For example:

        utrcalling call-peaks raw_sizes.h5ad peaks -b 7 12 15
        utrcalling call-peaks raw_sizes.h5ad peaks -b 2 scott:0.5 --kde-kernel gaussian linear
        utrcalling call-peaks raw_sizes.h5ad peaks -b optimize --genome GRCm39.fa

    Args:
        input_path: Count matrix of raw UTR sizes (.h5ad), from run -r.
        out_path: Start of the output file names (a .h5ad ending is dropped).
        kde_kernel: Kernel of the KDE; several give a file each, or with -b
            optimize, are chosen among.
        utr_end: Which end of the UTR the sizes count from, for -b optimize:
            three_prime for 3' protocols, five_prime for 5' ones. Only for
            matrices of utrcalling before 5.0.0, which are taken for 3' data:
            later ones record it.
    """
    from utrcalling.commands import _optimize
    from utrcalling.core import helpers, provenance
    from utrcalling.core import peak_calling_anndata as pc
    from utrcalling.core.peak_calling.settings import OPTIMIZE, setting_label

    adata = helpers.check_if_path_and_open(input_path, method="h5ad")
    adata = adata[:, adata.var.sort_values(by=["utr_name", "utr_size"], ascending=True).index]
    out_path = out_path.split(".h5ad")[0]
    choice: dict[str, Any] = {}
    # Bandwidth 0 keeps the raw sizes, whatever the kernel: one file.
    settings: list[tuple[str, Bandwidth]] = list(
        dict.fromkeys(
            (kernel if bandwidth else "gaussian", bandwidth)
            for kernel in kde_kernel
            for bandwidth in bandwidths
        )
    )
    if bandwidths == (OPTIMIZE,):
        record = provenance.recorded(adata)
        utr_end = utr_end or record.get("utr_end", "three_prime")
        _optimize.check(
            utr_end=utr_end,
            scores=scores,
            genome=genome,
            known_sites=known_sites,
            summarization_level=record.get("summarization_level"),
        )
        kernel, bandwidth, choice["bandwidth_scores"] = _optimize.choose(
            adata,
            candidates=candidates,
            kernels=kde_kernel,
            scores=scores,
            genome=genome,
            known_sites=known_sites,
            utr_end=utr_end,
            method=peak_caller,
            cores=cores,
            scores_path=f"{out_path}_bandwidth_scores.tsv",
        )
        settings = [(kernel, bandwidth)]
    for kernel, bandwidth in settings:
        adata_peaks = pc.peak_call_anndata_parallel(
            adata, method=peak_caller, bandwidth=bandwidth, n_jobs=cores, kernel=kernel
        )
        if adata_peaks is adata:  # bandwidth 0 leaves the sizes as they are
            adata_peaks = adata.copy()
            peak_calling = {}
        else:
            peak_calling = {"matrix": "peaks", "peak_caller": peak_caller, "kernel": kernel}
        provenance.record(adata_peaks, source=adata, bandwidth=bandwidth, **peak_calling, **choice)
        label = setting_label(bandwidth, kernel, peak_caller) if bandwidth else "bw0"
        adata_peaks.write_h5ad(f"{out_path}_{label}.h5ad")
