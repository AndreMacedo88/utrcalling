"""`utrcalling qc peaks`: the peaks a KDE bandwidth calls, how wide and how stable each one is."""

from __future__ import annotations

from typing import Annotated

from cyclopts import Parameter

from utrcalling.commands.options import Cores, KdeKernel, PeakCaller
from utrcalling.commands.options import bandwidth as read_bandwidth
from utrcalling.core.peak_calling.settings import Bandwidth


def peaks(
    input_path: str,
    out_path: str,
    *,
    bandwidth: Annotated[
        Bandwidth,
        Parameter(name=("--bandwidth", "-b"), converter=read_bandwidth, metavar="NT|RULE"),
    ] = 12,
    kde_kernel: KdeKernel = "gaussian",
    peak_caller: PeakCaller = "kde",
    bootstrap: Annotated[int, Parameter(name="--bootstrap", metavar="N")] = 0,
    seed: int = 0,
    cores: Cores = None,
) -> None:
    """Describe the peaks that a KDE bandwidth calls from raw UTR sizes.

    Calls the peaks of each UTR as `call-peaks` does, and writes a tab-separated
    table with a row per peak (site): its molecules and share of its UTR's, the
    number of peaks of its UTR, and the mean, standard deviation, minimum and
    maximum of the raw sizes it collects. A peak much wider than the bandwidth
    may join neighbouring sites; many narrow peaks with few molecules each
    suggest a bandwidth too small for the data.

    With --bootstrap N, also how stable each peak is: draws as many molecules
    as each UTR has from its molecules, with replacement, N times (100 is
    plenty), calls their peaks, and gives the share of draws in which a peak
    lies within the UTR's bandwidth of it. Peaks with few molecules, or on the
    flank of a larger one, come back less often: filter them out, or check
    them. Wider bandwidths make every peak more stable (one peak per UTR
    always comes back), so compare the peaks of one setting by it, not
    settings: qc score-bandwidths compares settings.

    For example:

        utrcalling qc peaks raw_sizes.h5ad peaks_bw12.tsv -b 12
        utrcalling qc peaks capseq_raw.h5ad peaks.tsv -b scott:0.5 --kde-kernel linear --bootstrap 100

    Args:
        input_path: Count matrix of raw UTR sizes (.h5ad), from run -r.
        out_path: Where to write the table (.tsv).
        bandwidth: KDE bandwidth of the peak calling: a number of nt, or a rule
            (see run --help), whose value for each UTR the table gives.
        bootstrap: How many draws of each UTR's molecules give the peaks'
            stability (column stability); 0 for none.
        seed: Seed of the random draws: the same seed gives the same table.
    """
    from utrcalling.core import helpers, provenance, storage
    from utrcalling.core import peak_calling_anndata as pc

    adata = helpers.check_if_path_and_open(input_path, method="h5ad")
    matrix = provenance.recorded(adata).get("matrix", "raw_sizes")
    if matrix != "raw_sizes":
        raise ValueError(
            f"This matrix holds {matrix.replace('_', ' ')}: give the raw sizes of run -r."
        )
    table = pc.peak_table(
        adata,
        bandwidth=bandwidth,
        n_jobs=cores,
        kernel=kde_kernel,
        method=peak_caller,
        bootstrap=bootstrap,
        seed=seed,
    )
    storage.write_table(table, out_path)
