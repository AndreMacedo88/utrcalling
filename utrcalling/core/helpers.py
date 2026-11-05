from __future__ import annotations

import os
import pathlib
import pickle
import re
from typing import TYPE_CHECKING, Literal, overload

import numpy as np
import pandas as pd

from utrcalling.core import module_log

if TYPE_CHECKING:
    import anndata

# Joblib defaults — used consistently across the codebase. Override per-run via
# the `--cores` CLI flag, which threads down to every Parallel call site as
# the `n_jobs` parameter (resolved by `resolve_n_jobs`).
#
# The default ``-2`` means all cores but one, as in joblib, leaving one core
# for the OS and interactive shell (``resolve_n_jobs`` gives every
# non-positive value that meaning). This is system-portable
# (a fixed cap like 12 oversubscribes 4-core boxes and underuses 27-core ones).
#
# Multi-job orchestrators (e.g. Snakemake) should pass ``--cores {threads}``
# in the rule's shell command so each utrcalling job stays within its
# Snakemake-allocated thread budget; without that, multiple parallel jobs
# will each claim all-but-one cores and oversubscribe the host.
#
# A previous revision capped this at 12 because the per-UTR Parallel loop in
# count_matrix.py was parent-bound (pickling overhead dominated worker work).
# That bottleneck has since been fixed via batching + obs pre-encoding, so
# the cap is no longer needed.
JOBLIB_DEFAULT_N_JOBS = -2
JOBLIB_DEFAULT_BACKEND = "loky"


def resolve_n_jobs(n_jobs: int | None) -> int:
    """Return how many worker processes to use.

    Args:
        n_jobs: Explicit worker count from a CLI flag or API caller. ``None``
            means the project default, ``JOBLIB_DEFAULT_N_JOBS``.

    Returns:
        ``n_jobs`` if it is positive; otherwise all cores but one, and at least
        one. joblib refuses 0, so every Parallel call takes this count.
    """
    n_jobs = n_jobs if n_jobs is not None else JOBLIB_DEFAULT_N_JOBS
    return n_jobs if n_jobs > 0 else max(1, get_number_cores(left_out=1))


def shutdown_joblib_pool() -> None:
    """Force termination of loky's reusable worker pool.

    ``joblib.parallel_config`` is a configuration context manager — it does
    not tear down the pool when the ``with`` block exits. Loky keeps workers
    alive for reuse in subsequent calls, which means worker memory (per-task
    inputs, BLAS arenas, mmaps) is held by live processes until the parent
    Python interpreter exits. For long pipelines that's effectively a leak
    in the eyes of the OS — the user can observe tens of GB held even after
    the parallel function returns.

    Calling this after each parallel block forces the pool to shut down
    immediately, releasing memory and file descriptors. The trade-off is
    that the next parallel call in the same process must spawn fresh
    workers (~1-3 s/worker for full re-imports of anndata/scipy/pandas);
    for typical CLI/Snakemake use this cost is invisible because each
    command runs in its own process anyway.
    """
    if JOBLIB_DEFAULT_BACKEND != "loky":
        return
    from joblib.externals.loky import get_reusable_executor

    try:
        get_reusable_executor().shutdown(wait=True)
    except Exception as exc:
        # Best-effort cleanup: if loky's internals refuse for any reason,
        # don't mask the real outcome of the parallel block.
        module_log.module_logger().debug("Could not shut down the joblib worker pool: %r", exc)


def ensure_mapped_reads_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce the canonical flat schema for mapped reads DataFrames.

    Ensures utr_name and observation_id are Categorical, and utr_size is int32.

    Args:
        df: A DataFrame with columns (utr_name, observation_id, utr_size).

    Returns:
        The DataFrame with canonical dtypes applied (modified in-place and returned).
    """
    if not isinstance(df["utr_name"].dtype, pd.CategoricalDtype):
        df["utr_name"] = pd.Categorical(df["utr_name"])
    if not isinstance(df["observation_id"].dtype, pd.CategoricalDtype):
        df["observation_id"] = pd.Categorical(df["observation_id"])
    if df["utr_size"].dtype != np.int32:
        df["utr_size"] = df["utr_size"].astype(np.int32)
    return df


def get_number_cores(left_out: int = 1) -> int:
    """Return the number of CPU cores to use, reserving some.

    Args:
        left_out: Number of cores to reserve (not use).

    Returns:
        Number of usable cores.
    """
    available_cores = len(os.sched_getaffinity(0))
    cores_to_use = available_cores - left_out
    return cores_to_use


def to_string(x: bytes | str) -> str:
    """Convert bytes to str without decoding.

    Handles both bytes and str inputs — bytes are converted by slicing
    the repr output, while str is passed through unchanged.

    Args:
        x: The bytes or str object representing a sequence.

    Returns:
        The string content without the b'' wrapper.
    """
    if isinstance(x, str):
        return x
    return repr(x)[2:-1]


def log_metrics_anndata(adata: anndata.AnnData) -> None:
    """Log summary metrics (peak count, sample/cell count) for an AnnData object.

    Args:
        adata: An AnnData object with a sparse or dense X matrix.
    """
    logger = module_log.module_logger()
    metrics = f"Called {adata.X.shape[1]} peaks in {adata.X.shape[0]} samples/cells"
    logger.info(metrics)
    logger.info(adata)


def check_if_exists_and_mkdir(folders_path: str) -> None:
    """Create a directory (and parents) if it does not already exist.

    Args:
        folders_path: Directory path to create.
    """
    if not os.path.exists(folders_path):
        os.makedirs(folders_path)


@overload
def check_if_path_and_open(
    to_check: str | pathlib.PurePath | pd.DataFrame,
    method: Literal["parquet", "pickle_pandas", "csv"] = ...,
) -> pd.DataFrame: ...
@overload
def check_if_path_and_open(
    to_check: str | pathlib.PurePath | anndata.AnnData, method: Literal["h5ad"]
) -> anndata.AnnData: ...
@overload
def check_if_path_and_open(to_check: object, method: Literal["pickle"]) -> object: ...
def check_if_path_and_open(to_check: object, method: str = "parquet") -> object:
    """If the input is a file path, load it using the specified method; otherwise pass through.

    Args:
        to_check: A file path (str or Path) to load, or an already-loaded object.
        method: File format to use for loading ('parquet', 'pickle', 'pickle_pandas',
            'h5ad', or 'csv').

    Returns:
        The loaded object if a path was given, or the original object unchanged.
    """
    if isinstance(to_check, (str, pathlib.PurePath)):
        method = method.lower()
        if method == "parquet":
            output = pd.read_parquet(to_check)
        elif method == "pickle":
            with open(to_check, "rb") as input_file:
                output = pickle.load(input_file)
        elif method == "pickle_pandas":
            output = pd.read_pickle(to_check)
        elif method == "h5ad":
            import anndata as ad

            output = ad.read_h5ad(to_check)
        elif method == "csv":
            output = pd.read_csv(to_check, sep=",")
        else:
            raise RuntimeError(f"Method {method} not implemented yet")
        return output
    else:
        return to_check


def load_annotations(annotation_path: str) -> pd.DataFrame:
    """Load a UTR reference written by utrcalling, by its file extension.

    Args:
        annotation_path: Path to a .parquet (or, from older versions, .pkl) file.

    Returns:
        The UTR reference DataFrame, indexed by utr_name.

    Raises:
        RuntimeError: If the file extension is not recognized.
    """
    if annotation_path.endswith(".parquet"):
        return pd.read_parquet(annotation_path)
    if annotation_path.endswith(".pkl"):
        return pd.read_pickle(annotation_path)
    raise RuntimeError(
        f"UTR reference {annotation_path} is not a .parquet or .pkl file: use the "
        "utr_reference_regions.parquet that run -i or prepare-reference-and-map-reads write."
    )


def sample_in_path(sample: str | int, bam_path: str) -> bool:
    """Return True if *sample* appears as a delimited token anywhere in *bam_path*.

    The function searches the entire path string (filename **and** all parent
    directory components) so it works whether the sample identifier is embedded
    in the BAM filename (common in test data / older pipelines) or encoded as the
    parent directory name (e.g. STAR output where every BAM is named
    ``Aligned.sortedByCoord.out.bam``).

    Path separators (``/``), underscores (``_``), and dots (``.``) are treated as
    token boundaries, which prevents a sample name that is a prefix of another
    (e.g. ``S1`` vs ``S10``) from producing a false match.

    *sample* is coerced to ``str``: annotation files whose ``sample_id`` column
    holds bare integers (e.g. ``102``) are read back by pandas as ``int64``, and
    ``re.escape`` rejects non-string input.
    """
    pattern = r"(?:^|(?<=[/_. ]))" + re.escape(str(sample)) + r"(?=$|[/_.])"
    return bool(re.search(pattern, str(bam_path)))
