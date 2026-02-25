from __future__ import annotations

from collections.abc import Iterable
import os
import pathlib
import pickle

import numpy as np
import pandas as pd
import pybedtools
import scanpy as sc

from utrcalling.core import module_log

# Joblib defaults — use consistently across the codebase
JOBLIB_DEFAULT_N_JOBS = -2  # all CPUs minus one
JOBLIB_DEFAULT_BACKEND = "loky"


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


def flatten(list_: Iterable) -> Iterable:
    """Flatten an irregular list of lists recursively.

    Args:
        list_: An irregular list of lists — each index may contain another list,
            a nested list of lists, an int, str, etc.

    Yields:
        The next element in the flattened sequence.
    """
    for element in list_:
        if isinstance(element, Iterable) and not isinstance(element, (str, bytes)):
            yield from flatten(element)
        else:
            yield element


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


def log_metrics_anndata(adata: object) -> None:
    """Log summary metrics (peak count, sample/cell count) for an AnnData object.

    Args:
        adata: An AnnData object with a sparse or dense X matrix.
    """
    logger = module_log.module_logger()
    metrics = f"Called {adata.X.shape[1]} peaks in {adata.X.shape[0]} samples/cells"
    logger.info(metrics)
    logger.info(adata)


def get_paths_from_folders(folders_path: str, suffix: str = ".bam") -> list[str]:
    """Recursively find all files with a given suffix under a directory.

    Args:
        folders_path: Root directory to search.
        suffix: File suffix to match (e.g., '.bam').

    Returns:
        List of absolute file paths.
    """
    file_list = []
    for path in pathlib.Path(folders_path).rglob(f"*{suffix}"):
        file_list.append(os.path.join(folders_path, path))
    return file_list


def get_parent_folders_names_from_file(file_path: str) -> str:
    """Return the name of the parent directory of a file path.

    Args:
        file_path: A file path string.

    Returns:
        The immediate parent directory name.
    """
    path = os.path.normpath(file_path)
    parent = path.split(os.sep)[-2]
    return parent


def check_if_exists_and_mkdir(folders_path: str) -> None:
    """Create a directory (and parents) if it does not already exist.

    Args:
        folders_path: Directory path to create.
    """
    if not os.path.exists(folders_path):
        os.makedirs(folders_path)


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
            output = sc.read_h5ad(to_check)
        elif method == "csv":
            output = pd.read_csv(to_check, sep=",")
        else:
            raise RuntimeError(f"Method {method} not implemented yet")
        return output
    else:
        return to_check


def check_file_exists_and_return_unique(filepath: str, append: int = 1) -> str:
    """Return a unique file path by appending a numeric suffix if the path exists.

    Recursively increments the suffix until a non-existing path is found.

    Args:
        filepath: The desired file path.
        append: Starting suffix number.

    Returns:
        A file path guaranteed not to exist yet.
    """
    if not os.path.exists(filepath):
        return filepath
    else:
        path_, ext = os.path.splitext(filepath)
        filepath = f"{path_}_{append}{ext}"
        counter = append + 1
        return check_file_exists_and_return_unique(filepath, append=counter)


def load_annotations(annotation_path: str) -> pd.DataFrame | pybedtools.BedTool:
    """Load an annotation file based on its file extension.

    Args:
        annotation_path: Path to a .parquet, .bed, or .pkl annotation file.

    Returns:
        A DataFrame (parquet/pickle) or BedTool (bed) object.

    Raises:
        RuntimeError: If the file extension is not recognized.
    """
    if annotation_path.endswith(".parquet"):
        output = pd.read_parquet(annotation_path)
    elif annotation_path.endswith(".bed"):
        output = pybedtools.BedTool(annotation_path)
    elif annotation_path.endswith(".pkl"):
        output = pd.read_pickle(annotation_path)
    else:
        raise RuntimeError(
            f"Annotation file {annotation_path} is of a not recognized type."
        )
    return output


def check_if_dir_and_list_files(
    path: str | list[str], suffix: str = ".parquet"
) -> str | list[str]:
    """If path is a directory, list files with the given suffix; otherwise pass through.

    Handles both single paths and lists of paths recursively.

    Args:
        path: A file path, directory path, or list of paths.
        suffix: File suffix to match when listing directories.

    Returns:
        A single file path, or a list of file paths.
    """
    if isinstance(path, (list, tuple)):
        paths = []
        for p in path:
            paths.append(check_if_dir_and_list_files(p, suffix=suffix))
        return list(flatten(paths))
    if os.path.isdir(path):
        file_list = get_paths_from_folders(path, suffix)
        return file_list
    else:
        return path



def split_list(a: list, n: int) -> Iterable[list]:
    """Split a list into n roughly equal-sized chunks.

    Args:
        a: The list to split.
        n: Number of chunks.

    Returns:
        A generator yielding n sub-lists.
    """
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def check_file_suffix_and_append_before(path: str, to_append: str, suffix: str) -> str:
    """Insert a string before the file suffix in a path.

    Args:
        path: The original file path.
        to_append: String to insert before the suffix.
        suffix: The file suffix (e.g., '.h5ad').

    Returns:
        The modified path with to_append inserted before the suffix.
    """
    if not suffix.startswith("."):
        suffix = f".{suffix}"
    if suffix in path:
        path = path.replace(suffix, "")

    return f"{path}{to_append}{suffix}"


