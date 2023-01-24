from collections.abc import Iterable
import os
import pathlib
import pickle
from collections import defaultdict

import pybedtools
import pandas as pd
import scanpy as sc

from utrcalling.core import module_log


def flatten(list_):
    """
    Flattens an irregular list of lists. Works for list-like objects.

    Args:
        list_ (list): An irregular list of lists, that is, a list that for each index
        might have another list, list of lists, an int, str, etc.

    Yields:
        Any: Yields the next element in the list, but flattened 
        (with one nest level removed).
    """
    for element in list_:
        if isinstance(element, Iterable) and not isinstance(element, (str, bytes)):
            yield from flatten(element)
        else:
            yield element


def get_number_cores(left_out=1):
    available_cores = len(os.sched_getaffinity(0))
    cores_to_use = available_cores - left_out
    return cores_to_use


def to_string(x):
    """
    Converts type=bytes to type=str without decoding. 
    Because HTSeq gives us bytes instead of strings.

    Args:
        x (bytes): The object in bytes, that in fact represent a string.

    Returns:
        str 
    """
    return repr(x)[2:-1]


def log_metrics_anndata(adata):
    logger = module_log.module_logger()
    metrics = f'Called {adata.X.shape[1]} peaks '
    f'in {adata.X.shape[1]} cells'
    logger.info(metrics)
    logger.info(adata)


def get_paths_from_folders(folders_path, suffix='.bam'):
    file_list = []
    for path in pathlib.Path(folders_path).rglob(f'*{suffix}'):
        file_list.append(os.path.join(folders_path, path))
    return file_list


def get_parent_folders_names_from_file(file_path):
    path = os.path.normpath(file_path)
    parent = path.split(os.sep)[-2]
    return parent


def check_if_exists_and_mkdir(folders_path):
    if not os.path.exists(folders_path):
        os.makedirs(folders_path)


def check_if_path_and_open(to_check, method='pickle'):
    if isinstance(to_check, (str, pathlib.PurePath)):
        method = method.lower()
        if method == 'pickle':
            with open(to_check, "rb") as input_file:
                output = pickle.load(input_file)
        elif method == "pickle_pandas":
            output = pd.read_pickle(to_check)
        elif method == "h5ad":
            output = sc.read_h5ad(to_check)
        elif method == "csv":
            output = pd.read_csv(to_check, sep=";")  # , engine="python"
        else:
            raise RuntimeError(f'Method {method} not implemented yet')
        return output
    else:
        return to_check


def check_file_exists_and_return_unique(filepath, append=1):
    if not os.path.exists(filepath):
        return filepath
    else:
        path_, ext = os.path.splitext(filepath)
        filepath = f"{path_}_{append}{ext}"
        counter = append + 1
        return check_file_exists_and_return_unique(filepath, append=counter)


def load_annotations(annotation_path):
    if annotation_path.endswith('.pkl'):
        output = pd.read_pickle(annotation_path)
    elif annotation_path.endswith('.bed'):
        output = pybedtools.BedTool(annotation_path)
    else:
        raise RuntimeError(
            f'Annotation file {annotation_path} is of a not recognized type.')
    return output


def check_if_dir_and_list_files(path, suffix='.pkl'):
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


def hamming_distance(a, b):
    if len(a) != len(b):
        raise RuntimeError('Hamming distance is valid for only equal length string')
    distance = sum(x != y for x, y in zip(a, b))

    return distance


def count_duplicates(list_of_strings, hamming=0):
    dups = defaultdict(list)
    for i, item in enumerate(list_of_strings):
        dups[item].append(i)

    if hamming == 0:
        return dups

    else:
        hammed = defaultdict(list)
        dupped_umis_sort = [k for k, _ in sorted(dups.items(), key=len)]

        while len(dupped_umis_sort) > 0:
            pivot = dupped_umis_sort[0]
            rest = dupped_umis_sort[1:]
            hammed[pivot] = dups[pivot]
            to_delete = []
            for idx, umi in enumerate(rest):
                dist = hamming_distance(pivot, umi)
                if dist <= 1:
                    hammed[pivot].extend(dups[umi])
                    to_delete.append(idx + 1)
            to_delete.append(0)
            to_delete = sorted(to_delete, reverse=True)
            for idx2 in to_delete:
                del dupped_umis_sort[idx2]
        return hammed


def split_list(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def check_file_suffix_and_append_before(path, to_append, suffix):
    if not suffix.startswith("."):
        suffix = f".{suffix}"
    if suffix in path:
        path = path.replace(suffix, "")

    return f"{path}{to_append}{suffix}"
