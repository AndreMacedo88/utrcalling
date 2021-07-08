from collections.abc import Iterable
import os
import pathlib
import pickle

import pybedtools
import pandas as pd

from utrcalling import module_log


def flatten(l):
    """
    Flattens an irregular list of lists. Works for list-like objects.

    Args:
        l (list): An irregular list of lists, that is, a list that for each index
        might have another list, list of lists, an int, str, etc.

    Yields:
        Any: Yields the next element in the list, but flattened 
        (with one nest level removed).
    """
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


def get_number_cores():
    available_cores = len(os.sched_getaffinity(0))
    cores_to_use = round(available_cores - available_cores * 0.1)
    if cores_to_use == available_cores:
        cores_to_use = available_cores - 1
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
        if method.lower() == 'pickle':
            with open(to_check, "rb") as input_file:
                output = pickle.load(input_file)
        else:
            raise RuntimeError(f'Method {method} not implemented yet')
        return output
    else:
        return to_check


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
    if os.path.isdir(path):
        file_list = get_paths_from_folders(path, suffix)
        return file_list
    else:
        return path
