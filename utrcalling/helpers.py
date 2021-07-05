from collections.abc import Iterable
import os

from utrcalling.module_log import module_logger


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
    logger = module_logger()
    metrics = f'Called {adata.X.shape[1]} peaks '
    f'in {adata.X.shape[1]} cells'
    logger.info(metrics)
    logger.info(adata)
