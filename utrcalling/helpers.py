from collections.abc import Iterable
import os


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
