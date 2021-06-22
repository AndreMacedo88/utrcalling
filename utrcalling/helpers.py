from collections.abc import Iterable

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
