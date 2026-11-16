import itertools
from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.neighbors import KernelDensity

# Maps one UTR's molecule sizes to their consensus peak sizes.
SizeTransform = Callable[[npt.ArrayLike], npt.NDArray[np.integer[Any]]]

# Methods to aggregate individual molecules' UTR sizes to consensus peaks:


def get_caller(method: str, bandwidth: int = 12) -> SizeTransform:
    """Return a size transform callable for the given peak calling method.

    Args:
        method: Peak calling algorithm name (currently only 'kde').
        bandwidth: Bandwidth parameter for KDE-based peak calling.

    Returns:
        A callable that takes an array of UTR sizes and returns transformed sizes.
    """
    method = method.lower()
    if method == "kde":
        return partial(call_peaks_using_kernel_density_estimator, bandwidth=bandwidth)
    else:
        raise NotImplementedError(
            f"Method {method} for calling consensus peaks is not implemented."
        )


def call_peaks_using_kernel_density_estimator(
    utr_sizes_vector: npt.ArrayLike, bandwidth: float = 12
) -> npt.NDArray[np.int32]:
    """Assign each molecule to a consensus peak of its UTR, with a Gaussian KDE.

    The density of the UTR sizes has a peak (local maximum) per site; the
    lowest point between two neighbouring peaks splits their molecules. Each
    molecule gets the position of its peak. With 3 molecules or fewer, all get
    their median.

    Args:
        utr_sizes_vector: The UTR sizes of the molecules, as a vector or a
            column (n x 1) array.
        bandwidth: Bandwidth of the Gaussian kernel.

    Returns:
        The peak position of each molecule, in input order.
    """
    sizes = np.asarray(utr_sizes_vector).ravel()

    if len(sizes) == 0:  # a UTR without molecules, e.g. in a subset of the cells
        return np.zeros(0, dtype=np.int32)
    if len(sizes) <= 3:
        return np.full(len(sizes), round(np.median(sizes)), dtype=np.int32)

    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(sizes.reshape(-1, 1))

    # Evaluate the density at each size of a window padded beyond the sizes, so
    # that it rises to the first peak and falls after the last one.
    minimum_size, maximum_size = sizes.min(), sizes.max()
    padding = max((maximum_size - minimum_size) * 0.1, 10)
    window = np.arange(round(minimum_size - padding), round(maximum_size + padding) + 1)
    return _assign_to_peaks(sizes, window, kde.score_samples(window.reshape(-1, 1)))


def _assign_to_peaks(
    sizes: npt.NDArray[np.integer[Any]],
    window: npt.NDArray[np.integer[Any]],
    density: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.int32]:
    """Each molecule's peak: the local maximum of the density on its side of the valleys.

    The density may have flat stretches of equal values. A flat stretch higher
    than both its sides is one peak, at its middle (the lower of the two
    middle sizes); one on a slope (a step) is not a peak. Two neighbouring
    peaks split at the lowest point between them (the middle of a flat
    valley): the molecules below it go to the peak before, the others to the
    peak after. Beyond the ends of *window*, the density counts as lower than
    at the ends.

    Args:
        sizes: The molecules' UTR sizes.
        window: Consecutive sizes covering *sizes*, where the density was
            evaluated.
        density: The density at each size of *window*, or any increasing
            function of it, such as its log (minus infinity where it is 0).

    Returns:
        The peak position of each molecule, in the order of *sizes*.
    """
    starts, ends, levels = _flat_stretches(density)
    middles = (starts + ends) // 2
    higher_than_before = levels > np.r_[-np.inf, levels[:-1]]
    higher_than_after = levels > np.r_[levels[1:], -np.inf]
    peaks = np.flatnonzero(higher_than_before & higher_than_after)
    splits = [
        middles[before + 1 + int(np.argmin(levels[before + 1 : after]))]
        for before, after in itertools.pairwise(peaks)
    ]
    peak_sizes = window[middles[peaks]]
    return peak_sizes[np.searchsorted(window[splits], sizes, side="right")].astype(np.int32)


def _flat_stretches(
    density: npt.NDArray[np.floating[Any]],
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp], npt.NDArray[np.floating[Any]]]:
    """The runs of equal neighbouring values: their first and last index, and their value."""
    starts = np.flatnonzero(np.r_[True, density[1:] != density[:-1]])
    ends = np.r_[starts[1:] - 1, len(density) - 1]
    return starts, ends, density[starts]
