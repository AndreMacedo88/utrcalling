from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.signal import argrelextrema
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

    The density of the UTR sizes has a peak (local maximum) per polyadenylation
    site; the local minima between peaks split the molecules among them. Each
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

    # Evaluate the density on a window padded beyond the sizes, so that it rises
    # to the first peak and falls after the last one.
    minimum_size, maximum_size = sizes.min(), sizes.max()
    padding = max((maximum_size - minimum_size) * 0.1, 10)
    window = np.arange(round(minimum_size - padding), round(maximum_size + padding) + 1)
    evaluation = kde.score_samples(window.reshape(-1, 1))
    # [1:]: the window's first point always counts as a minimum.
    minima = argrelextrema(evaluation, np.less_equal, order=1)[0][1:]
    maxima = argrelextrema(evaluation, np.greater_equal, order=1)[0]
    # A flat extremum spans several points: keep its first one.
    minima = [minimum for minimum in minima if minimum - 1 not in minima]
    maxima = [maximum for maximum in maxima if maximum - 1 not in maxima]

    # Molecules below the first minimum belong to the first peak, those between
    # the first and second minima to the second peak, and so on. The window's
    # last point is a minimum above every size, so every molecule gets a peak.
    # (A mask, not peaks == 0, tracks this: a peak can lie at UTR size 0.)
    peaks = np.zeros(len(sizes), dtype=np.int32)
    assigned = np.zeros(len(sizes), dtype=bool)
    for peak, boundary in zip(window[maxima], window[minima], strict=False):
        in_peak = (sizes < boundary) & ~assigned
        peaks[in_peak] = round(peak)
        assigned |= in_peak
    return peaks
