"""KDE peak calling: a site at each peak of the density of a UTR's molecule sizes.

A kernel density estimate (KDE) of a UTR's sizes has a peak per site. The
kernel's shape and its bandwidth set how much the sizes are smoothed: too
little splits a site into several, too much joins neighbouring sites and moves
the peak off the sizes where the molecules are. This is utrcalling's peak
caller ``kde`` (see callers).
"""

from __future__ import annotations

import itertools
import math
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from sklearn.neighbors import KernelDensity

from utrcalling.core.peak_calling.bandwidth_rules import FEWEST_FOR_A_DENSITY, utr_bandwidth
from utrcalling.core.peak_calling.settings import Bandwidth, check_kernel, parse_bandwidth

if TYPE_CHECKING:
    from utrcalling.core.peak_calling.callers import SizeTransform

# The exponential kernel has no end: beyond 40 bandwidths its weight is below
# e^-40 (4e-18) of a molecule's own, too little to change a double.
EXPONENTIAL_REACH = 40


def kde_caller(bandwidth: Bandwidth = 12, kernel: str = "gaussian") -> SizeTransform:
    """The KDE peak caller with a bandwidth and a kernel (a callers.PeakCallerFactory).

    Raises:
        ValueError: If the bandwidth or the kernel is not one.
    """
    return partial(
        call_peaks_using_kernel_density_estimator,
        bandwidth=parse_bandwidth(bandwidth),
        kernel=check_kernel(kernel),
    )


def call_peaks_using_kernel_density_estimator(
    utr_sizes_vector: npt.ArrayLike, bandwidth: Bandwidth = 12, kernel: str = "gaussian"
) -> npt.NDArray[np.int32]:
    """Assign each molecule to a consensus peak of its UTR, with a KDE.

    The density of the UTR sizes has a peak (local maximum) per site; the
    lowest point between two neighbouring peaks splits their molecules. Each
    molecule gets the position of its peak. With 3 molecules or fewer, all get
    their median.

    Args:
        utr_sizes_vector: The UTR sizes of the molecules, as a vector or a
            column (n x 1) array.
        bandwidth: Bandwidth of the kernel: a number of nt, or a rule (see
            bandwidth_rules).
        kernel: The kernel, one of settings.KERNELS.

    Returns:
        The peak position of each molecule, in input order.

    Raises:
        ValueError: If the kernel or the bandwidth is not one.
    """
    sizes = np.asarray(utr_sizes_vector).ravel()
    check_kernel(kernel)
    parse_bandwidth(bandwidth)

    if len(sizes) == 0:  # a UTR without molecules, e.g. in a subset of the cells
        return np.zeros(0, dtype=np.int32)
    if len(sizes) <= FEWEST_FOR_A_DENSITY:
        return np.full(len(sizes), round(np.median(sizes)), dtype=np.int32)
    utr_bandwidth_ = utr_bandwidth(sizes, bandwidth)
    if utr_bandwidth_ == 0:  # a rule, on molecules all of one size
        return np.full(len(sizes), sizes[0], dtype=np.int32)

    # Evaluate the density at each size of a window padded beyond the sizes, so
    # that it rises to the first peak and falls after the last one.
    minimum_size, maximum_size = sizes.min(), sizes.max()
    padding = max((maximum_size - minimum_size) * 0.1, 10)
    window = np.arange(round(minimum_size - padding), round(maximum_size + padding) + 1)
    return _assign_to_peaks(sizes, window, _density(sizes, window, utr_bandwidth_, kernel))


def _density(
    sizes: npt.NDArray[np.integer[Any]],
    window: npt.NDArray[np.integer[Any]],
    bandwidth: float,
    kernel: str,
) -> npt.NDArray[np.floating[Any]]:
    """The KDE of *sizes* at each size of *window*, up to a constant factor.

    Where the density is flat, its peaks depend on equal values coming out
    equal. The Gaussian density is scikit-learn's log density, as in earlier
    versions, so its peaks stay the same. For the other kernels, scikit-learn's
    sums differ by rounding where they should be equal, which made peaks out of
    flat stretches in one 5' UTR in ten. They are summed here over the
    molecules per size instead: the linear, tophat and Epanechnikov densities
    from whole-number sums, which are exact; the cosine and exponential ones
    adding the sizes k nt before and after each size together, so densities
    that mirror each other come out equal.

    Args:
        sizes: The molecules' UTR sizes, all within *window*.
        window: Consecutive sizes.
        bandwidth: Bandwidth of the kernel.
        kernel: The kernel, one of settings.KERNELS.

    Returns:
        The density, or its log for the Gaussian kernel.
    """
    if kernel == "gaussian":
        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(sizes.reshape(-1, 1))
        return kde.score_samples(window.reshape(-1, 1))

    counts = np.bincount(sizes - window[0], minlength=len(window))
    if kernel == "exponential":
        reach = math.ceil(EXPONENTIAL_REACH * bandwidth)
    else:  # zero from the bandwidth on
        reach = math.ceil(bandwidth) - 1
    # The molecules lie in the window: none is further from a size of it.
    reach = min(reach, len(window) - 1)
    offsets = np.arange(-reach, reach + 1)
    if kernel in ("tophat", "linear", "epanechnikov"):

        def summed(weights: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
            """Sum over the molecules of their weight by offset, at each size of the window."""
            return np.convolve(counts, weights, mode="full")[reach : reach + len(window)]

        # Molecules within reach, and their summed distances and squared distances.
        within = summed(np.ones(len(offsets), dtype=np.int64))
        if kernel == "tophat":
            return within.astype(np.float64)
        if kernel == "linear":  # sum of (h - |k|) c(x + k)
            return bandwidth * within - summed(np.abs(offsets))
        return bandwidth**2 * within - summed(offsets**2)

    distances = np.arange(1, reach + 1)
    if kernel == "cosine":
        weights = np.cos(np.pi * distances / (2 * bandwidth))
    else:
        weights = np.exp(-distances / bandwidth)
    density = counts.astype(np.float64)  # the weight at distance 0 is 1
    for distance, weight in zip(distances, weights, strict=True):
        mirrored = np.zeros(len(window), dtype=np.int64)
        mirrored[distance:] += counts[:-distance]
        mirrored[:-distance] += counts[distance:]
        density += weight * mirrored
    return density


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
