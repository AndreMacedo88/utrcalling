from typing import Callable

import numpy as np
import numpy.typing as npt
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema

# Methods to aggregate individual molecules' UTR sizes to consensus peaks:


def get_caller(method: str) -> Callable[[npt.ArrayLike, int], npt.ArrayLike]:

    method = method.lower()
    if method == "kde":
        return call_peaks_using_kernel_density_estimator
    else:
        raise NotImplementedError(
            (f"Method {method} for calling consensus peaks is not " f"implemented.")
        )


def call_peaks_using_kernel_density_estimator(utr_sizes_vector, bandwidth=12):
    """
    Fit a kernel density model to predict the peak's ditribution density function

    Args:
        utr_sizes_vector (np.ndarray): column vector with the UTR sizes to call peaks.

    Returns:
        np.ndarray: The vector with the peaks.
    """

    if len(utr_sizes_vector) <= 3:
        peaks = np.repeat(round(np.median(utr_sizes_vector)), len(utr_sizes_vector))
        return peaks

    if np.ndim(utr_sizes_vector) == 1:
        utr_sizes_vector = utr_sizes_vector.reshape(-1, 1)

    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(utr_sizes_vector)

    # calculate an appropriate window to check the model
    minimum_size, maximum_size = utr_sizes_vector.min(), utr_sizes_vector.max()
    padding = max((maximum_size - minimum_size) * 0.1, 10)
    window = np.arange(round(minimum_size - padding), round(maximum_size + padding) + 1)
    # get the model predictions for that window. Specifically, the extrema.
    evaluation = kde.score_samples(window.reshape(-1, 1))
    minima = argrelextrema(evaluation, np.less_equal, order=1)[0][1:]
    maxima = argrelextrema(evaluation, np.greater_equal, order=1)[0]
    # Note: [1:] because less_equal always outputs index 0,
    # since padding guarantees that kde rises until first value.

    minima = [minimum for minimum in minima if minimum - 1 not in minima]
    maxima = [maximum for maximum in maxima if maximum - 1 not in maxima]

    if minima[0] < maxima[0]:
        raise RuntimeError(
            "First minima is larger than the first maxima. \
            Peak calling using this method is incorrect for this UTR."
        )

    # create a container to hold the UTR peak sizes
    peaks = np.zeros(len(utr_sizes_vector))
    # attribute a peak to each cell according to their UTR sizes and the kde model:
    # first iterate over the function' minima
    idx_minimum = 0  # for the unbound warning... change it in the future!
    for idx_minimum, val in enumerate(window[minima]):
        # get the maximum upstream that minimum (the peak) and store it
        try:
            max_ = round(window[maxima][idx_minimum])
        except IndexError:
            continue
        for idx_maximum, cond in enumerate(utr_sizes_vector < val):
            if cond and peaks[idx_maximum] == 0:
                peaks[idx_maximum] = max_
    # get the last maximum, just downstream the last minimum.
    try:
        max_ = round(window[maxima][idx_minimum + 1])
    except IndexError:  # for those cases with just one peak, there will be no minimum
        # so we take the median as the peak
        max_ = round(np.median(utr_sizes_vector))

    # place the last maximum in those molecules that haven't yet had their
    # peaks scored
    peaks[peaks == 0] = max_
    return peaks


def bypass_calling_peaks(utr_sizes_vector):
    if np.issubdtype(utr_sizes_vector.dtype, np.integer):
        return utr_sizes_vector
    else:
        try:
            return np.rint(utr_sizes_vector)
        except TypeError:
            utr_sizes_vector = utr_sizes_vector.astype(float)
            return np.rint(utr_sizes_vector)
