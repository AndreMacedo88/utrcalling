"""Bandwidth rules: each UTR's bandwidth, set from the spread of its molecules' sizes.

A rule is a function of one UTR's molecule sizes (four or more, one per
molecule) that returns its bandwidth in nt. `-b scott` uses the rule
``scott``, and `-b scott:0.5` half of it. To add a rule, write such a function
and register it (see settings.BANDWIDTH_RULES and docs/extending.md)::

    from utrcalling.core.peak_calling.settings import BANDWIDTH_RULES

    @BANDWIDTH_RULES.register("iqr")
    def iqr(sizes):
        quartile_3, quartile_1 = np.percentile(sizes, [75, 25])
        return (quartile_3 - quartile_1) / 1.34 * len(sizes) ** -0.2

Both rules here narrow as the molecules grow in number, so deeply sequenced
UTRs get the finer peaks that their data support.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from utrcalling.core.peak_calling.settings import BANDWIDTH_RULES, Bandwidth, bandwidth_rule

# One UTR's molecule sizes, one per molecule, to its bandwidth in nt.
BandwidthRule = Callable[[npt.NDArray[np.integer[Any]]], float]

# With this many molecules or fewer, a UTR's molecules take their median size,
# without a KDE (see kde.call_peaks_using_kernel_density_estimator).
FEWEST_FOR_A_DENSITY = 3


def utr_bandwidth(sizes: npt.NDArray[np.integer[Any]], bandwidth: Bandwidth) -> float:
    """The bandwidth of the KDE of one UTR's molecules.

    A number is the bandwidth of every UTR. A rule sets it from the UTR's
    molecule sizes, times its factor (see :func:`scott` and :func:`silverman`).

    Args:
        sizes: The molecules' UTR sizes, one per molecule.
        bandwidth: A number of nt, or a rule (see settings.parse_bandwidth).

    Returns:
        The bandwidth, 0 if a rule meets molecules all of one size, or NaN for
        3 molecules or fewer, which get their median without a KDE.
    """
    if len(sizes) <= FEWEST_FOR_A_DENSITY:
        return math.nan
    if not isinstance(bandwidth, str):
        return float(bandwidth)
    rule, factor = bandwidth_rule(bandwidth)
    return factor * float(BANDWIDTH_RULES.get(rule)(np.asarray(sizes)))


def scott(sizes: npt.NDArray[np.integer[Any]]) -> float:
    """SD x n^-1/5, for n molecules whose sizes have the standard deviation SD.

    Scott's rule (1992), the default of SciPy's gaussian_kde. Eugenia Guerrero
    Aruffo used it, halved (``scott:0.5``), with the linear kernel for 5'
    Cap-seq data (master's thesis, 2026).
    """
    return float(np.std(sizes, ddof=1)) * len(sizes) ** -0.2


def silverman(sizes: npt.NDArray[np.integer[Any]]) -> float:
    """0.9 x min(SD, IQR / 1.34) x n^-1/5, with the SD if the IQR is 0.

    Silverman's rule of thumb (1986, eq. 3.31), as R's bw.nrd0, for n molecules
    whose sizes have the standard deviation SD and interquartile range IQR.
    Several sites in a UTR spread its sizes: they inflate the SD more than the
    IQR. Not SciPy's "silverman", which is Scott's rule times 1.06.
    """
    spread = float(np.std(sizes, ddof=1))
    quartile_3, quartile_1 = np.percentile(sizes, [75, 25])
    spread = min(spread, (quartile_3 - quartile_1) / 1.34) or spread
    return 0.9 * spread * len(sizes) ** -0.2
