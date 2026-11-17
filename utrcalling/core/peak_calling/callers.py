"""Peak callers: the methods that collapse one UTR's molecule sizes into sites.

A peak caller is a function that takes the UTR sizes of one UTR's molecules,
one per molecule, and returns the size of the site of each molecule, in the
same order. `run`, `count`, `call-peaks` and the `qc` commands all call peaks
through one (see count_matrix and peak_calling_anndata), so a new method works
in all of them.

Methods are registered by name (settings.PEAK_CALLERS, `--peak-caller`) as a
*factory*: a function that takes the bandwidth and the kernel of the command
line (`-b`, `--kde-kernel`) and returns the peak caller. utrcalling's method
is ``kde`` (see kde). A method without a kernel ignores it; one without a
bandwidth may still read it as its scale in nt, for `-b optimize` to choose.
The peak caller runs in worker processes, so it must pickle: a module-level
function, or a functools.partial of one, not a lambda or a nested function.
See docs/extending.md for an example.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from utrcalling.core.peak_calling.settings import PEAK_CALLERS, Bandwidth

# Maps one UTR's molecule sizes to the size of each molecule's site.
SizeTransform = Callable[[npt.ArrayLike], npt.NDArray[np.integer[Any]]]


class PeakCallerFactory(Protocol):
    """Builds a peak caller from the settings of the command line."""

    def __call__(self, bandwidth: Bandwidth, kernel: str) -> SizeTransform: ...


def get_caller(
    method: str = "kde", bandwidth: Bandwidth = 12, kernel: str = "gaussian"
) -> SizeTransform:
    """The peak caller of a method, with a bandwidth and a kernel.

    Args:
        method: The method's name in settings.PEAK_CALLERS, such as ``kde``.
        bandwidth: A number of nt, or a rule (see bandwidth_rules).
        kernel: The kernel, one of settings.KERNELS, for methods that have one.

    Returns:
        A function from one UTR's molecule sizes to the size of each
        molecule's site.

    Raises:
        ValueError: If no method has that name, or the method refuses the
            bandwidth or the kernel.
    """
    return PEAK_CALLERS.get(method)(bandwidth=bandwidth, kernel=kernel)
