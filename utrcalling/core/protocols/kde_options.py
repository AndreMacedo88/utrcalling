"""The settings of KDE peak calling, checked without heavy imports.

The command line checks them as it parses, so this module imports only the
standard library; core.protocols.peak_calling does the calling.
"""

from typing import Literal, get_args

# The kernels of scikit-learn's KernelDensity. The Gaussian (the default)
# smooths the most; it suits 3' data, whose sites spread over a few nt. The
# linear kernel (a triangle) is sharp at its centre and zero beyond the
# bandwidth, so its peaks stay on the sizes where the molecules are: it suits
# transcription start sites, which are exact to the nucleotide (5' data). For
# all kernels but the Gaussian and the exponential, the bandwidth is the
# half-width of the kernel, beyond which it is zero; for those two it is the
# standard deviation and the decay length.
Kernel = Literal["gaussian", "linear", "epanechnikov", "exponential", "tophat", "cosine"]
KERNELS: tuple[str, ...] = get_args(Kernel)


def check_kernel(kernel: str) -> str:
    """Return *kernel* if it is one of KERNELS.

    Raises:
        ValueError: If it is not.
    """
    if kernel not in KERNELS:
        raise ValueError(f"Unknown kernel {kernel!r}: expected one of {', '.join(KERNELS)}")
    return kernel
