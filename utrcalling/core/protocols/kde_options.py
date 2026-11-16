"""The settings of KDE peak calling, checked without heavy imports.

The command line checks them as it parses, so this module imports only the
standard library; core.protocols.peak_calling does the calling.
"""

import math
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


# A bandwidth is a number of nt, the same for every UTR, or a rule that sets one
# per UTR from the spread of its molecules' sizes, optionally scaled:
# "scott", "scott:0.5", "silverman:2". The rules (see peak_calling.utr_bandwidth):
# scott, SD x n^-1/5, the default of SciPy's gaussian_kde (Scott 1992); and
# silverman, 0.9 x min(SD, IQR / 1.34) x n^-1/5, the rule of thumb of
# Silverman (1986, eq. 3.31) and R's bw.nrd0. Silverman's suits UTRs with
# several sites better: their spread inflates the SD, but less the IQR.
Bandwidth = int | float | str
BANDWIDTH_RULES = ("scott", "silverman")


def parse_bandwidth(value: Bandwidth, *, allow_zero: bool = False) -> Bandwidth:
    """A bandwidth as the record and the file names write it.

    Whole numbers become ints (12, not 12.0), other numbers floats, and rules
    lower case, without a factor of 1 ("scott:1" is "scott").

    Args:
        value: A number of nt, or a rule with an optional factor, as text or
            as a number.
        allow_zero: Whether 0 is a bandwidth (no peak calling, for call-peaks).

    Raises:
        ValueError: If *value* is neither, or not above 0 (or 0, if allowed).
    """
    text = str(value).strip().lower()
    rule, _, factor_text = text.partition(":")
    if rule in BANDWIDTH_RULES:
        factor = _number(
            factor_text or "1", f"bandwidth {value!r}: the factor of {rule} must be a number"
        )
        if not factor > 0:
            raise ValueError(f"bandwidth {value!r}: the factor of {rule} must be above 0")
        return rule if factor == 1 else f"{rule}:{_compact(factor)}"
    number = _number(text, f"bandwidth {value!r}: expected a number of nt, scott or silverman")
    if not (number > 0 or (allow_zero and number == 0)):
        raise ValueError(f"bandwidth {value!r}: must be above 0")
    return int(number) if number.is_integer() else number


def bandwidth_rule(bandwidth: str) -> tuple[str, float]:
    """The rule of a bandwidth and its factor: "scott:0.5" gives ("scott", 0.5)."""
    rule, _, factor = str(parse_bandwidth(bandwidth)).partition(":")
    return rule, float(factor or 1)


def bandwidth_label(bandwidth: Bandwidth) -> str:
    """A bandwidth as file names write it: 12, 7.5, scott, scott0.5."""
    return str(parse_bandwidth(bandwidth, allow_zero=True)).replace(":", "")


def _number(text: str, message: str) -> float:
    """*text* as a finite number, or a ValueError with *message*."""
    try:
        number = float(text)
    except ValueError:
        raise ValueError(message) from None
    if not math.isfinite(number):
        raise ValueError(message)
    return number


def _compact(number: float) -> str:
    return str(int(number)) if number.is_integer() else repr(number)


# The bandwidths that `qc score-bandwidths` scores, and `-b optimize` chooses
# from, by default: for 3' data with the PAS score, for 5' data with the
# initiator score (see peak_calling_anndata.score_bandwidths). On the 3' data of
# mouse neural stem cells (Kalamakis et al. 2019) the PAS score barely changes
# between 6 and 20 nt, the paper's 12 included; rules score lower there. On 5'
# Cap-seq data (Eugenia Guerrero Aruffo's thesis, 2026), the initiator score is
# highest with the linear kernel from 1 to 3 nt, or scott:0.25 to 0.5, and falls
# with the Gaussian kernel as the bandwidth grows.
CANDIDATE_BANDWIDTHS: dict[str, tuple[Bandwidth, ...]] = {
    "three_prime": (4, 6, 8, 10, 12, 15, 20),
    "five_prime": (
        *(1, 2, 3, 5, 8, 12),
        *("scott:0.25", "scott:0.5", "scott", "silverman:0.5", "silverman"),
    ),
}
