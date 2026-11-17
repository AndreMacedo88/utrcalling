"""The settings of peak calling and the names of its methods, checked without heavy imports.

The command line checks them as it parses, so this module imports only the
standard library: the registries below import a method only when it is used.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal, get_args

from utrcalling.core.registry import Registry

if TYPE_CHECKING:
    from utrcalling.core.peak_calling.bandwidth_rules import BandwidthRule
    from utrcalling.core.peak_calling.callers import PeakCallerFactory
    from utrcalling.core.peak_calling.scores import SiteScore

# ---------------------------------------------------------------------------
# The methods, by name. Plugins add theirs (see core.registry).
# ---------------------------------------------------------------------------

# How to call peaks (--peak-caller): each name gives a function that builds a
# peak caller from a bandwidth and a kernel (see callers).
PEAK_CALLERS: Registry[PeakCallerFactory] = Registry(
    "peak caller",
    "utrcalling.peak_callers",
    {"kde": "utrcalling.core.peak_calling.kde:kde_caller"},
)
# Rules that set each UTR's bandwidth from its molecules' sizes, as in
# -b scott:0.5 (see bandwidth_rules).
BANDWIDTH_RULES: Registry[BandwidthRule] = Registry(
    "bandwidth rule",
    "utrcalling.bandwidth_rules",
    {
        "scott": "utrcalling.core.peak_calling.bandwidth_rules:scott",
        "silverman": "utrcalling.core.peak_calling.bandwidth_rules:silverman",
    },
)
# Scores of peak-calling settings, by how well their sites fall where true
# sites are (--score; see scores).
SCORES: Registry[type[SiteScore]] = Registry(
    "score",
    "utrcalling.scores",
    {
        "pas": "utrcalling.core.peak_calling.scores.pas:PasScore",
        "initiator": "utrcalling.core.peak_calling.scores.initiator:InitiatorScore",
        "known-sites": "utrcalling.core.peak_calling.scores.known_sites:KnownSitesScore",
    },
)
# The score of the sites of each UTR end, by default: the polyadenylation
# signal before 3' sites, the initiator at 5' sites.
DEFAULT_SCORES = {"three_prime": "pas", "five_prime": "initiator"}

# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Bandwidths
# ---------------------------------------------------------------------------

# A bandwidth is a number of nt, the same for every UTR, or a rule that sets one
# per UTR from the spread of its molecules' sizes, optionally scaled:
# "scott", "scott:0.5", "silverman:2" (see bandwidth_rules).
Bandwidth = int | float | str


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
    try:
        number = float(text)
    except ValueError:
        number = None
    if number is not None:
        if not math.isfinite(number):
            raise ValueError(f"bandwidth {value!r}: expected a number of nt")
        if not (number > 0 or (allow_zero and number == 0)):
            raise ValueError(f"bandwidth {value!r}: must be above 0")
        return int(number) if number.is_integer() else number

    rule, _, factor_text = text.partition(":")
    if rule not in BANDWIDTH_RULES:
        raise ValueError(
            f"bandwidth {value!r}: expected a number of nt or a rule "
            f"({', '.join(BANDWIDTH_RULES.names())}), as in scott:0.5"
        )
    message = f"bandwidth {value!r}: the factor of {rule} must be a number above 0"
    try:
        factor = float(factor_text or "1")
    except ValueError:
        raise ValueError(message) from None
    if not (math.isfinite(factor) and factor > 0):
        raise ValueError(message)
    return rule if factor == 1 else f"{rule}:{_compact(factor)}"


def bandwidth_rule(bandwidth: str) -> tuple[str, float]:
    """The rule of a bandwidth and its factor: "scott:0.5" gives ("scott", 0.5)."""
    rule, _, factor = str(parse_bandwidth(bandwidth)).partition(":")
    return rule, float(factor or 1)


def bandwidth_label(bandwidth: Bandwidth) -> str:
    """A bandwidth as file names write it: 12, 7.5, scott, scott0.5."""
    return str(parse_bandwidth(bandwidth, allow_zero=True)).replace(":", "")


def setting_label(bandwidth: Bandwidth, kernel: str = "gaussian", method: str = "kde") -> str:
    """A peak-calling setting as file names write it: bw12, linear_bwscott0.5.

    The method and the kernel are named only when they are not the defaults,
    so Gaussian KDE files keep the names of earlier versions (peaks_bw12.h5ad).
    """
    parts = [method] if method != "kde" else []
    parts += [kernel] if kernel != "gaussian" else []
    return "_".join([*parts, f"bw{bandwidth_label(bandwidth)}"])


def _compact(number: float) -> str:
    return str(int(number)) if number.is_integer() else repr(number)


# ---------------------------------------------------------------------------
# Choosing a bandwidth
# ---------------------------------------------------------------------------

# The bandwidth that `run`, `count` and `call-peaks` take to choose one: they
# score the candidates (below) and call peaks with the best.
OPTIMIZE = "optimize"

# The bandwidths that `qc score-bandwidths` scores, and `-b optimize` chooses
# from, by default: for 3' data with the PAS score, for 5' data with the
# initiator score (see scores). On the 3' data of mouse neural stem cells
# (Kalamakis et al. 2019) the PAS score barely changes between 6 and 20 nt, the
# paper's 12 included; rules score lower there. On 5' Cap-seq data (Eugenia
# Guerrero Aruffo's thesis, 2026), the initiator score is highest with the
# linear kernel from 1 to 3 nt, or scott:0.25 to 0.5, and falls with the
# Gaussian kernel as the bandwidth grows.
CANDIDATE_BANDWIDTHS: dict[str, tuple[Bandwidth, ...]] = {
    "three_prime": (4, 6, 8, 10, 12, 15, 20),
    "five_prime": (
        *(1, 2, 3, 5, 8, 12),
        *("scott:0.25", "scott:0.5", "scott", "silverman:0.5", "silverman"),
    ),
}
