"""Effect sizes of a fitted model, and descriptive measures of site use.

Effects are read from the fitted full model, the same model that is tested.
At each point of a contrast (two levels of a category, or a grid along a
numeric variable) every unit of the UTR gets that value, its proportions are
predicted, and they are averaged over the units: the proportions of the UTR's
units had they all had that value, the other variables staying as they are
(average predictive margins, or standardisation). From them:

- the length index at the start and end of the contrast and their
  difference, ``delta_length_index``, from -1 (from all at the longest site
  to all at the shortest) to 1;
- the total variation distance between the start and end proportions
  (half the sum of the absolute changes) and the largest change of a site;
- along a numeric variable, the range of the length index (signed: positive
  when its maximum comes after its minimum), where it is lowest and highest,
  and how far from monotonic it is: 1 - abs(delta) / abs(range) (0 for a
  monotonic change, near 1 when it rises and returns).

The descriptive measures need no model: the Spearman correlation of each
unit's length index and entropy with a numeric variable, or the difference
of their means between two levels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.stats

from utrcalling.core.stats import dirmult, measures

if TYPE_CHECKING:
    import numpy.typing as npt

    from utrcalling.core.stats.design import Contrast


def effect_sizes(
    beta: npt.NDArray[np.float64],
    at_points: npt.NDArray[np.float64],
    ranks: npt.NDArray[np.float64],
    contrast: Contrast,
) -> tuple[dict[str, float], dict[str, npt.NDArray[np.float64]]]:
    """The effect sizes of one UTR's fitted model.

    Args:
        beta: The fitted coefficients of the full model.
        at_points: The UTR's units' design at each point of the contrast,
            points x units x columns (the columns of *beta*).
        ranks: The sites' scaled ranks.
        contrast: Where the effects are read.

    Returns:
        The UTR's effect sizes, and each site's proportions at the start and
        end of the contrast and their difference.
    """
    margins = np.array([dirmult.proportions(beta, design).mean(axis=0) for design in at_points])
    index = measures.length_index(margins, ranks)
    start, end = margins[0], margins[-1]
    change = end - start
    gene = {
        "length_index_from": float(index[0]),
        "length_index_to": float(index[-1]),
        "delta_length_index": float(index[-1] - index[0]),
        "tvd": float(0.5 * np.abs(change).sum()),
        "max_abs_delta_proportion": float(np.abs(change).max()),
    }
    if contrast.kind == "numeric":
        lowest, highest = int(np.argmin(index)), int(np.argmax(index))
        spread = float(index[highest] - index[lowest])
        signed = spread if highest >= lowest else -spread
        gene |= {
            "length_index_range": signed,
            "min_at": float(contrast.points[lowest]),
            "max_at": float(contrast.points[highest]),
            "non_monotonicity": (
                0.0
                if spread < 1e-12
                else float(np.clip(1 - abs(gene["delta_length_index"]) / spread, 0, 1))
            ),
        }
    sites = {"proportion_from": start, "proportion_to": end, "delta_proportion": change}
    return gene, sites


def _spearman(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> float:
    """Spearman's correlation, NaN when either side does not vary."""
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(scipy.stats.spearmanr(x, y).statistic)


def descriptive(
    shares: npt.NDArray[np.float64],
    values: npt.NDArray[Any],
    ranks: npt.NDArray[np.float64],
    contrast: Contrast,
) -> dict[str, float]:
    """Model-free measures of how a UTR's site use follows the contrast's variable.

    Args:
        shares: Each unit's site shares (rows add up to 1).
        values: Each unit's value of the variable.
        ranks: The sites' scaled ranks.
        contrast: The variable and, for categories, the two levels.

    Returns:
        For a numeric variable, the Spearman correlation of the units' length
        index and entropy with it (``rho_length_index``, ``rho_entropy``);
        for categories, the difference of their means from the start level to
        the end level (``observed_delta_length_index``,
        ``observed_delta_entropy``; NaN if a level has no unit).
    """
    index = measures.length_index(shares, ranks)
    entropy = measures.normalized_entropy(shares)
    if contrast.kind == "numeric":
        numbers = np.asarray(values, float)
        return {
            "rho_length_index": _spearman(index, numbers),
            "rho_entropy": _spearman(entropy, numbers),
        }
    start = np.asarray(values == contrast.start, dtype=bool)
    end = np.asarray(values == contrast.end, dtype=bool)
    if not start.any() or not end.any():
        return {"observed_delta_length_index": np.nan, "observed_delta_entropy": np.nan}
    return {
        "observed_delta_length_index": float(index[end].mean() - index[start].mean()),
        "observed_delta_entropy": float(entropy[end].mean() - entropy[start].mean()),
    }
