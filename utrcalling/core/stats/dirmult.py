"""The Dirichlet-multinomial regression of a UTR's site counts.

A unit (cell or sample) with n molecules in a UTR of K sites spreads them over
the sites as a Dirichlet-multinomial: a multinomial whose site probabilities
themselves vary between units around their mean, the proportions pi, with a
spread set by the *precision* s (the larger, the closer to a multinomial;
the dispersion is 1 / (1 + s)). The proportions follow the design through a
softmax link, with the last site as reference:

    pi_k = exp(x' beta_k) / sum_j exp(x' beta_j),   beta_K = 0.

Each UTR has its own precision, shared by its units. The likelihood of a
unit's counts y is, up to a term without parameters,

    lgamma(s) - lgamma(n + s) + sum_k [lgamma(y_k + s pi_k) - lgamma(s pi_k)],

where sites without molecules add nothing: the cost of each evaluation grows
with the molecules' entries, not with units x sites.

The precision of UTRs with few or shallow units is unsure. It is shrunk
toward a target, the trend of moment estimates against depth over all UTRs
(or their common value), by a penalty strength x (log s - log target)^2 with
strength 1 / sqrt(units), which fades as units accumulate. DRIMSeq (Nowicka
and Robinson 2016, F1000Research 5:1356) moderates its precision with
another method, on the same model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import scipy.optimize
import scipy.special

if TYPE_CHECKING:
    import numpy.typing as npt

# Limits of the log precision: from units that each take one site to a multinomial.
LOG_PRECISION_BOUNDS = (-5.0, 15.0)
# Convergence of L-BFGS-B: relative change of the objective, and projected gradient.
FTOL = 1e-10
GTOL = 1e-6
MAX_ITERATIONS = 5000
# A run that stops otherwise (such as a line search that cannot improve at the
# precision of the objective) has converged if its gradient is this small
# relative to the objective, a sum over units.
RELATIVE_GRADIENT = 1e-6
# Tolerance when comparing the full and reduced log-likelihoods.
LIKELIHOOD_TOLERANCE = 1e-6
# Limit of each coefficient times its design column's largest absolute value,
# which only keeps the numbers finite. A site without molecules in a group of
# units (a batch, or a stretch of a trajectory) has coefficients whose
# estimates are infinite: tighter limits lower the likelihood of splines whose
# coefficients cancel out.
COEFFICIENT_BOUND = 1e4
# Steps and ridge of the regressions that give the starting coefficients.
INITIAL_STEPS = 25
INITIAL_RIDGE = 1e-6


@dataclass(frozen=True)
class Counts:
    """A UTR's counts in the units with molecules in it, as their nonzero entries.

    Attributes:
        rows: The unit of each entry.
        columns: The site of each entry.
        values: The molecules of each entry.
        totals: The molecules of each unit.
        n_sites: The number of sites.
    """

    rows: npt.NDArray[np.intp]
    columns: npt.NDArray[np.intp]
    values: npt.NDArray[np.float64]
    totals: npt.NDArray[np.float64]
    n_sites: int

    @classmethod
    def from_matrix(cls, matrix: npt.ArrayLike) -> Counts:
        """The counts of a dense matrix of units x sites (every unit with molecules)."""
        dense = np.asarray(matrix, float)
        rows, columns = np.nonzero(dense)
        return cls(rows, columns, dense[rows, columns], dense.sum(axis=1), dense.shape[1])

    @property
    def n_units(self) -> int:
        """The number of units."""
        return len(self.totals)

    def dense(self) -> npt.NDArray[np.float64]:
        """The counts as a matrix of units x sites."""
        matrix = np.zeros((self.n_units, self.n_sites))
        matrix[self.rows, self.columns] = self.values
        return matrix

    def constant(self) -> float:
        """The log-likelihood's term without parameters: the multinomial coefficients."""
        return float(
            scipy.special.gammaln(self.totals + 1).sum()
            - scipy.special.gammaln(self.values + 1).sum()
        )


@dataclass(frozen=True)
class Fit:
    """A fitted model of one UTR.

    Attributes:
        beta: The coefficients, a row per design column and a column per site
            but the last.
        precision: The precision s.
        log_likelihood: The log-likelihood, without :meth:`Counts.constant`
            and without the penalty of the precision.
        converged: Whether the optimiser converged.
    """

    beta: npt.NDArray[np.float64]
    precision: float
    log_likelihood: float
    converged: bool


def _log_proportions(beta: npt.ArrayLike, design: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """The log proportions of each unit's sites (a log-softmax, the last site as reference)."""
    linear = np.asarray(design, float) @ np.asarray(beta, float)
    linear = np.hstack([linear, np.zeros((linear.shape[0], 1))])
    return linear - scipy.special.logsumexp(linear, axis=1, keepdims=True)


def proportions(beta: npt.ArrayLike, design: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """The site proportions of each unit: a softmax with the last site as reference.

    Args:
        beta: Coefficients, design columns x (sites - 1).
        design: The units' design matrix.

    Returns:
        The proportions, units x sites; rows add up to 1.
    """
    return np.exp(_log_proportions(beta, design))


def _entry_terms(
    log_shares: npt.NDArray[np.float64], precision: float, values: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Each nonzero entry's likelihood term and its share times the term's derivative.

    lgamma(y + a) - lgamma(a), with a = s x share, is computed as
    log s + log share + lgamma(y + a) - lgamma(1 + a), and share x (digamma(y
    + a) - digamma(a)) as share x (digamma(y + a) - digamma(1 + a)) + 1 / s:
    both stay finite when a share is too small for floating point.
    """
    shares = np.exp(log_shares)
    alpha = precision * shares
    terms = (
        np.log(precision)
        + log_shares
        + scipy.special.gammaln(values + alpha)
        - scipy.special.gammaln(1 + alpha)
    )
    slopes = (
        shares * (scipy.special.digamma(values + alpha) - scipy.special.digamma(1 + alpha))
        + 1 / precision
    )
    return terms, slopes


def log_likelihood(
    beta: npt.ArrayLike, precision: float, design: npt.ArrayLike, counts: Counts
) -> float:
    """The log-likelihood of a UTR's counts, without :meth:`Counts.constant`.

    Args:
        beta: Coefficients, design columns x (sites - 1).
        precision: The precision s.
        design: The units' design matrix.
        counts: The UTR's counts.

    Returns:
        The log-likelihood.
    """
    log_shares = _log_proportions(beta, design)[counts.rows, counts.columns]
    terms, _ = _entry_terms(log_shares, precision, counts.values)
    return float(
        terms.sum()
        + counts.n_units * scipy.special.gammaln(precision)
        - scipy.special.gammaln(counts.totals + precision).sum()
    )


def _objective(
    parameters: npt.NDArray[np.float64],
    design: npt.NDArray[np.float64],
    counts: Counts,
    precision: float | None,
    target: float | None,
    strength: float,
) -> tuple[float, npt.NDArray[np.float64]]:
    """The negative (penalised) log-likelihood and its gradient, for L-BFGS-B.

    The parameters are the coefficients, column by column of beta, then the
    log precision when *precision* is None (a joint fit).
    """
    n_coefficients = design.shape[1] * (counts.n_sites - 1)
    beta = parameters[:n_coefficients].reshape(design.shape[1], counts.n_sites - 1, order="F")
    log_precision = parameters[n_coefficients] if precision is None else np.log(precision)
    s = float(np.exp(log_precision))
    log_shares = _log_proportions(beta, design)
    shares = np.exp(log_shares)
    terms, slopes = _entry_terms(log_shares[counts.rows, counts.columns], s, counts.values)
    value = (
        terms.sum()
        + counts.n_units * scipy.special.gammaln(s)
        - scipy.special.gammaln(counts.totals + s).sum()
    )
    # d/d(linear_k) = s (share_k d_k - share_k sum_j share_j d_j), d_k the
    # derivative of entry k's term in its alpha (0 where a site has no molecules).
    by_site = np.zeros_like(shares)
    by_site[counts.rows, counts.columns] = slopes
    weighted = by_site.sum(axis=1, keepdims=True)
    linear_gradient = s * (by_site[:, :-1] - shares[:, :-1] * weighted)
    gradient = (design.T @ linear_gradient).ravel(order="F")
    if precision is None:
        precision_gradient = s * (
            counts.n_units * scipy.special.digamma(s)
            - scipy.special.digamma(counts.totals + s).sum()
            + slopes.sum()
        )
        if target is not None and strength > 0:
            distance = log_precision - np.log(target)
            value -= strength * distance**2
            precision_gradient -= 2 * strength * distance
        gradient = np.append(gradient, precision_gradient)
    return -float(value), -gradient


def pooled_beta(design: npt.ArrayLike, counts: Counts) -> npt.NDArray[np.float64]:
    """Starting coefficients that put every unit at the UTR's pooled proportions.

    The pooled log-odds of each site against the last, as a least-squares
    combination of the design's columns (exact when they span a constant).
    """
    pooled = np.bincount(counts.columns, weights=counts.values, minlength=counts.n_sites) + 0.5
    log_odds = np.log(pooled[:-1]) - np.log(pooled[-1])
    matrix = np.asarray(design, float)
    target = np.ones((matrix.shape[0], 1)) * log_odds[None, :]
    return np.asarray(np.linalg.lstsq(matrix, target, rcond=None)[0], dtype=np.float64)


def binomial_beta(design: npt.ArrayLike, counts: Counts) -> npt.NDArray[np.float64]:
    """Starting coefficients from a logistic regression of each site against the rest.

    Each site's share follows its own logistic regression on the design,
    fitted by iteratively reweighted least squares (a few steps, with a small
    ridge that keeps the coefficients of sites absent from a group of units
    finite). The units' fitted shares, made to add up to 1, give log-ratios
    to the last site, whose least-squares coefficients start the softmax.
    """
    matrix = np.asarray(design, float)
    observed = counts.dense()
    totals = counts.totals
    pooled = observed.sum(axis=0) + 0.5
    pooled /= pooled.sum()
    linear = np.empty_like(observed)
    for site in range(counts.n_sites):
        constant = np.full(matrix.shape[0], scipy.special.logit(pooled[site]))
        beta = np.linalg.lstsq(matrix, constant, rcond=None)[0]
        for _ in range(INITIAL_STEPS):
            mean = scipy.special.expit(matrix @ beta)
            weights = totals * mean * (1 - mean) + 1e-10
            working = matrix @ beta + (observed[:, site] - totals * mean) / weights
            normal = matrix.T @ (weights[:, None] * matrix)
            normal += INITIAL_RIDGE * np.trace(normal) / len(normal) * np.eye(len(normal))
            beta = np.linalg.solve(normal, matrix.T @ (weights * working))
        linear[:, site] = matrix @ beta
    log_shares = scipy.special.log_expit(linear)
    log_shares -= scipy.special.logsumexp(log_shares, axis=1, keepdims=True)
    log_ratios = log_shares[:, :-1] - log_shares[:, [-1]]
    return np.asarray(np.linalg.lstsq(matrix, log_ratios, rcond=None)[0], dtype=np.float64)


def fit(
    design: npt.ArrayLike,
    counts: Counts,
    precision: float | None = None,
    *,
    start: npt.ArrayLike | None = None,
    start_precision: float = 10.0,
    target: float | None = None,
    strength: float = 0.0,
) -> Fit:
    """Fit the model of one UTR by maximum likelihood (L-BFGS-B, analytic gradients).

    Args:
        design: The units' design matrix, with linearly independent columns.
        counts: The UTR's counts in those units.
        precision: Fit the coefficients at this precision; with None, fit the
            precision too, jointly.
        start: Starting coefficients; by default the better (by likelihood)
            of :func:`pooled_beta` and :func:`binomial_beta`.
        start_precision: Starting precision of a joint fit.
        target: The precision that the penalty shrinks toward, in a joint fit.
        strength: The penalty's strength (0 for none).

    Returns:
        The fitted model.
    """
    matrix = np.asarray(design, float)
    low, high = LOG_PRECISION_BOUNDS
    initial_precision = precision if precision is not None else start_precision
    initial_precision = float(np.exp(np.clip(np.log(initial_precision), low, high)))
    if start is None:
        candidates = [pooled_beta(matrix, counts), binomial_beta(matrix, counts)]
        likelihoods = [log_likelihood(b, initial_precision, matrix, counts) for b in candidates]
        beta = candidates[int(np.nanargmax(likelihoods))]
    else:
        beta = np.asarray(start, float)
    # Fit on columns scaled to a largest absolute value of 1, which conditions
    # the problem and makes the bound mean the same for every column.
    scale = np.abs(matrix).max(axis=0)
    scale[scale == 0] = 1.0
    matrix = matrix / scale
    beta = beta * scale[:, None]
    parameters = np.clip(beta.ravel(order="F"), -COEFFICIENT_BOUND, COEFFICIENT_BOUND)
    bounds: list[tuple[float | None, float | None]] = [
        (-COEFFICIENT_BOUND, COEFFICIENT_BOUND)
    ] * len(parameters)
    if precision is None:
        parameters = np.append(parameters, np.log(initial_precision))
        bounds.append(LOG_PRECISION_BOUNDS)
    result = scipy.optimize.minimize(
        _objective,
        parameters,
        args=(matrix, counts, precision, target, strength),
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": FTOL, "gtol": GTOL, "maxiter": MAX_ITERATIONS},
    )
    n_coefficients = beta.size
    fitted = result.x[:n_coefficients].reshape(beta.shape, order="F")
    s = float(np.exp(result.x[n_coefficients])) if precision is None else float(precision)
    flat = np.abs(result.jac).max(initial=0) <= RELATIVE_GRADIENT * max(1.0, abs(result.fun))
    converged = bool(result.success) or bool(flat)
    likelihood = log_likelihood(fitted, s, matrix, counts)
    return Fit(fitted / scale[:, None], s, likelihood, converged)


def moment_precision(counts: Counts) -> float:
    """A moment estimate of the precision, from the spread of units around the pooled shares.

    Pearson's statistic X^2 = sum over units and sites of (y - n p)^2 / (n p),
    with p the pooled proportions, has the expectation
    (K - 1) [N - 1 + rho (sum (n_i - 1) - (sum n_i^2 / sum n_i - 1))] under the
    Dirichlet-multinomial, with rho = 1 / (1 + s) the dispersion. Units with
    one molecule say nothing about rho, so they leave it out by construction.

    Args:
        counts: A UTR's counts.

    Returns:
        The precision, or NaN if the units carry no information on it (all
        with one molecule); within :data:`LOG_PRECISION_BOUNDS`.
    """
    totals = counts.totals
    pooled = np.bincount(counts.columns, weights=counts.values, minlength=counts.n_sites)
    pooled = pooled / pooled.sum()
    expected = totals[:, None] * pooled[None, :]
    observed = counts.dense()
    used = pooled > 0
    pearson = ((observed[:, used] - expected[:, used]) ** 2 / expected[:, used]).sum()
    information = (totals - 1).sum() - ((totals**2).sum() / totals.sum() - 1)
    if information <= 0:
        return float("nan")
    rho = (pearson / (used.sum() - 1) - (counts.n_units - 1)) / information
    low, high = np.exp(LOG_PRECISION_BOUNDS)
    if rho <= 0:
        return float(high)
    return float(np.clip(1 / rho - 1, low, high))


def loess(
    x: npt.ArrayLike, y: npt.ArrayLike, span: float = 0.3, points: int = 200
) -> npt.NDArray[np.float64]:
    """A local quadratic regression of *y* on *x* (as R's loess), at each *x*.

    Each fit takes the nearest *span* share of the points, weighted by the
    tricube of their distance. The curve is evaluated on *points* evenly
    spaced values of *x* and interpolated between them.

    Args:
        x: The predictor.
        y: The response.
        span: The share of points in each local fit.
        points: The number of values the curve is evaluated at.

    Returns:
        The fitted values at each *x*.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    neighbours = min(len(x), max(3, int(np.ceil(span * len(x)))))
    grid = np.linspace(x.min(), x.max(), points) if x.max() > x.min() else x[:1]
    fitted = np.empty(len(grid))
    for position, centre in enumerate(grid):
        distances = np.abs(x - centre)
        # The nearest points, and any as near as the last of them.
        reach = np.partition(distances, neighbours - 1)[neighbours - 1]
        nearest = distances <= reach
        weights = (
            (1 - (distances[nearest] / reach) ** 3) ** 3 if reach > 0 else np.ones(nearest.sum())
        )
        offsets = x[nearest] - centre
        basis = np.column_stack([np.ones(len(offsets)), offsets, offsets**2])
        root = np.sqrt(weights)
        coefficients = np.linalg.lstsq(basis * root[:, None], y[nearest] * root, rcond=None)[0]
        fitted[position] = coefficients[0]
    return np.interp(x, grid, fitted)


def variance_ratios(
    fitted: npt.NDArray[np.float64], counts: Counts, precision: float
) -> npt.NDArray[np.float64]:
    """How much more the sites' shares spread around the fitted proportions than the model allows.

    Under the fitted model the share of site k in a unit with n molecules
    has the variance pi_k (1 - pi_k) (n + s) / (n (1 + s)). The ratio of the
    mean squared difference between the shares and pi_k to that variance's
    mean is near 1 for a good fit; much above 1, the units spread more than
    the precision allows, and the chi-squared p-values are too small.

    Args:
        fitted: The fitted proportions, units x sites.
        counts: The UTR's counts.
        precision: The fitted precision s.

    Returns:
        The ratio for each site; NaN with fewer than 3 units.
    """
    if counts.n_units < 3:
        return np.full(counts.n_sites, np.nan)
    totals = counts.totals[:, None]
    shares = counts.dense() / totals
    observed = ((shares - fitted) ** 2).mean(axis=0)
    expected = (fitted * (1 - fitted) * (totals + precision) / (totals * (1 + precision))).mean(
        axis=0
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(expected > 0, observed / expected, np.nan)
