"""Differential site usage: which UTRs change the use of their sites with a variable.

For each UTR with two or more sites, a Dirichlet-multinomial regression of
its site counts on the full model is compared with the reduced model by a
likelihood-ratio test, both at the precision that the full model estimates
(as DRIMSeq does): the statistic is chi-squared with (columns of the full
model - columns of the reduced) x (sites - 1) degrees of freedom. The
precision comes from :mod:`~utrcalling.core.stats.dirmult` (engine
``dirmult``) or from DRIMSeq (engine ``drimseq``).

The chi-squared p-value treats the units as independent replicates. Cells of
one sample are not: optionally, a permutation p-value shuffles the tested
variables in a way that respects the replicates (see
:mod:`~utrcalling.core.stats.permutation`), refitting everything, the
precision included, for each permutation. Effect sizes come from the fitted
full model (see :mod:`~utrcalling.core.stats.effects`).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import scipy.sparse
import scipy.stats

from utrcalling.core.stats import design as designs
from utrcalling.core.stats import dirmult, effects, measures, permutation

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    import anndata
    import numpy.typing as npt

ENGINES = ("dirmult", "drimseq")
MODERATIONS = ("trended", "common", "none")
# The strength of the precision's penalty is this over the square root of the units.
MODERATION_STRENGTH = 1.0
# The share of UTRs in each local fit of the precision trend.
TREND_SPAN = 0.3
# Statuses of a UTR whose test is reported.
TESTED = ("ok", "not_converged")


@dataclass(frozen=True)
class SiteUsageResult:
    """The result of a test of differential site usage.

    Attributes:
        genes: A row per UTR with two or more sites (see :func:`site_usage`).
        sites: A row per site of those UTRs.
        settings: How the test was run.
    """

    genes: pd.DataFrame
    sites: pd.DataFrame
    settings: dict[str, Any]


@dataclass(frozen=True)
class _Shared:
    """What every UTR's test needs, sent once to each batch of UTRs."""

    design: designs.Design
    contrast: designs.Contrast | None
    scheme: permutation.Scheme | None
    stages: tuple[int, ...]
    seed: int
    effects_only: bool


@dataclass
class _Job:
    """One UTR's test: its counts in the units with molecules, and its sites."""

    position: int
    rows: npt.NDArray[np.intp]
    counts: npt.NDArray[np.float64]
    ranks: npt.NDArray[np.float64]
    target: float = float("nan")
    precision: float = float("nan")
    shares_rows: npt.NDArray[np.intp] | None = None
    shares: npt.NDArray[np.float64] | None = None
    result: dict[str, Any] = field(default_factory=dict)


def _benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjusted p-values (NaN stays NaN)."""
    finite = p_values.dropna()
    adjusted = pd.Series(np.nan, index=p_values.index)
    if finite.empty:
        return adjusted
    order = np.argsort(finite.to_numpy())
    ranked = finite.to_numpy()[order] * len(finite) / np.arange(1, len(finite) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1)
    values = np.empty(len(finite))
    values[order] = ranked
    adjusted[finite.index] = values
    return adjusted


def _project(
    target: npt.NDArray[np.float64], linear: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Coefficients of *target*'s columns that best give the linear predictor *linear*."""
    return np.asarray(np.linalg.lstsq(target, linear, rcond=None)[0], dtype=np.float64)


def _likelihood_ratio(
    full: npt.NDArray[np.float64],
    reduced: npt.NDArray[np.float64],
    counts: dirmult.Counts,
    target: float,
    strength: float,
    *,
    start: npt.NDArray[np.float64] | None = None,
    start_precision: float | None = None,
    null_start: npt.NDArray[np.float64] | None = None,
) -> tuple[dirmult.Fit, dirmult.Fit]:
    """Fit the full model (with its precision) and the reduced one at that precision.

    If the reduced model fits better, which an optimiser stuck early can make
    happen, the full model is fitted again from the reduced fit, which it
    contains: it then fits at least as well.
    """
    initial_precision = start_precision if start_precision is not None else target
    full_fit = dirmult.fit(
        full,
        counts,
        start=start,
        start_precision=initial_precision if np.isfinite(initial_precision) else 10.0,
        target=target if np.isfinite(target) else None,
        strength=strength,
    )
    if null_start is None:
        null_start = _project(reduced, full @ full_fit.beta)
    null_fit = dirmult.fit(reduced, counts, precision=full_fit.precision, start=null_start)
    if full_fit.log_likelihood < null_fit.log_likelihood - dirmult.LIKELIHOOD_TOLERANCE:
        refit = dirmult.fit(
            full,
            counts,
            precision=full_fit.precision,
            start=_project(full, reduced @ null_fit.beta),
        )
        if refit.log_likelihood > full_fit.log_likelihood:
            full_fit = refit
    return full_fit, null_fit


def _test_utr(job: _Job, shared: _Shared) -> dict[str, Any]:
    """Test one UTR: the likelihood ratio, its permutations, its effect sizes."""
    design = shared.design
    counts = dirmult.Counts.from_matrix(job.counts)
    full_columns = designs.independent_columns(design.full[job.rows])
    reduced_columns = designs.independent_columns(design.reduced[job.rows])
    full = design.full[job.rows][:, full_columns]
    reduced = design.reduced[job.rows][:, reduced_columns]
    df = (len(full_columns) - len(reduced_columns)) * (counts.n_sites - 1)
    result: dict[str, Any] = {"df": df}
    if df < 1:
        return result | {"status": "not_identifiable"}
    if counts.n_units <= len(full_columns):
        return result | {"status": "too_few_units"}
    strength = MODERATION_STRENGTH / math.sqrt(counts.n_units) if np.isfinite(job.target) else 0.0

    if shared.effects_only:
        full_fit = dirmult.fit(full, counts, precision=job.precision)
        result["status"] = "ok"
    else:
        full_fit, null_fit = _likelihood_ratio(full, reduced, counts, job.target, strength)
        gap = full_fit.log_likelihood - null_fit.log_likelihood
        result |= {"precision": full_fit.precision, "precision_target": job.target}
        if gap < -dirmult.LIKELIHOOD_TOLERANCE:
            return result | {"status": "optimization_failed"}
        statistic = max(2 * gap, 0.0)
        converged = full_fit.converged and null_fit.converged
        result |= {
            "status": "ok" if converged else "not_converged",
            "statistic": statistic,
            "p_value": float(scipy.stats.chi2.sf(statistic, df)),
        }
        # The statistic of a fit that did not settle is unsure: no permutations.
        if shared.scheme is not None and converged:
            result |= _permute(
                job, shared, shared.scheme, counts, null_fit, reduced, statistic, strength
            )

    fitted = dirmult.proportions(full_fit.beta, full)
    ratios = dirmult.variance_ratios(fitted, counts, full_fit.precision)
    result |= {
        "max_variance_ratio": float(np.nanmax(ratios)) if np.isfinite(ratios).any() else np.nan,
        "min_variance_ratio": float(np.nanmin(ratios)) if np.isfinite(ratios).any() else np.nan,
    }
    if shared.contrast is not None:
        at_points = design.at_points(job.rows, shared.contrast)[:, :, full_columns]
        gene, sites = effects.effect_sizes(full_fit.beta, at_points, job.ranks, shared.contrast)
        result |= gene
        result["sites"] = sites
        if job.shares is not None and job.shares_rows is not None:
            shares, share_rows = job.shares, job.shares_rows
        else:
            shares, share_rows = job.counts / job.counts.sum(axis=1, keepdims=True), job.rows
        values = design.data[shared.contrast.variable].to_numpy()[share_rows]
        result |= effects.descriptive(shares, values, job.ranks, shared.contrast)
    return result


def _permute(
    job: _Job,
    shared: _Shared,
    scheme: permutation.Scheme,
    counts: dirmult.Counts,
    null_fit: dirmult.Fit,
    reduced: npt.NDArray[np.float64],
    observed: float,
    strength: float,
) -> dict[str, Any]:
    """The permutation p-value of one UTR's likelihood ratio."""
    design = shared.design
    units = design.data.iloc[job.rows]
    tested = units[list(scheme.variables)]
    null_linear = reduced @ null_fit.beta

    def statistic(sources: npt.NDArray[np.intp]) -> float:
        permuted = units.copy()
        for variable in scheme.variables:
            permuted[variable] = tested[variable].iloc[sources].to_numpy()
        matrix = design.evaluate_full(permuted)
        full = matrix[:, designs.independent_columns(matrix)]
        full_fit, permuted_null = _likelihood_ratio(
            full,
            reduced,
            counts,
            job.target,
            strength,
            start=_project(full, null_linear),
            start_precision=null_fit.precision,
            null_start=null_fit.beta,
        )
        return max(2 * (full_fit.log_likelihood - permuted_null.log_likelihood), 0.0)

    rng = np.random.default_rng([shared.seed, job.position])
    found = permutation.permutation_test(
        statistic, observed, scheme, job.rows, design.data, rng, shared.stages
    )
    return {
        "permutation_p_value": found.p_value,
        "permutations": found.permutations,
        "permutation_exact": found.exact,
    }


def _test_batch(jobs: list[_Job], shared: _Shared) -> list[tuple[int, dict[str, Any]]]:
    """Test a batch of UTRs, in a worker process: their positions and results."""
    return [(job.position, _test_utr(job, shared)) for job in jobs]


def _check_matrix(adata: anndata.AnnData) -> scipy.sparse.csc_matrix:
    """The counts of a matrix of sites, refusing raw sizes, measures and non-counts."""
    from utrcalling.core import provenance

    matrix = provenance.recorded(adata).get("matrix", "peaks")
    if matrix == "raw_sizes":
        raise ValueError(
            "This matrix holds raw sizes: a UTR has hundreds of sizes with a few molecules "
            "each, too many to model. Call its sites (call-peaks), remove the weak ones "
            "(process filter-sites), and test the site counts."
        )
    if matrix in provenance.MEASURES:
        raise ValueError(
            f"This matrix holds {matrix.replace('_', ' ')}: give the site counts it comes from."
        )
    counts = scipy.sparse.csc_matrix(adata.X)
    values = counts.data
    if values.size and ((values < 0).any() or not np.allclose(values, np.round(values))):
        raise ValueError(
            "The matrix holds values that are not counts (negative or fractional): test "
            "the molecule counts of sites, not normalised values."
        )
    return counts


def _aligned(
    adata: anndata.AnnData, column: str | None, design: designs.Design
) -> pd.Series | None:
    """A column of the annotations for the units of the design, without missing values."""
    if column is None:
        return None
    if column not in adata.obs.columns:
        raise ValueError(f"No column {column} in the cell annotations (.obs)")
    values = adata.obs[column][design.kept]
    if values.isna().any():
        raise ValueError(f"{int(values.isna().sum())} units of the models have no {column}")
    return values


def _check_replication(
    design: designs.Design, replicates: pd.Series | None, permute: bool, engine: str
) -> None:
    """Refuse cell-level p-values of variables that are the same within each replicate."""
    from utrcalling.core.module_log import module_logger

    if replicates is None:
        module_logger().info(
            f"No --replicates: each of the {len(design.data)} units counts as an "
            "independent replicate. Cells of one sample are not: give --replicates."
        )
        return
    codes = pd.factorize(replicates, sort=True)[0]
    if np.bincount(codes).max() == 1:
        return
    constant = permutation.constant_within(design.data, design.tested, codes)
    if not constant:
        return
    message = (
        f"{', '.join(constant)} is the same for all units of each {replicates.name}, so its "
        f"{len(design.data)} units are not independent replicates of it: their chi-squared "
        "p-values would be far too small."
    )
    remedy = (
        f" Sum them into one unit per replicate (utrcalling process pseudobulk --by "
        f"{replicates.name})"
    )
    if engine == "drimseq":
        raise ValueError(f"{message}{remedy}, or use the dirmult engine with --permute.")
    if not permute:
        raise ValueError(f"{message}{remedy}, or add --permute, which swaps it between replicates.")
    module_logger().warning(
        f"{message} Read the permutation p-values, which swap it between replicates."
    )


def _utr_groups(
    var: pd.DataFrame, utr_name_column: str
) -> tuple[list[str], list[npt.NDArray[np.intp]]]:
    """The UTRs with two or more sites, and their sites' positions, sorted by size."""
    names = var[utr_name_column].astype(str).to_numpy()
    sizes = var["utr_size"].to_numpy(float)
    order = np.lexsort((sizes, names))
    utrs, starts, numbers = np.unique(names[order], return_index=True, return_counts=True)
    kept = numbers >= 2
    return (
        [str(utr) for utr in utrs[kept]],
        [
            order[start : start + number]
            for start, number in zip(starts[kept], numbers[kept], strict=True)
        ],
    )


def _precision_targets(jobs: Sequence[_Job], moderation: str) -> npt.NDArray[np.float64]:
    """The precision each UTR's estimate is shrunk toward: the trend, the common value, or none."""
    from utrcalling.core.module_log import module_logger

    if moderation == "none" or not jobs:
        return np.full(len(jobs), np.nan)
    moments = np.array(
        [dirmult.moment_precision(dirmult.Counts.from_matrix(job.counts)) for job in jobs]
    )
    depths = np.array([job.counts.sum() / len(job.rows) for job in jobs])
    known = np.isfinite(moments)
    if known.sum() == 0:
        module_logger().info(
            "No UTR has the depth for a moment estimate of the precision: no moderation"
        )
        return np.full(len(jobs), np.nan)
    common = float(np.exp(np.median(np.log(moments[known]))))
    if moderation == "common" or known.sum() < 10:
        return np.full(len(jobs), common)
    trend = np.exp(dirmult.loess(np.log(depths[known]), np.log(moments[known]), span=TREND_SPAN))
    targets = np.full(len(jobs), float(np.median(trend)))
    targets[known] = trend
    return targets


def _jobs(
    counts: scipy.sparse.csc_matrix,
    utr_sites: Sequence[npt.NDArray[np.intp]],
    sizes: npt.NDArray[np.float64],
    min_units: int,
    min_counts: int,
) -> tuple[list[_Job], dict[int, dict[str, Any]], dict[int, npt.NDArray[np.intp]]]:
    """The UTRs to test, with the sites and units they use; and the statuses of the others."""
    jobs: list[_Job] = []
    skipped: dict[int, dict[str, Any]] = {}
    used_sites: dict[int, npt.NDArray[np.intp]] = {}
    for position, sites in enumerate(utr_sites):
        block = counts[:, sites].toarray().astype(float)
        site_totals = block.sum(axis=0)
        with_molecules = site_totals > 0
        used_sites[position] = sites[with_molecules]
        unit_totals = block.sum(axis=1)
        rows = np.flatnonzero(unit_totals > 0)
        summary = {
            "n_sites": int(with_molecules.sum()),
            "n_units": len(rows),
            "total_counts": int(site_totals.sum()),
        }
        if with_molecules.sum() < 2:
            skipped[position] = summary | {"status": "single_site"}
        elif len(rows) < min_units:
            skipped[position] = summary | {"status": "too_few_units"}
        elif site_totals.sum() < min_counts:
            skipped[position] = summary | {"status": "too_few_counts"}
        else:
            used = sites[with_molecules]
            site_ranks = measures.site_ranks(np.zeros(len(used)), sizes[used])
            job = _Job(position, rows, block[np.ix_(rows, with_molecules)], site_ranks)
            job.result = summary
            jobs.append(job)
    return jobs, skipped, used_sites


def _batches(jobs: Sequence[_Job], n_batches: int) -> Iterator[list[_Job]]:
    """Deal the jobs round-robin into batches (no more than jobs), which spreads the slow ones."""
    for start in range(n_batches):
        yield list(jobs[start::n_batches])


def _run_jobs(jobs: list[_Job], shared: _Shared, n_jobs: int | None) -> None:
    """Test the jobs, in parallel; each job keeps its result."""
    import joblib

    from utrcalling.core import helpers
    from utrcalling.core.module_log import module_logger

    workers = helpers.resolve_n_jobs(n_jobs)
    if workers == 1 or len(jobs) < 2:
        for job in jobs:
            job.result |= _test_utr(job, shared)
        return
    by_position = {job.position: job for job in jobs}
    batches = list(_batches(jobs, min(len(jobs), workers * 8)))
    done = 0
    with joblib.parallel_config(backend=helpers.JOBLIB_DEFAULT_BACKEND, inner_max_num_threads=1):
        for finished in joblib.Parallel(n_jobs=workers, return_as="generator")(
            joblib.delayed(_test_batch)(batch, shared) for batch in batches
        ):
            for position, result in finished:
                by_position[position].result |= result
            done += 1
            if done % max(1, len(batches) // 10) == 0:
                module_logger().info(f"Tested {done} of {len(batches)} batches of UTRs")
    helpers.shutdown_joblib_pool()


def site_usage(
    adata: anndata.AnnData,
    full: str,
    reduced: str = "~ 1",
    *,
    engine: Literal["dirmult", "drimseq"] = "dirmult",
    replicates: str | None = None,
    strata: str | None = None,
    permute: bool = False,
    permutations: Sequence[int] = permutation.STAGES,
    seed: int = 0,
    effect_of: str | None = None,
    effect_range: tuple[float, float] = (10, 90),
    contrast: tuple[Any, Any] | None = None,
    min_units: int = 10,
    min_counts: int = 10,
    precision_moderation: Literal["trended", "common", "none"] = "trended",
    smooth_graph: str | None = None,
    smooth_rounds: int = 2,
    rscript: str = "Rscript",
    utr_name_column: str = "utr_name",
    n_jobs: int | None = None,
) -> SiteUsageResult:
    """Test which UTRs change the use of their sites between two models.

    Args:
        adata: Site counts (peaks), units (cells or samples) by sites, with
            the models' variables in ``.obs``.
        full: The full model, as in ``~ batch + bs(pseudotime, df=5)``.
        reduced: The reduced model, nested in the full one, as in ``~ batch``.
        engine: ``dirmult`` (utrcalling's own fit) or ``drimseq`` (DRIMSeq's,
            in R, for few samples).
        replicates: Column of ``.obs`` naming each unit's biological
            replicate (the sample of a cell): refuses chi-squared p-values of
            tested variables that are the same within replicates, and decides
            how permutations shuffle.
        strata: Column of ``.obs`` within which replicates swap the tested
            variables, such as the family of paired samples.
        permute: Add a permutation p-value (engine ``dirmult``).
        permutations: Totals of random permutations after which to stop once
            the statistic was reached 5 times; with no more distinct
            permutations than the last, all are used (exact).
        seed: Seed of the permutations.
        effect_of: The variable whose effects to measure, by default the one
            tested variable.
        effect_range: Percentiles of a numeric variable between which effects
            are read.
        contrast: For a categorical variable, the level to compare and the
            reference level (by default the last and the first).
        min_units: Units with molecules a UTR needs to be tested.
        min_counts: Molecules a UTR needs, in all units, to be tested.
        precision_moderation: Shrink each UTR's precision toward the trend
            over UTRs of similar depth, their common value, or not at all.
        smooth_graph: Key of a graph of neighbouring cells in ``.obsp``: the
            descriptive measures (not the test) use counts averaged over it.
        smooth_rounds: How many times to average over *smooth_graph*.
        rscript: The Rscript program of the drimseq engine.
        utr_name_column: Column of ``.var`` with the UTR names.
        n_jobs: Worker processes; by default all cores but one.

    Returns:
        The genes table, indexed by UTR: ``gene_name`` (if annotated),
        ``n_sites``, ``n_units``, ``total_counts``, ``status`` (``ok``,
        ``not_converged``, ``optimization_failed``, ``not_identifiable``, or
        why it was not tested: ``single_site``, ``too_few_units``,
        ``too_few_counts``), ``statistic``, ``df``, ``p_value`` and
        ``p_adjusted`` (Benjamini-Hochberg over the UTRs with status ok),
        ``precision``; with ``dirmult``, ``precision_target`` and the
        variance ratios; with permutations, ``permutation_p_value``,
        ``permutation_p_adjusted``, ``permutations`` and
        ``permutation_exact``; and the effect sizes (see
        :mod:`~utrcalling.core.stats.effects`). The sites table gives each
        site's UTR, size, scaled rank and proportions; with ``drimseq``, its
        own test. The settings record how the test ran.

    Raises:
        ValueError: If an option is unknown or invalid, if the matrix does not
            hold site counts, or for the reasons of
            :func:`~utrcalling.core.stats.design.build`, of
            :func:`~utrcalling.core.stats.permutation.scheme`, and when tested
            variables are the same within replicates (see :func:`_check_replication`).
        RuntimeError: If the drimseq engine cannot run DRIMSeq.
    """
    from utrcalling._metadata import __version__
    from utrcalling.core import anndata_processing as ap
    from utrcalling.core import provenance
    from utrcalling.core.module_log import module_logger
    from utrcalling.core.stats import drimseq

    if engine not in ENGINES:
        raise ValueError(f"Unknown engine {engine!r}: expected one of {', '.join(ENGINES)}")
    if precision_moderation not in MODERATIONS:
        raise ValueError(
            f"Unknown precision moderation {precision_moderation!r}: expected one of "
            f"{', '.join(MODERATIONS)}"
        )
    stages = tuple(int(stage) for stage in permutations)
    if not stages or stages[0] < 1 or any(b <= a for a, b in itertools.pairwise(stages)):
        raise ValueError(f"Permutations {stages}: give increasing totals, from 1 up")
    if min_units < 2:
        raise ValueError(f"min_units {min_units}: a test needs 2 units or more")
    if permute and engine == "drimseq":
        raise ValueError(
            "Permutations refit the model hundreds of times per UTR: use the dirmult engine "
            "with --permute (on the same units) for a permutation p-value."
        )
    counts = _check_matrix(adata)
    design = designs.build(adata.obs, full, reduced)
    replicate_values = _aligned(adata, replicates, design)
    strata_values = _aligned(adata, strata, design)
    if strata is not None and not permute:
        raise ValueError("--strata says how permutations swap values: add --permute")
    _check_replication(design, replicate_values, permute, engine)
    scheme = None
    if permute:
        scheme = permutation.scheme(
            design.data, design.tested, design.reduced_variables, replicate_values, strata_values
        )
        module_logger().info(
            f"Permutations shuffle {', '.join(scheme.variables)} {scheme.kind} replicates"
        )
    chosen = design.contrast(effect_of, effect_range, contrast)
    drimseq_version = drimseq.check(rscript) if engine == "drimseq" else None

    var = adata.var
    utrs, utr_sites = _utr_groups(var, utr_name_column)
    if not utrs:
        raise ValueError("No UTR has two or more sites: there is no site usage to compare.")
    units_counts = counts[design.kept]
    jobs, skipped, used_sites = _jobs(
        units_counts.tocsc(), utr_sites, var["utr_size"].to_numpy(float), min_units, min_counts
    )
    module_logger().info(
        f"Testing {len(jobs)} of {len(utrs)} multi-site UTRs on {len(design.data)} units "
        f"({design.full_formula} against {design.reduced_formula})"
    )
    graph = None if smooth_graph is None else ap.neighbour_graph(adata, smooth_graph)
    if graph is not None and chosen is not None:
        smoothed = ap.smooth_counts(scipy.sparse.csr_matrix(adata.X), graph, smooth_rounds)
        smoothed = smoothed[design.kept].tocsc()
        for job in jobs:
            block = smoothed[:, used_sites[job.position]].toarray()
            totals = block.sum(axis=1)
            job.shares_rows = np.flatnonzero(totals > 0)
            job.shares = block[job.shares_rows] / totals[job.shares_rows, None]

    settings: dict[str, Any] = {
        "utrcalling": __version__,
        "command": provenance.command_line(),
        "engine": engine,
        "full": full,
        "reduced": reduced,
        "tested": list(design.tested),
        "units": len(design.data),
        "units_left_out": int((~design.kept).sum()),
        "replicates": replicates,
        "min_units": min_units,
        "min_counts": min_counts,
        "precision_moderation": precision_moderation,
    }
    if chosen is not None:
        settings["effect_of"] = chosen.variable
        settings["contrast"] = {"from": _plain(chosen.start), "to": _plain(chosen.end)}
        module_logger().info(
            f"Effect sizes of {chosen.variable}, from {chosen.start} to {chosen.end}"
        )
    if scheme is not None:
        settings["permutation"] = {
            "shuffles": f"{scheme.kind} replicates",
            "strata": strata,
            "stages": list(stages),
            "stop_after": permutation.STOP_AFTER,
            "seed": seed,
        }
    if smooth_graph is not None:
        settings["smoothing"] = f"{smooth_graph} x {smooth_rounds}"

    drimseq_found = None
    if engine == "drimseq":
        full_columns = designs.independent_columns(design.full)
        reduced_columns = designs.independent_columns(design.reduced)
        drimseq_found = drimseq.run(
            [utrs[job.position] for job in jobs],
            [list(var.index[used_sites[job.position]]) for job in jobs],
            [units_counts[:, used_sites[job.position]].toarray() for job in jobs],
            design.full[:, full_columns],
            design.reduced[:, reduced_columns],
            moderation=precision_moderation,
            cores=_cores(n_jobs),
            seed=seed,
            rscript=rscript,
        )
        settings["versions"] = drimseq_found.versions
        settings["drimseq"] = drimseq_version
        precisions = drimseq_found.genes["precision"]
        for job in jobs:
            job.precision = float(precisions[utrs[job.position]])
            if not np.isfinite(job.precision):
                skipped[job.position] = job.result | {"status": "not_tested"}
        jobs = [job for job in jobs if np.isfinite(job.precision)]
    else:
        for job, target in zip(jobs, _precision_targets(jobs, precision_moderation), strict=True):
            job.target = float(target)

    shared = _Shared(design, chosen, scheme, stages, seed, effects_only=engine == "drimseq")
    _run_jobs(jobs, shared, n_jobs)

    genes, sites = _tables(var, utrs, utr_sites, used_sites, jobs, skipped, utr_name_column)
    if drimseq_found is not None:
        found = drimseq_found.genes
        tested = genes["status"].eq("ok") & found.reindex(genes.index)["p_value"].notna()
        for column in ("statistic", "df", "p_value", "precision"):
            genes[column] = found.reindex(genes.index)[column].where(tested)
        genes.loc[genes["status"].eq("ok") & ~tested, "status"] = "not_tested"
        site_found = drimseq_found.sites.reindex(sites.index)
        sites["statistic"] = site_found["statistic"]
        sites["p_value"] = site_found["p_value"]
        sites["p_adjusted"] = _benjamini_hochberg(sites["p_value"])
    genes.insert(
        genes.columns.get_loc("p_value") + 1,
        "p_adjusted",
        _benjamini_hochberg(genes["p_value"].where(genes["status"].eq("ok"))),
    )
    if "permutation_p_value" in genes:
        genes.insert(
            genes.columns.get_loc("permutation_p_value") + 1,
            "permutation_p_adjusted",
            _benjamini_hochberg(genes["permutation_p_value"].where(genes["status"].eq("ok"))),
        )
    module_logger().info(
        "UTRs by status: "
        + ", ".join(
            f"{status} {number}" for status, number in genes["status"].value_counts().items()
        )
    )
    return SiteUsageResult(genes, sites, settings)


def _cores(n_jobs: int | None) -> int:
    """The worker processes of DRIMSeq."""
    from utrcalling.core import helpers

    return helpers.resolve_n_jobs(n_jobs)


def _plain(value: object) -> object:
    """A value that JSON can write: numpy numbers become Python numbers."""
    return value.item() if isinstance(value, np.generic) else value


# Columns of the genes table, in order (those a run computes).
GENE_COLUMNS = (
    "gene_name",
    "n_sites",
    "n_units",
    "total_counts",
    "status",
    "statistic",
    "df",
    "p_value",
    "precision",
    "precision_target",
    "permutation_p_value",
    "permutations",
    "permutation_exact",
    "length_index_from",
    "length_index_to",
    "delta_length_index",
    "length_index_range",
    "min_at",
    "max_at",
    "non_monotonicity",
    "tvd",
    "max_abs_delta_proportion",
    "rho_length_index",
    "rho_entropy",
    "observed_delta_length_index",
    "observed_delta_entropy",
    "max_variance_ratio",
    "min_variance_ratio",
)


def _tables(
    var: pd.DataFrame,
    utrs: Sequence[str],
    utr_sites: Sequence[npt.NDArray[np.intp]],
    used_sites: dict[int, npt.NDArray[np.intp]],
    jobs: Sequence[_Job],
    skipped: dict[int, dict[str, Any]],
    utr_name_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The genes and sites tables, from the jobs' results."""
    rows: dict[int, dict[str, Any]] = dict(skipped)
    site_effects: dict[str, npt.NDArray[np.float64]] = {"rank": np.full(len(var), np.nan)}
    for job in jobs:
        result = dict(job.result)
        found = result.pop("sites", None)
        rows[job.position] = result
        site_effects["rank"][used_sites[job.position]] = job.ranks
        if found is not None:
            for name, values in found.items():
                site_effects.setdefault(name, np.full(len(var), np.nan))[
                    used_sites[job.position]
                ] = values
    genes = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    genes.index = pd.Index([utrs[position] for position in genes.index])
    if "gene_name" in var.columns:
        first_site = {utrs[position]: sites[0] for position, sites in enumerate(utr_sites)}
        genes["gene_name"] = [str(var["gene_name"].iloc[first_site[utr]]) for utr in genes.index]
    for column in ("statistic", "p_value", "precision"):
        if column not in genes:
            genes[column] = np.nan
    genes = genes[[column for column in GENE_COLUMNS if column in genes.columns]]
    genes = genes.astype({"n_sites": int, "n_units": int, "total_counts": int})
    if "df" in genes:
        genes["df"] = genes["df"].astype("Int64")

    in_tests = np.concatenate(list(utr_sites))
    site_columns = [utr_name_column, "utr_size"] + (["gene_name"] if "gene_name" in var else [])
    sites = var.iloc[in_tests][site_columns].copy()
    for name, values in site_effects.items():
        sites[name] = values[in_tests]
    return genes, sites
