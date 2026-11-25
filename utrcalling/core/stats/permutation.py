"""Permutations of the tested variables that respect the units' replicates.

A permutation test compares a statistic with its values after shuffling the
tested variables, which breaks any link between them and the site counts
while keeping everything else. It is valid when the units are exchangeable
under the null hypothesis, so which units are shuffled depends on the
replicates (Winkler et al. 2015, NeuroImage 123:268, on exchangeability blocks):

- ``within``: the tested variables vary within replicates (pseudotime among
  the cells of a sample). Values are shuffled among the units of each
  replicate, which keeps every replicate's own distribution of them.
- ``between``: the tested variables are the same for all units of a
  replicate (a sample's condition). The replicates swap their values, each
  keeping all its units, optionally only within strata (the two samples of
  a family). Units that are their own replicate (bulk samples) are the
  simplest case.

When the distinct permutations are few, all are used and the p-value is
exact; otherwise random ones are drawn, in stages, stopping once the
statistic has been reached often enough for a clearly large p-value (as in
Besag and Clifford 1991, Biometrika 78:301), and p = (b + 1) / (B + 1)
(Phipson and Smyth 2010, Stat Appl Genet Mol Biol 9:39).
"""

from __future__ import annotations

import itertools
import math
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    import numpy.typing as npt

# Stages of random permutations, and the exceedances that stop them early.
STAGES = (49, 199, 999)
STOP_AFTER = 5


@dataclass(frozen=True)
class Scheme:
    """How the tested variables are shuffled.

    Attributes:
        kind: ``within`` replicates, or ``between`` them.
        variables: The tested variables, shuffled together.
        replicate: The replicate of each unit of the design, as codes.
        stratum: The stratum of each unit, as codes (``between`` only; all 0
            without strata).
    """

    kind: Literal["within", "between"]
    variables: tuple[str, ...]
    replicate: npt.NDArray[np.intp]
    stratum: npt.NDArray[np.intp]


def constant_within(
    data: pd.DataFrame, variables: Sequence[str], groups: npt.ArrayLike
) -> list[str]:
    """The variables whose value is the same for all units of each group."""
    grouped = data.groupby(np.asarray(groups), sort=False)
    return [variable for variable in variables if (grouped[variable].nunique() <= 1).all()]


def scheme(
    data: pd.DataFrame,
    tested: Sequence[str],
    reduced_variables: Sequence[str] = (),
    replicates: pd.Series | None = None,
    strata: pd.Series | None = None,
) -> Scheme:
    """Choose how to shuffle the tested variables, from the replicates.

    Args:
        data: The variables of the models, a row per unit.
        tested: The variables to shuffle.
        reduced_variables: The variables of the reduced model, which the
            shuffles must not break away from the counts.
        replicates: The replicate of each unit, aligned with *data*; by
            default each unit is its own.
        strata: The stratum of each unit, within which replicates swap values.

    Returns:
        The scheme.

    Raises:
        ValueError: If nothing is tested, if some tested variables vary
            within replicates and others don't, if strata are given while the
            tested variables vary within replicates, or if a replicate spans
            strata.
    """
    from utrcalling.core.module_log import module_logger

    if not tested:
        raise ValueError(
            "The full model adds no variable to shuffle (it only adds interactions of "
            "variables in the reduced model): permutations need a tested variable."
        )
    replicate = (
        np.arange(len(data)) if replicates is None else pd.factorize(replicates, sort=True)[0]
    )
    constant = constant_within(data, tested, replicate)
    if constant and len(constant) < len(tested):
        varying = [variable for variable in tested if variable not in constant]
        raise ValueError(
            f"{', '.join(constant)} are the same within replicates but {', '.join(varying)} "
            "vary within them: test them separately to permute them."
        )
    others = [variable for variable in reduced_variables if variable not in tested]
    if not constant:
        if strata is not None:
            raise ValueError(
                "--strata applies when the tested variables are the same for all units of a "
                "replicate, which then swap them; here they vary within replicates, which "
                "shuffle them among their own units."
            )
        loose = [v for v in others if v not in constant_within(data, [v], replicate)]
        if loose:
            module_logger().warning(
                f"{', '.join(loose)} vary within replicates: shuffling the tested variables "
                "within replicates also breaks their link with these, so the permutation "
                "p-value is approximate."
            )
        return Scheme("within", tuple(tested), replicate, np.zeros(len(data), dtype=np.intp))
    stratum = np.zeros(len(data), dtype=np.intp)
    if strata is not None:
        stratum = pd.factorize(strata, sort=True)[0]
        spanning = pd.Series(stratum).groupby(replicate).nunique() > 1
        if spanning.any():
            raise ValueError(
                f"{int(spanning.sum())} replicates have units in several strata: each "
                "replicate must belong to one stratum."
            )
    per_replicate = [v for v in others if v in constant_within(data, [v], replicate)]
    unequal = [v for v in per_replicate if v not in constant_within(data, [v], stratum)]
    if unequal:
        module_logger().warning(
            f"{', '.join(unequal)} differ between replicates of a stratum: swapping the tested "
            f"variables between them makes the permutation p-value approximate. With "
            f"--strata {unequal[0]}, replicates swap only with others alike."
        )
    return Scheme("between", tuple(tested), replicate, stratum)


@dataclass(frozen=True)
class _Groups:
    """The positions a UTR's permutations shuffle, by group, with their labels."""

    groups: list[npt.NDArray[np.intp]]  # positions (units or replicates) per group
    labels: list[npt.NDArray[np.intp]]  # their label codes
    representative: npt.NDArray[np.intp]  # a unit with each label
    unit_position: npt.NDArray[np.intp]  # each unit's position in the groups' frame
    n_positions: int


def _groups(scheme_: Scheme, rows: npt.NDArray[np.intp], data: pd.DataFrame) -> _Groups:
    """What a UTR's units shuffle: units within replicates, or replicates within strata."""
    values = data.iloc[rows][list(scheme_.variables)]
    labels, _ = pd.factorize(pd.MultiIndex.from_frame(values.astype(object)))
    labels = np.asarray(labels, dtype=np.intp)
    representative = np.zeros(labels.max() + 1, dtype=np.intp)
    representative[labels[::-1]] = np.arange(len(labels))[::-1]
    if scheme_.kind == "within":
        group_of = scheme_.replicate[rows]
        unit_position = np.arange(len(rows))
        position_labels = labels
    else:
        replicates, unit_position = np.unique(scheme_.replicate[rows], return_inverse=True)
        first_unit = np.zeros(len(replicates), dtype=np.intp)
        first_unit[unit_position[::-1]] = np.arange(len(rows))[::-1]
        position_labels = labels[first_unit]
        group_of = scheme_.stratum[rows][first_unit]
    groups = [np.flatnonzero(group_of == group) for group in np.unique(group_of)]
    return _Groups(
        groups,
        [position_labels[group] for group in groups],
        representative,
        np.asarray(unit_position, dtype=np.intp),
        len(position_labels),
    )


def count_permutations(labels: Sequence[npt.NDArray[np.intp]]) -> int:
    """The number of distinct arrangements of the labels within each group."""
    total = 1
    for group in labels:
        arrangements = math.factorial(len(group))
        for repeats in Counter(group.tolist()).values():
            arrangements //= math.factorial(repeats)
        total *= arrangements
    return total


def _arrangements(labels: npt.NDArray[np.intp]) -> list[tuple[int, ...]]:
    """Every distinct order of a multiset of labels."""
    counts = Counter(labels.tolist())
    keys = sorted(counts)
    arrangements: list[tuple[int, ...]] = []

    def extend(prefix: list[int]) -> None:
        if len(prefix) == len(labels):
            arrangements.append(tuple(prefix))
            return
        for key in keys:
            if counts[key]:
                counts[key] -= 1
                prefix.append(key)
                extend(prefix)
                prefix.pop()
                counts[key] += 1

    extend([])
    return arrangements


def _sources(groups: _Groups, position_labels: npt.NDArray[np.intp]) -> npt.NDArray[np.intp]:
    """For each unit, a unit whose tested values it takes, given labels by position."""
    return groups.representative[position_labels[groups.unit_position]]


def all_permutations(groups: _Groups) -> Iterator[npt.NDArray[np.intp]]:
    """Every distinct permutation (the observed one included), as source units."""
    per_group = [_arrangements(labels) for labels in groups.labels]
    for combination in itertools.product(*per_group):
        position_labels = np.empty(groups.n_positions, dtype=np.intp)
        for positions, arrangement in zip(groups.groups, combination, strict=True):
            position_labels[positions] = arrangement
        yield _sources(groups, position_labels)


def random_permutation(groups: _Groups, rng: np.random.Generator) -> npt.NDArray[np.intp]:
    """A random permutation, as source units."""
    position_labels = np.empty(groups.n_positions, dtype=np.intp)
    for positions, labels in zip(groups.groups, groups.labels, strict=True):
        position_labels[positions] = rng.permutation(labels)
    return _sources(groups, position_labels)


@dataclass(frozen=True)
class PermutationResult:
    """The permutation p-value of one UTR.

    Attributes:
        p_value: The p-value.
        permutations: The number of permutations it used.
        exact: Whether all distinct permutations were used.
    """

    p_value: float
    permutations: int
    exact: bool


def permutation_test(
    statistic: Callable[[npt.NDArray[np.intp]], float],
    observed: float,
    scheme_: Scheme,
    rows: npt.NDArray[np.intp],
    data: pd.DataFrame,
    rng: np.random.Generator,
    stages: Sequence[int] = STAGES,
    stop_after: int = STOP_AFTER,
) -> PermutationResult:
    """The permutation p-value of a UTR's statistic.

    Args:
        statistic: The statistic of the counts with permuted tested values:
            it gets, for each unit, the unit whose values it takes.
        observed: The statistic of the data as they are.
        scheme_: How to shuffle.
        rows: The UTR's units, as positions in *data*.
        data: The variables of the models, a row per unit.
        rng: The random numbers of this UTR.
        stages: Totals of random permutations after which to stop if the
            statistic was reached *stop_after* times; with no more distinct
            permutations than the last, all are used instead.
        stop_after: Exceedances that stop the permutations at a stage.

    Returns:
        The p-value and how it was reached.
    """
    groups = _groups(scheme_, rows, data)
    tolerance = 1e-7 * max(1.0, abs(observed))
    total = count_permutations(groups.labels)
    if total <= stages[-1]:
        reached = sum(
            statistic(sources) >= observed - tolerance for sources in all_permutations(groups)
        )
        return PermutationResult(reached / total, total, True)
    reached = 0
    done = 0
    for stage in stages:
        for _ in range(stage - done):
            reached += statistic(random_permutation(groups, rng)) >= observed - tolerance
        done = stage
        if reached >= stop_after:
            break
    return PermutationResult((reached + 1) / (done + 1), done, False)
