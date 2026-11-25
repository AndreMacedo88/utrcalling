"""Model formulas, read into the design matrices of a test.

A test compares two models of how the units (cells, or samples) use a UTR's
sites, written as R-style formulas of columns of the cell annotations: the
*full* model, such as ``~ batch + bs(pseudotime, df=5)``, and the *reduced*
model nested in it, such as ``~ batch``. The variables that the full model
adds are the *tested* variables. formulaic reads the formulas, with R's
syntax: ``+``, ``:``, ``*``, ``C()`` for categories, ``bs()`` for splines and
so on (https://matthewwardrop.github.io/formulaic/).

A formula's transforms remember what they learnt from the data: a spline
keeps the knots it placed on all units. Evaluating it on other values, such
as permuted ones, or the points where effect sizes are read, then gives the
same functions, not new ones.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import scipy.linalg

if TYPE_CHECKING:
    import numpy.typing as npt
    from formulaic.model_spec import ModelSpec

# Relative size of the pivots of a QR decomposition below which a column
# counts as a combination of the others (as in R's qr, tol = 1e-7).
RANK_TOLERANCE = 1e-7
# Points along a numeric variable where effect sizes are read.
GRID_POINTS = 50
# Numeric variables with this many distinct whole numbers or fewer, outside
# C(), may be categories coded as numbers (such as batch numbers).
FEW_NUMBERS = 10


def _formula_text(formula: str, which: str) -> str:
    """Check that *formula* is an R-style model formula without a response."""
    response, tilde, terms = formula.partition("~")
    if not tilde or response.strip():
        raise ValueError(
            f"The {which} model {formula!r} must be written as ~ terms, as in "
            "'~ batch + condition', without a response: the site counts are the response."
        )
    return terms.strip() or "1"


def _required_variables(formula: str, which: str) -> set[str]:
    """The columns of the data that *formula* needs."""
    from formulaic import Formula
    from formulaic.errors import FormulaicError

    try:
        return set(Formula(formula).required_variables)
    except FormulaicError as error:
        raise ValueError(f"Could not read the {which} model {formula!r}: {error}") from error


def independent_columns(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.intp]:
    """The positions of a largest set of linearly independent columns of *matrix*.

    Columns that are combinations of others (such as a batch with no units
    in a subset of the rows) are left out, by a pivoted QR decomposition.

    Args:
        matrix: A design matrix.

    Returns:
        The positions of the columns to keep, in their original order.
    """
    if matrix.size == 0:
        return np.arange(0)
    _, triangle, pivots = scipy.linalg.qr(matrix, mode="economic", pivoting=True)
    diagonal = np.abs(np.diag(triangle))
    if diagonal[0] == 0:
        return np.arange(0)
    rank = int((diagonal > RANK_TOLERANCE * diagonal[0]).sum())
    return np.sort(pivots[:rank])


def matrix_rank(matrix: npt.NDArray[np.float64]) -> int:
    """The rank of *matrix*, with the tolerance of :func:`independent_columns`."""
    return len(independent_columns(matrix))


@dataclass(frozen=True)
class Contrast:
    """Where the effect sizes of a variable are read: two values, or a grid between them.

    Attributes:
        variable: The column of the annotations whose effect is measured.
        kind: ``numeric`` (read along a grid from *start* to *end*) or
            ``categorical`` (read at two levels).
        points: The values the variable is set to: the grid, or the
            reference level and the other level.
        start: The value effects start from (the low percentile, or the reference level).
        end: The value effects go to (the high percentile, or the other level).
    """

    variable: str
    kind: Literal["numeric", "categorical"]
    points: tuple[Any, ...]
    start: Any
    end: Any


@dataclass(frozen=True)
class Design:
    """The full and reduced models of a test, evaluated on its units.

    Attributes:
        full_formula: The full model.
        reduced_formula: The reduced model, nested in the full one.
        data: The variables of the models for each unit in them (units with
            a missing value are left out), indexed like the units.
        kept: For each unit of the annotations, whether it is in the models.
        full: The full design matrix, a row per unit of *data*.
        reduced: The reduced design matrix.
        full_columns: The names of the columns of *full*.
        reduced_columns: The names of the columns of *reduced*.
        tested: The variables that the full model adds.
        reduced_variables: The variables of the reduced model.
        categorical: The variables that the full model treats as categories.
        full_spec: formulaic's model spec of the full model, which evaluates
            it on other data with the same knots and levels.
    """

    full_formula: str
    reduced_formula: str
    data: pd.DataFrame
    kept: npt.NDArray[np.bool]
    full: npt.NDArray[np.float64]
    reduced: npt.NDArray[np.float64]
    full_columns: tuple[str, ...]
    reduced_columns: tuple[str, ...]
    tested: tuple[str, ...]
    reduced_variables: tuple[str, ...]
    categorical: frozenset[str]
    full_spec: ModelSpec = field(repr=False)

    def evaluate_full(self, data: pd.DataFrame) -> npt.NDArray[np.float64]:
        """The full design matrix of other data: permuted values, or effect points.

        Args:
            data: The full model's variables, a row per unit.

        Returns:
            The design matrix, with the knots and levels learnt from all units.
        """
        return np.asarray(self.full_spec.get_model_matrix(data, na_action="raise"), float)

    def contrast(
        self,
        variable: str | None = None,
        effect_range: tuple[float, float] = (10, 90),
        levels: tuple[Any, Any] | None = None,
    ) -> Contrast | None:
        """Where to read the effect sizes of a variable of the full model.

        Args:
            variable: The variable, by default the one tested variable.
            effect_range: For a numeric variable, the percentiles of its values
                (over all units) between which effects are read.
            levels: For a categorical variable, the level to compare and the
                reference level, compared as text; by default the last level
                and the first (the model's reference).

        Returns:
            The contrast, or None if the full model adds no variable (as when
            it adds an interaction of variables in the reduced model) and none
            is given.

        Raises:
            ValueError: If the model adds several variables and none is given,
                if the variable is not in the full model, if the range is not
                two percentiles from low to high, or if a level is unknown.
        """
        if variable is None:
            if not self.tested:
                return None
            if len(self.tested) > 1:
                raise ValueError(
                    f"The full model adds several variables ({', '.join(self.tested)}): "
                    "choose the one to measure effects of, with --effect-of."
                )
            variable = self.tested[0]
        if variable not in self.data.columns:
            raise ValueError(
                f"{variable} is not a variable of the full model ({', '.join(self.data.columns)})"
            )
        values = self.data[variable]
        if variable in self.categorical:
            categories = _levels(values)
            if levels is None:
                reference, level = categories[0], categories[-1]
            else:
                by_text = {str(category): category for category in categories}
                unknown = [str(text) for text in levels if str(text) not in by_text]
                if unknown:
                    raise ValueError(
                        f"{variable} has no level {', '.join(unknown)}: it has "
                        f"{', '.join(map(str, categories))}"
                    )
                level, reference = (by_text[str(text)] for text in levels)
            if level == reference:
                raise ValueError(
                    f"Compare two different levels of {variable}, not {level} with itself"
                )
            return Contrast(variable, "categorical", (reference, level), reference, level)
        low, high = effect_range
        if not 0 <= low < high <= 100:
            raise ValueError(
                f"Effect range {low} {high}: give two percentiles from 0 to 100, low then high"
            )
        start, end = np.percentile(values.to_numpy(float), [low, high])
        grid = tuple(np.linspace(start, end, GRID_POINTS))
        return Contrast(variable, "numeric", grid, float(start), float(end))

    def at_points(self, rows: npt.NDArray[np.intp], contrast: Contrast) -> npt.NDArray[np.float64]:
        """The full design of some units with the contrast's variable set to each point.

        Args:
            rows: Positions of the units in :attr:`data`.
            contrast: Where effects are read.

        Returns:
            An array of points x units x columns.
        """
        units = self.data.iloc[rows]
        stacked = pd.concat(
            [_with_value(units, contrast.variable, point) for point in contrast.points],
            ignore_index=True,
        )
        matrix = self.evaluate_full(stacked)
        return matrix.reshape(len(contrast.points), len(rows), matrix.shape[1])


def _levels(values: pd.Series) -> list[Any]:
    """The levels of a categorical variable, in the order the model codes them."""
    if isinstance(values.dtype, pd.CategoricalDtype):
        return [level for level in values.cat.categories if (values == level).any()]
    return sorted(values.unique())


def _with_value(units: pd.DataFrame, column: str, value: object) -> pd.DataFrame:
    """*units*, with *column* set to *value* in every row (keeping its categories)."""
    changed = units.copy()
    original = units[column]
    if isinstance(original.dtype, pd.CategoricalDtype):
        changed[column] = pd.Categorical([value] * len(units), categories=original.cat.categories)
    else:
        changed[column] = pd.Series([value] * len(units), index=units.index, dtype=original.dtype)
    return changed


def _is_categorical(variable: str, values: pd.Series, spec: ModelSpec) -> bool:
    """Whether the model treats *variable* as categories (text, booleans, or in C())."""
    if not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
        return True
    pattern = re.compile(rf"(?<![\w.]){re.escape(variable)}(?![\w])")
    return any(
        state[0].value == "categorical" and pattern.search(str(factor))
        for factor, state in spec.encoder_state.items()
    )


def build(obs: pd.DataFrame, full: str, reduced: str = "~ 1") -> Design:
    """Read the full and reduced models, and evaluate them on the units.

    Units with a missing value in a variable of the models are left out, such
    as cells without a pseudotime outside the lineage that has one.

    Args:
        obs: The units' annotations (``.obs`` of a count matrix).
        full: The full model, as in ``~ batch + bs(pseudotime, df=5)``.
        reduced: The reduced model, nested in the full one, as in ``~ batch``.

    Returns:
        The design of the test.

    Raises:
        ValueError: If a formula cannot be read or has a response, if a
            variable is not a column of *obs*, if no unit has all variables,
            if the reduced model is not nested in the full one, or if the full
            model adds nothing to it.
    """
    from formulaic import model_matrix
    from formulaic.errors import FormulaicError

    from utrcalling.core.module_log import module_logger

    full_terms = _formula_text(full, "full")
    reduced_terms = _formula_text(reduced, "reduced")
    full_variables = _required_variables(f"~ {full_terms}", "full")
    reduced_variables = _required_variables(f"~ {reduced_terms}", "reduced")
    missing = sorted((full_variables | reduced_variables) - set(obs.columns))
    if missing:
        raise ValueError(
            f"The models use {', '.join(missing)}, which the cell annotations (.obs) "
            f"lack: they have {', '.join(map(str, obs.columns))}"
        )
    variables = sorted(full_variables | reduced_variables)
    data = obs[variables]
    kept = data.notna().all(axis=1).to_numpy()
    if not kept.any():
        raise ValueError(f"No unit has a value for all of {', '.join(variables)}")
    if not kept.all():
        module_logger().info(
            f"Left out {(~kept).sum()} of {len(kept)} units with a missing value in "
            f"{', '.join(variables)}"
        )
    data = data[kept]
    try:
        full_matrix = model_matrix(f"~ {full_terms}", data, na_action="raise")
        reduced_matrix = model_matrix(f"~ {reduced_terms}", data, na_action="raise")
    except FormulaicError as error:
        raise ValueError(f"Could not evaluate the models: {error}") from error
    full_array = np.asarray(full_matrix, float)
    reduced_array = np.asarray(reduced_matrix, float)
    full_rank = matrix_rank(full_array)
    if matrix_rank(np.hstack([full_array, reduced_array])) > full_rank:
        raise ValueError(
            f"The reduced model ({reduced}) is not nested in the full model ({full}): "
            "the full model must contain all its terms."
        )
    if full_rank <= matrix_rank(reduced_array):
        raise ValueError(
            f"The full model ({full}) adds nothing to the reduced one ({reduced}) on these "
            "units: are the tested variables confounded with the others?"
        )
    if full_rank < full_array.shape[1]:
        module_logger().warning(
            f"{full_array.shape[1] - full_rank} columns of the full model are combinations "
            "of others: each UTR's fit leaves them out"
        )
    spec = full_matrix.model_spec
    for variable in variables:
        values = data[variable]
        coded = (
            pd.api.types.is_numeric_dtype(values)
            and not pd.api.types.is_bool_dtype(values)
            and values.nunique() <= FEW_NUMBERS
            and bool((values == values.round()).all())
            and not _is_categorical(variable, values, spec)
            and not _is_categorical(variable, values, reduced_matrix.model_spec)
        )
        if coded:
            module_logger().warning(
                f"{variable} holds {values.nunique()} whole numbers: the models treat it as a "
                f"number (a slope). Write C({variable}) if its values are categories, such "
                "as batches."
            )
    return Design(
        full_formula=full,
        reduced_formula=reduced,
        data=data,
        kept=kept,
        full=full_array,
        reduced=reduced_array,
        full_columns=tuple(map(str, full_matrix.columns)),
        reduced_columns=tuple(map(str, reduced_matrix.columns)),
        tested=tuple(sorted(full_variables - reduced_variables)),
        reduced_variables=tuple(sorted(reduced_variables)),
        categorical=frozenset(
            variable
            for variable in full_variables
            if _is_categorical(variable, data[variable], spec)
        ),
        full_spec=spec,
    )
