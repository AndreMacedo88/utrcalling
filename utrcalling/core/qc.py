"""Quality control of the pipeline and of the count matrices it makes.

Each function returns a table; the qc commands write them as tab-separated text.
None of them changes a count matrix.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from utrcalling.core.utr_mapping import MAPPING_STEPS

# The molecules left after each step that follows the mapping, over all inputs:
# merging the samples (and, at gene level, the UTRs of each gene), removing
# UTR sizes below 2 (3' protocols), keeping the cells of the annotation file,
# and the molecules in the count matrix.
COUNTING_STEPS = (
    "molecules_after_merge",
    "molecules_after_size_filter",
    "molecules_in_cell_list",
    "molecules_counted",
)


def mapping_summary(
    inputs: Sequence[str],
    mapped: Sequence[Mapping[str, int]],
    counting: Mapping[str, int],
) -> pd.DataFrame:
    """The reads and molecules left after each step of the pipeline.

    Args:
        inputs: The input files: BAM files, or stored read mappings.
        mapped: For each input, its counts at the steps of
            ``utr_mapping.MAPPING_STEPS`` that are known (at least
            ``molecules``).
        counting: The molecules left after the steps of COUNTING_STEPS that
            are done, over all inputs.

    Returns:
        A row per input and a last row, ``total``, indexed by ``input``. The
        columns are the known steps, in the order of the pipeline. The total
        sums the mapping steps that every input has; the counting steps exist
        only for the total.
    """
    rows = [dict(steps) for steps in mapped]
    shared = [step for step in MAPPING_STEPS if all(step in row for row in rows)]
    total = {step: sum(row[step] for row in rows) for step in shared} | dict(counting)
    known = {step for row in [*rows, total] for step in row}
    columns = [step for step in (*MAPPING_STEPS, *COUNTING_STEPS) if step in known]
    index = pd.Index([*map(str, inputs), "total"], name="input")
    return pd.DataFrame([*rows, total], index=index, columns=columns).astype("Int64")


def join_mapping_summaries(summaries: Sequence[pd.DataFrame | None]) -> pd.DataFrame | None:
    """The mapping summary of experiments joined into one count matrix.

    Args:
        summaries: Each experiment's mapping summary (see :func:`mapping_summary`).

    Returns:
        The inputs of all experiments, and a total that sums theirs (missing for
        a step that some experiment lacks); None if an experiment has no summary.
    """
    present = [summary for summary in summaries if summary is not None]
    if len(present) < len(summaries):
        return None
    joined = pd.concat([summary.drop(index="total") for summary in present])
    totals = pd.concat([summary.loc[["total"]] for summary in present])
    joined.loc["total"] = totals.sum(skipna=False)
    columns = [step for step in (*MAPPING_STEPS, *COUNTING_STEPS) if step in joined]
    return joined[columns].astype("Int64")
