"""Quality control of the pipeline and of the count matrices it makes.

Each function returns a table; the qc commands write them as tab-separated text.
None of them changes a count matrix.
"""

from __future__ import annotations

import gzip
import io
import itertools
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pysam
import scipy.sparse
import scipy.stats

from utrcalling.core import protocol, provenance
from utrcalling.core.utr_mapping import MAPPING_STEPS

if TYPE_CHECKING:
    import anndata
    import numpy.typing as npt

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


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------


def require_counts(adata: anndata.AnnData) -> None:
    """Refuse matrices that do not count molecules (fractions or mean sizes).

    Raises:
        ValueError: If the matrix records that it holds fractions or mean sizes.
    """
    matrix = provenance.recorded(adata).get("matrix")
    if matrix in ("fractions", "mean_sizes"):
        raise ValueError(
            f"This matrix holds {matrix.replace('_', ' ')}: give a count matrix "
            "(raw sizes or peaks) instead."
        )


def _molecules_per_cell_and_utr(
    adata: anndata.AnnData,
) -> tuple[pd.DataFrame, npt.NDArray[np.int64], int]:
    """The molecules of each cell in each UTR, from the sparse entries of ``.X``.

    Returns:
        A table with a row per (cell, UTR) with molecules: ``cell`` and ``utr``
        (positions), ``molecules``, ``size_sum`` (the sum of the molecules' UTR
        sizes) and ``sites`` (with molecules); each site's UTR position; and
        the number of UTRs.
    """
    utr_of_site, utrs = pd.factorize(adata.var["utr_name"].astype(str))
    n_utrs = max(len(utrs), 1)
    counts = scipy.sparse.coo_matrix(adata.X)
    counts.eliminate_zeros()
    sizes = adata.var["utr_size"].to_numpy(dtype=np.float64)

    molecules = counts.data.astype(np.float64)
    keys = counts.row.astype(np.int64) * n_utrs + utr_of_site[counts.col]
    cell_utrs, entry = np.unique(keys, return_inverse=True)
    table = pd.DataFrame(
        {
            "cell": cell_utrs // n_utrs,
            "utr": cell_utrs % n_utrs,
            "molecules": np.bincount(entry, weights=molecules, minlength=len(cell_utrs)),
            "size_sum": np.bincount(
                entry, weights=molecules * sizes[counts.col], minlength=len(cell_utrs)
            ),
            "sites": np.bincount(entry, minlength=len(cell_utrs)),
        }
    )
    return table, utr_of_site.astype(np.int64), n_utrs


def cell_metrics(adata: anndata.AnnData, min_molecules: int = 5) -> pd.DataFrame:
    """How much each cell (or sample) tells about its UTR sizes.

    Args:
        adata: Count matrix of raw sizes or peaks, with ``utr_name`` and
            ``utr_size`` in ``.var``.
        min_molecules: Molecules a cell needs in a UTR to measure the UTR's
            size in that cell alone.

    Returns:
        A row per cell, indexed by ``cell``, with its ``molecules``, the
        ``sites`` and ``utrs`` with molecules, the ``utrs_measured`` (with at
        least *min_molecules*), the ``multi_site_utrs`` (with molecules at two
        or more sites), and ``relative_utr_size``: the mean, over the cell's
        molecules, of each molecule's UTR size divided by its UTR's mean size
        in all cells. Above 1, the cell's UTRs are longer than usual (missing
        for cells without molecules; UTRs whose sizes are all 0, as 5' UTRs
        can be, do not count).

    Raises:
        ValueError: If the matrix holds fractions or mean sizes.
    """
    require_counts(adata)
    per_utr, _, n_utrs = _molecules_per_cell_and_utr(adata)
    n_cells = adata.n_obs

    def per_cell(weights: npt.ArrayLike | None = None) -> npt.NDArray[np.int64]:
        return np.bincount(per_utr.cell, weights=weights, minlength=n_cells).astype(np.int64)

    # Each UTR's mean size over all its molecules.
    utr_molecules = np.bincount(per_utr.utr, weights=per_utr.molecules, minlength=n_utrs)
    utr_sizes = np.bincount(per_utr.utr, weights=per_utr.size_sum, minlength=n_utrs)
    utr_mean = np.divide(utr_sizes, utr_molecules, out=np.zeros(n_utrs), where=utr_molecules > 0)[
        per_utr.utr
    ]
    comparable = utr_mean > 0
    cells_compared = per_utr.cell[comparable]
    relative_sum = np.bincount(
        cells_compared,
        weights=per_utr.size_sum[comparable] / utr_mean[comparable],
        minlength=n_cells,
    )
    relative_n = np.bincount(
        cells_compared, weights=per_utr.molecules[comparable], minlength=n_cells
    )
    return pd.DataFrame(
        {
            "molecules": per_cell(per_utr.molecules),
            "sites": per_cell(per_utr.sites),
            "utrs": per_cell(),
            "utrs_measured": per_cell(per_utr.molecules >= min_molecules),
            "multi_site_utrs": per_cell(per_utr.sites >= 2),
            "relative_utr_size": np.divide(
                relative_sum, relative_n, out=np.full(n_cells, np.nan), where=relative_n > 0
            ),
        },
        index=pd.Index(adata.obs_names, name="cell"),
    )


def depth_correlation(cells: pd.DataFrame) -> tuple[float, int]:
    """Spearman correlation of the cells' relative UTR size with their molecules.

    UTR sizes measured with few molecules can depend on sequencing depth: a
    strong correlation means comparisons of UTR sizes between cells mix depth
    with biology.

    Args:
        cells: The table of :func:`cell_metrics`.

    Returns:
        The correlation (missing with fewer than 3 cells with a relative UTR
        size) and the number of those cells.
    """
    measured = cells.dropna(subset=["relative_utr_size"])
    if len(measured) < 3:
        return float("nan"), len(measured)
    rho = scipy.stats.spearmanr(measured.relative_utr_size, measured.molecules)[0]
    return float(rho), len(measured)


# ---------------------------------------------------------------------------
# Batches
# ---------------------------------------------------------------------------


def batch_consistency(
    adata: anndata.AnnData, groupby: str, min_molecules: int = 10
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """How alike the site usage of groups of cells (batches, samples, replicates) is.

    Sums the molecules of each group's cells (pseudobulk). A site's usage is
    its share of its UTR's molecules in the group.

    Args:
        adata: Count matrix of raw sizes or peaks, with ``utr_name`` in ``.var``.
        groupby: Column of ``.obs`` with each cell's group. Cells without one
            are left out.
        min_molecules: Molecules a UTR needs in both groups to compare them.

    Returns:
        Two tables. The groups: a row per group, indexed by ``group``, with
        its ``cells``, ``molecules`` and ``sites`` with molecules, and the
        ``sites_only_here`` (in no other group) with their ``molecules_only_here``.
        The pairs of groups: a row per pair, indexed by ``group_1`` and
        ``group_2``, with the ``utrs`` compared (two or more sites, and at
        least *min_molecules* in both groups), the ``correlation`` (Pearson) of
        their sites' usage, and the ``usage_difference``: the share of a UTR's
        molecules that sit at other sites in one group than in the other, on
        average over the UTRs (half the summed absolute difference of usage).

    Raises:
        KeyError: If ``.obs`` has no column *groupby*.
        ValueError: If the matrix holds fractions or mean sizes.
    """
    require_counts(adata)
    if groupby not in adata.obs.columns:
        raise KeyError(
            f"No column {groupby!r} in the cell annotations: "
            f"choose one of {', '.join(map(str, adata.obs.columns))}"
        )
    labels = adata.obs[groupby]
    has_group = labels.notna().to_numpy()
    group_of_cell, groups = pd.factorize(labels[has_group].astype(str), sort=True)
    membership = scipy.sparse.csr_matrix(
        (np.ones(len(group_of_cell)), (group_of_cell, np.flatnonzero(has_group))),
        shape=(len(groups), adata.n_obs),
    )
    site_molecules = np.asarray((membership @ scipy.sparse.csr_matrix(adata.X)).todense())

    present = site_molecules > 0
    only_here = present & (present.sum(axis=0) == 1)
    group_table = pd.DataFrame(
        {
            "cells": np.bincount(group_of_cell, minlength=len(groups)),
            "molecules": site_molecules.sum(axis=1),
            "sites": present.sum(axis=1),
            "sites_only_here": only_here.sum(axis=1),
            "molecules_only_here": (site_molecules * only_here).sum(axis=1),
        },
        index=pd.Index(groups, name="group"),
    ).astype(np.int64)

    utr_of_site, utrs = pd.factorize(adata.var["utr_name"].astype(str))
    utr_totals = np.zeros((len(groups), len(utrs)))
    np.add.at(utr_totals.T, utr_of_site, site_molecules.T)
    usage = np.divide(
        site_molecules,
        utr_totals[:, utr_of_site],
        out=np.zeros_like(site_molecules, dtype=np.float64),
        where=utr_totals[:, utr_of_site] > 0,
    )
    multi_site = np.bincount(utr_of_site, minlength=len(utrs)) >= 2

    rows = []
    for first, second in itertools.combinations(range(len(groups)), 2):
        compared = multi_site & (utr_totals[[first, second]] >= min_molecules).all(axis=0)
        sites = compared[utr_of_site]
        difference = np.zeros(len(utrs))
        np.add.at(difference, utr_of_site, np.abs(usage[first] - usage[second]))
        if sites.sum() >= 3:
            correlation = scipy.stats.pearsonr(usage[first, sites], usage[second, sites])[0]
        else:
            correlation = np.nan
        rows.append(
            {
                "group_1": groups[first],
                "group_2": groups[second],
                "utrs": int(compared.sum()),
                "correlation": float(correlation),
                "usage_difference": difference[compared].mean() / 2 if compared.any() else np.nan,
            }
        )
    columns = ["group_1", "group_2", "utrs", "correlation", "usage_difference"]
    pair_table = pd.DataFrame(rows, columns=columns).set_index(["group_1", "group_2"])
    return group_table, pair_table


# ---------------------------------------------------------------------------
# Known sites
# ---------------------------------------------------------------------------


def site_positions(var: pd.DataFrame, utr_end: str) -> npt.NDArray[np.int64]:
    """The genome position (1-based) of each site.

    UTR sizes count from the UTR's start in transcript orientation for 3'
    protocols (``utr_end`` ``three_prime``) and from its end for 5' protocols
    (``five_prime``), as utr_mapping measures them.

    Args:
        var: Site annotations with ``utr_size`` and the UTR's ``start``,
            ``end`` (1-based, inclusive) and ``strand``.
        utr_end: ``three_prime`` or ``five_prime``.

    Returns:
        The positions, in the order of *var*.

    Raises:
        ValueError: If *utr_end* is neither, or the UTR coordinates are not
            numbers, as in gene-level matrices.
    """
    if utr_end not in ("three_prime", "five_prime"):
        raise ValueError(f"utr_end {utr_end!r}: expected three_prime or five_prime")
    for column in ("start", "end"):
        if not pd.api.types.is_integer_dtype(var[column]):
            raise ValueError(
                "The sites have no genome positions: their UTR coordinates are not "
                "numbers (gene-level matrices join the coordinates of a gene's UTRs)."
            )
    size = var["utr_size"].to_numpy(dtype=np.int64)
    start = var["start"].to_numpy(dtype=np.int64)
    end = var["end"].to_numpy(dtype=np.int64)
    plus = (var["strand"].astype(str) == "+").to_numpy()
    if utr_end == "three_prime":
        return np.where(plus, start + size, end - size)
    return np.where(plus, end - size + 1, start + size - 1)


def read_known_sites(path: str | Path) -> pd.DataFrame:
    """Known polyadenylation or transcription start sites, from a BED file.

    Args:
        path: BED file (plain or gzipped) with chromosome, start (0-based) and
            end, and optionally name, score and strand. Lines starting with
            ``#``, ``track`` or ``browser`` are skipped.

    Returns:
        A table with ``chrom``, ``start``, ``end`` and ``strand`` (``.`` when
        the file has none).
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as bed_file:
        lines = [line for line in bed_file if not line.startswith(("#", "track", "browser"))]
    bed = pd.read_csv(io.StringIO("".join(lines)), sep="\t", header=None, dtype={0: str})
    known = pd.DataFrame(
        {
            "chrom": bed[0].to_numpy(),
            "start": bed[1].astype(np.int64).to_numpy(),
            "end": bed[2].astype(np.int64).to_numpy(),
            # Lines without a strand count on both strands.
            "strand": bed[5].fillna(".").astype(str).to_numpy() if bed.shape[1] >= 6 else ".",
        }
    )
    return known


def _chromosome(name: object) -> str:
    """Chromosome names compared without a leading "chr" (chr1 and 1 match)."""
    text = str(name)
    return text[3:] if text.lower().startswith("chr") else text


def _distance_to_nearest(
    positions: npt.NDArray[np.int64], starts: npt.NDArray[np.int64], ends: npt.NDArray[np.int64]
) -> npt.NDArray[np.float64]:
    """Distance from 0-based positions to the nearest interval [start, end): 0 inside one."""
    order = np.argsort(starts, kind="stable")
    starts, ends = starts[order], ends[order]
    reach = np.maximum.accumulate(ends)  # furthest end among the intervals so far
    before = np.searchsorted(starts, positions, side="right")
    distance = np.full(len(positions), np.inf)
    has_before = before > 0
    last_end = reach[np.maximum(before - 1, 0)]
    inside = has_before & (last_end > positions)
    distance[has_before] = positions[has_before] - (last_end[has_before] - 1)
    distance[inside] = 0
    has_after = before < len(starts)
    after = starts[np.minimum(before, len(starts) - 1)] - positions
    distance[has_after & ~inside] = np.minimum(distance, after)[has_after & ~inside]
    return distance


def known_site_distances(
    adata: anndata.AnnData,
    known_sites: pd.DataFrame,
    utr_end: str | None = None,
    window: int = 20,
) -> pd.DataFrame:
    """Each site's distance to the nearest known site on its strand.

    Args:
        adata: Count matrix at UTR level, with ``utr_name``, ``utr_size``,
            ``chrom``, ``start``, ``end`` and ``strand`` in ``.var``.
        known_sites: The table of :func:`read_known_sites`. Known sites without
            a strand count on both strands.
        utr_end: Where the UTR sizes count from (see :func:`site_positions`).
            By default, what the matrix records.
        window: Distance (nt) up to which a site counts as near a known site.

    Returns:
        A row per site, indexed by ``site``, with ``gene_name`` (if ``.var``
        has it), ``utr_name``, ``utr_size``, ``chrom``, ``position`` (1-based),
        ``strand``, ``molecules`` (in all cells; for fractions, their sum),
        ``known_site_distance`` (nt; missing without a known site on the
        chromosome and strand) and ``near_known_site``.

    Raises:
        ValueError: If neither *utr_end* nor the matrix tells where the sizes
            count from, if the matrix is at gene level, or if no chromosome of
            the matrix is among the known sites'.
    """
    var = adata.var
    positions = site_positions(var, _positioned_utr_end(adata, utr_end))
    chroms = var["chrom"].map(_chromosome).astype(str).to_numpy()
    strands = var["strand"].astype(str).to_numpy()
    known_chroms = known_sites["chrom"].map(_chromosome).astype(str).to_numpy()
    if not set(chroms) & set(known_chroms):
        raise ValueError(
            "No chromosome of the matrix is among the known sites': check that both "
            "come from the same genome assembly."
        )

    distance = np.full(adata.n_vars, np.inf)
    for (chrom, strand), sites in pd.Series(range(adata.n_vars)).groupby([chroms, strands]):
        on_strand = (known_chroms == chrom) & np.isin(known_sites["strand"], [strand, "."])
        if on_strand.any():
            distance[sites.to_numpy()] = _distance_to_nearest(
                positions[sites.to_numpy()] - 1,
                known_sites["start"].to_numpy()[on_strand],
                known_sites["end"].to_numpy()[on_strand],
            )

    table = _site_table(adata, positions)
    found = np.isfinite(distance)
    known_distance = pd.array(np.where(found, distance, 0).astype(np.int64), dtype="Int64")
    known_distance[~found] = pd.NA
    table["known_site_distance"] = known_distance
    table["near_known_site"] = found & (distance <= window)
    return table


def _positioned_utr_end(adata: anndata.AnnData, utr_end: str | None) -> str:
    """Which end of the UTR the sizes count from: *utr_end*, or what the matrix records.

    Raises:
        ValueError: If neither tells, or if the matrix is at gene level.
    """
    record = provenance.recorded(adata)
    utr_end = utr_end or record.get("utr_end")
    if utr_end is None:
        raise ValueError(
            "The matrix does not record which end of the UTR its sizes count from "
            "(it predates utrcalling 5.0.0): give it, three_prime or five_prime."
        )
    if record.get("summarization_level") == "gene":
        raise ValueError(
            "Gene-level sites have no genome positions: their sizes are positions "
            "along the gene's joined UTRs. Use a UTR-level matrix (-s utr)."
        )
    return utr_end


def _site_table(adata: anndata.AnnData, positions: npt.NDArray[np.int64]) -> pd.DataFrame:
    """A row per site, indexed by ``site``: where it lies and its molecules in all cells."""
    var = adata.var
    table = var[[c for c in ("gene_name", "utr_name", "utr_size") if c in var]].copy()
    table["chrom"] = var["chrom"].astype(str).to_numpy()
    table["position"] = positions
    table["strand"] = var["strand"].astype(str).to_numpy()
    table["molecules"] = np.asarray(adata.X.sum(axis=0)).ravel()
    table.index.name = "site"
    return table


# ---------------------------------------------------------------------------
# Priming artefacts
# ---------------------------------------------------------------------------

# Internal priming (3' protocols): the oligo-dT primer binds A's that the genome
# encodes right after the site, not the poly(A) tail. The rule for ESTs of
# Gautheret et al. 1998 (Genome Res 8:524): 6 A's in a row, or 7 A's in 10 nt.
A_RUN = 6
A_IN_10_NT = 7
# Strand invasion (5' protocols): the template-switching oligo (TSO) binds inside
# the RNA, where it resembles the TSO's 3' end, not at the cap. The rule of Tang
# et al. 2013 (Nucleic Acids Res 41:e44): the 9 nt before the site within an
# edit distance of the TSO's last 9 nt, and 2 of the 3 nt before the site G's.
TSO_END = 9
G_BEFORE_SITE = 2


def priming_artefacts(
    adata: anndata.AnnData,
    genome: str | Path,
    utr_end: str | None = None,
    *,
    window: int = 20,
    tso: str | None = None,
    max_edit_distance: int = 4,
) -> pd.DataFrame:
    """Flag the sites that the genome around them explains as priming artefacts.

    For 3' protocols, internal priming: a site is flagged when the genome's
    *window* nt after it hold A_RUN A's in a row, or A_IN_10_NT A's in 10 nt.
    For 5' protocols, strand invasion: a site is flagged when the TSO_END nt
    before it are within *max_edit_distance* (substitutions, insertions and
    deletions) of the TSO's last TSO_END nt, and G_BEFORE_SITE of the 3 nt
    before it are G's. Sequences are read on the site's strand.

    A flag says that the genome could explain the site as an artefact, not
    that it is one. It needs sites at the molecules' ends, which the standard
    and bulk protocols do not give: their reads stop short of the ends.

    Args:
        adata: Count matrix at UTR level, with ``utr_name``, ``utr_size``,
            ``chrom``, ``start``, ``end`` and ``strand`` in ``.var``.
        genome: Genome FASTA (plain or bgzip-compressed; indexed if it has no
            ``.fai``) of the annotation. Chromosomes match with or without "chr".
        utr_end: Where the UTR sizes count from (see :func:`site_positions`).
            By default, what the matrix records.
        window: How many nt after a 3' site to look for A's.
        tso: The TSO, or its 3' end, for 5' protocols: random bases (such as a
            UMI) as N, which match any base, and riboguanosines as G. By
            default, the TSO of the protocol that the matrix records.
        max_edit_distance: The edit distance up to which the genome before a
            5' site resembles the TSO.

    Returns:
        The columns of :func:`known_site_distances` up to ``molecules``, then,
        for 3' protocols, ``downstream`` (the genome after the site),
        ``longest_a_run``, ``most_a_in_10_nt`` and ``internal_priming``; for
        5' protocols, ``upstream`` (the genome before the site),
        ``tso_edit_distance``, ``g_before_site`` (G's among its last 3 nt) and
        ``strand_invasion``. These are missing for sites on chromosomes that
        the genome lacks.

    Raises:
        ValueError: If neither *utr_end* nor the matrix tells where the sizes
            count from, if the matrix is at gene level, if no chromosome of the
            matrix is in the genome, if *window* is below 1, or if a 5'
            matrix has no TSO (given or recorded) or the TSO is not DNA.
    """
    utr_end = _positioned_utr_end(adata, utr_end)
    if utr_end == "three_prime" and window < 1:
        raise ValueError(f"window {window}: look at 1 nt after each site or more")
    tso_end = _tso_end(adata, tso) if utr_end == "five_prime" else ""
    positions = site_positions(adata.var, utr_end)
    table = _site_table(adata, positions)
    length = window if utr_end == "three_prime" else TSO_END
    sequences, found = _flanking_sequences(
        genome, table, after=utr_end == "three_prime", length=length
    )
    codes = np.frombuffer("".join(sequences).encode(), dtype=np.uint8).reshape(len(table), length)

    if utr_end == "three_prime":
        is_a = codes == ord("A")
        run = np.zeros(len(table), dtype=np.int64)
        longest = np.zeros(len(table), dtype=np.int64)
        for column in is_a.T:
            run = (run + 1) * column
            longest = np.maximum(longest, run)
        span = min(10, length)
        so_far = np.concatenate([np.zeros((len(table), 1), dtype=np.int64), is_a.cumsum(axis=1)], 1)
        most = (so_far[:, span:] - so_far[:, :-span]).max(axis=1)
        columns = {
            "downstream": sequences,
            "longest_a_run": longest,
            "most_a_in_10_nt": most,
            "internal_priming": (longest >= A_RUN) | (most >= A_IN_10_NT),
        }
    else:
        distance = _edit_distances(codes[:, length - len(tso_end) :], tso_end)
        g_before = (codes[:, -3:] == ord("G")).sum(axis=1)
        columns = {
            "upstream": sequences,
            "tso_edit_distance": distance,
            "g_before_site": g_before,
            "strand_invasion": (distance <= max_edit_distance) & (g_before >= G_BEFORE_SITE),
        }
    for name, values in columns.items():
        dtype = {"U": "string", "b": "boolean"}.get(np.asarray(values).dtype.kind, "Int64")
        column = pd.array(values, dtype=dtype)
        column[~found] = pd.NA
        table[name] = column
    return table


def _tso_end(adata: anndata.AnnData, tso: str | None) -> str:
    """The last TSO_END nt of *tso*, or of the TSO of the matrix's protocol.

    Raises:
        ValueError: If there is no TSO, or it holds other letters than A, C, G, T and N.
    """
    if tso is None:
        record = provenance.recorded(adata)
        if "protocol_platform" in record and "protocol_version" in record:
            protocol_ = protocol.protocol_class(
                str(record["protocol_platform"]), int(record["protocol_version"])
            )
            tso = protocol_.template_switching_oligo
    if tso is None:
        raise ValueError(
            "Strand invasion needs the template-switching oligo (TSO): the matrix "
            "records no protocol that has one. Give it (or its 3' end)."
        )
    tso = tso.upper()
    if not tso or set(tso) - set("ACGTN"):
        raise ValueError(f"TSO {tso!r}: write it with A, C, G, T and N (random bases)")
    return tso[-TSO_END:]


def _flanking_sequences(
    genome: str | Path, sites: pd.DataFrame, *, after: bool, length: int
) -> tuple[npt.NDArray[np.str_], npt.NDArray[np.bool_]]:
    """The genome's *length* nt after (or before) each site, on the site's strand.

    Args:
        genome: Genome FASTA.
        sites: Table with ``chrom``, ``position`` (1-based) and ``strand``.
        after: Whether to read the nt after the site in transcript orientation
            (or those before it).
        length: How many nt to read.

    Returns:
        The sequences in upper case, with N for other letters and beyond the
        chromosome's ends (all N on chromosomes that the genome lacks), and
        whether the genome has each site's chromosome.

    Raises:
        ValueError: If the genome has no chromosome of the sites.
    """
    complement = str.maketrans("ACGT", "TGCA")
    with pysam.FastaFile(str(genome)) as fasta:
        names = {
            chrom: next(
                (name for name in (chrom, f"chr{chrom}", _chromosome(chrom)) if name in fasta),
                None,
            )
            for chrom in sites["chrom"].unique()
        }
        if all(name is None for name in names.values()):
            raise ValueError(
                "No chromosome of the matrix is in the genome: check that both come "
                "from the same genome assembly."
            )
        sequences = []
        for chrom, position, strand in sites[["chrom", "position", "strand"]].itertuples(
            index=False
        ):
            name = names[chrom]
            if name is None:
                sequences.append("N" * length)
                continue
            # 0-based: the nt after a + strand site start at its 1-based position.
            reads_forward = (strand == "-") != after
            start = position if reads_forward else position - length - 1
            end = start + length
            size = fasta.get_reference_length(name)
            sequence = fasta.fetch(name, max(start, 0), min(end, size)).upper()
            sequence = "N" * max(-start, 0) + sequence + "N" * max(end - size, 0)
            sequence = re.sub("[^ACGT]", "N", sequence)
            if strand == "-":
                sequence = sequence.translate(complement)[::-1]
            sequences.append(sequence)
    found = sites["chrom"].map(names).notna().to_numpy()
    return np.array(sequences, dtype=f"<U{length}"), found


def _edit_distances(sequences: npt.NDArray[np.uint8], pattern: str) -> npt.NDArray[np.int64]:
    """The edit distance of each row of *sequences* (ASCII codes) to *pattern*.

    The Levenshtein distance: substitutions, insertions and deletions count 1
    each. N in *pattern* matches any base.
    """
    n_rows, n_columns = sequences.shape
    previous = np.tile(np.arange(len(pattern) + 1), (n_rows, 1))
    for i in range(n_columns):
        current = np.empty_like(previous)
        current[:, 0] = i + 1
        for j, base in enumerate(pattern, start=1):
            mismatch = (sequences[:, i] != ord(base)) & (base != "N")
            current[:, j] = np.minimum(
                np.minimum(previous[:, j], current[:, j - 1]) + 1, previous[:, j - 1] + mismatch
            )
        previous = current
    return previous[:, -1]


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


# The columns of the qc report, in order; whole numbers among them come first.
REPORT_COUNTS = (
    "steps",
    "bandwidth",
    "cells",
    "sites",
    "utrs",
    "genes",
    "molecules",
    "multi_site_utrs",
    "mapping_alignments",
    "mapping_reads",
    "mapping_molecules",
    "mapping_molecules_counted",
)
REPORT_COLUMNS = (
    "matrix",
    "version",
    "steps",
    "bandwidth",
    "cells",
    "sites",
    "utrs",
    "genes",
    "molecules",
    "median_molecules_per_cell",
    "median_sites_per_cell",
    "median_utrs_measured_per_cell",
    "multi_site_utrs",
    "molecules_in_multi_site_utrs",
    "depth_correlation",
    "median_values_per_cell",
    "mapping_alignments",
    "mapping_reads",
    "mapping_molecules",
    "mapping_molecules_counted",
)


def report_table(reports: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    """The qc report: a row per matrix, from :func:`matrix_report`.

    Args:
        reports: Each matrix's report, by its file name.

    Returns:
        A row per matrix, indexed by ``file``, with the columns of
        REPORT_COLUMNS that any matrix has, whole numbers as such (missing
        where a matrix has none).
    """
    table = pd.DataFrame.from_dict(
        {name: dict(row) for name, row in reports.items()}, orient="index"
    )
    table = table[[column for column in REPORT_COLUMNS if column in table]]
    counts = [column for column in REPORT_COUNTS if column in table]
    table[counts] = table[counts].astype("Int64")
    table.index.name = "file"
    return table


def matrix_report(adata: anndata.AnnData, min_molecules: int = 5) -> dict[str, Any]:
    """A summary of a count matrix, from any step: one row of the qc report.

    Args:
        adata: A matrix that utrcalling wrote: raw sizes, peaks, fractions or
            mean sizes.
        min_molecules: Molecules a cell needs in a UTR to measure its size
            there (see :func:`cell_metrics`).

    Returns:
        What the matrix holds and the version and steps that made it; its
        cells, sites, UTRs and genes. For counts: the molecules, the median
        molecules, sites and measured UTRs per cell, the multi-site UTRs and
        the share of molecules in them, and the depth correlation (see
        :func:`depth_correlation`). For fractions and mean sizes: the median
        values per cell. With a mapping summary: its total alignments, reads,
        molecules and molecules counted, prefixed ``mapping_``. These describe
        the run that made the matrix, so after filtering its molecules are
        fewer than ``mapping_molecules_counted``.
    """
    record = provenance.recorded(adata)
    matrix = record.get("matrix", "unknown")
    var = adata.var
    report: dict[str, Any] = {
        "matrix": matrix,
        "version": record.get("version", "unknown"),
        "steps": len(record.get("commands", [])),
        "cells": adata.n_obs,
    }
    if matrix == "mean_sizes":  # a column per UTR
        report["utrs"] = adata.n_vars
    else:
        report["sites"] = adata.n_vars
        report["utrs"] = int(var["utr_name"].nunique())
    if "gene_name" in var:
        report["genes"] = int(var["gene_name"].nunique())
    if "bandwidth" in record:
        report["bandwidth"] = record["bandwidth"]

    if matrix in ("fractions", "mean_sizes"):
        # A mean size is missing where the cell has no molecules; a fraction is 0.
        if scipy.sparse.issparse(adata.X):
            values = np.diff(scipy.sparse.csr_matrix(adata.X).indptr)
        elif matrix == "mean_sizes":
            values = np.isfinite(adata.X).sum(axis=1)
        else:
            values = (np.asarray(adata.X) != 0).sum(axis=1)
        report["median_values_per_cell"] = float(np.median(values))
    else:
        cells = cell_metrics(adata, min_molecules=min_molecules)
        sites_per_utr = var.groupby("utr_name", observed=True).size()
        multi_site = var["utr_name"].map(sites_per_utr >= 2).to_numpy(dtype=bool)
        site_molecules = np.asarray(adata.X.sum(axis=0)).ravel()
        molecules = site_molecules.sum()
        report |= {
            "molecules": int(molecules),
            "median_molecules_per_cell": float(cells.molecules.median()),
            "median_sites_per_cell": float(cells.sites.median()),
            "median_utrs_measured_per_cell": float(cells.utrs_measured.median()),
            "multi_site_utrs": int((sites_per_utr >= 2).sum()),
            "molecules_in_multi_site_utrs": (
                float(site_molecules[multi_site].sum() / molecules) if molecules else np.nan
            ),
            "depth_correlation": depth_correlation(cells)[0],
        }

    summary = record.get("mapping_summary")
    if summary is not None:
        total = summary.loc["total"]
        for step in ("alignments", "reads", "molecules", "molecules_counted"):
            if step in total and not pd.isna(total[step]):
                report[f"mapping_{step}"] = int(total[step])
    return report
