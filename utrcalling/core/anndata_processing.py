"""Post-processing utilities for AnnData objects produced by the utrcalling pipeline."""

from __future__ import annotations

import anndata
import numpy as np
import pandas as pd
import scipy.sparse


def site_names(
    var: pd.DataFrame, utr_name_column: str = "utr_name", gene_name_column: str = "gene_name"
) -> pd.Index:
    """Name UTR sites ``<gene>_<n>-<size>``, as utrcalling names all its outputs.

    *gene* is the gene name; genes without one (missing or ``"None"``) use the
    first part of their UTR name, the Ensembl ID. *n* numbers the sites of each
    gene from 1, by UTR name and then size, so ``1`` is the shortest site of the
    gene's first UTR. *size* is the UTR size (for peaks, the peak position).
    The names are unique: *n* is unique within each gene, and a name splits
    back into gene, *n* and size in only one way.

    Args:
        var: Site annotations with *utr_name_column*, ``utr_size`` and, if
            available, *gene_name_column*.
        utr_name_column: Column of *var* with the UTR names.
        gene_name_column: Column of *var* with the gene names.

    Returns:
        The site names, in the order of the rows of *var*.
    """
    utr_names = var[utr_name_column].astype(str).reset_index(drop=True)
    if gene_name_column in var.columns:
        genes = var[gene_name_column].astype(object).reset_index(drop=True)
    else:
        genes = pd.Series([None] * len(var), dtype=object)
    missing = genes.isna() | (genes.astype(str) == "None")
    ensembl_ids = utr_names.str.split(r"[-_]", regex=True).str[0]
    genes = genes.astype(str).where(~missing, ensembl_ids)
    sizes = var["utr_size"].astype(int).reset_index(drop=True)

    in_order = pd.DataFrame({"gene": genes, "utr": utr_names, "size": sizes}).sort_values(
        ["gene", "utr", "size"], kind="stable"
    )
    number = (in_order.groupby("gene", sort=False).cumcount() + 1).sort_index()
    names = genes + "_" + number.astype(str) + "-" + sizes.astype(str)
    return pd.Index(names.to_numpy(), dtype=object)


def generate_utr_site_names_from_gene_name(
    adata: anndata.AnnData,
    utr_name_column: str = "utr_name",
    gene_name_column: str = "gene_name",
) -> anndata.AnnData:
    """Renumber the sites of *adata* with :func:`site_names`.

    utrcalling already names sites ``<gene>_<n>-<size>`` when it writes them.
    Renumbering after filtering steps closes the gaps that removed sites leave
    in *n*. The previous names are kept in ``.var['UTR_site']``.

    Args:
        adata: The AnnData object whose ``var_names`` will be renamed.
        utr_name_column: Column in ``.var`` containing UTR identifiers.
        gene_name_column: Column in ``.var`` containing gene name annotations.

    Returns:
        A copy of *adata* with renamed ``var_names``.
    """
    adata = adata.copy()
    adata.var["UTR_site"] = adata.var_names
    adata.var_names = site_names(adata.var, utr_name_column, gene_name_column)
    adata.var.index.name = None
    return adata


def select_alternative_transcription_events(
    adata: anndata.AnnData,
) -> anndata.AnnData:
    """Select UTRs with at least two alternative transcription sites.

    Removes UTRs that have fewer than 2 alternative transcription sites,
    effectively selecting alternative polyadenylation (APA) or alternative
    transcription initiation (ATI) events.

    Args:
        adata: The AnnData object to filter.

    Returns:
        A filtered copy of *adata* containing only multi-site UTRs.
    """
    unique_df = adata.var.groupby("utr_name", observed=True).nunique()

    return adata[:, adata.var.utr_name.isin(unique_df[unique_df.utr_size > 1].index)].copy()


def select_utrs_with_molecules(adata: anndata.AnnData, minimum_molecules: int) -> anndata.AnnData:
    """Keep the UTRs with at least *minimum_molecules* molecules over all cells.

    Peak calling keeps every molecule of a UTR, so a raw-sizes matrix and its
    peaks keep the same UTRs. Few molecules make sites unsure: Eugenia
    Guerrero Aruffo kept the 5' UTRs with 500 molecules or more of her bulk
    Cap-seq data (master's thesis, 2026).

    Args:
        adata: A count matrix: raw sizes or sites.
        minimum_molecules: The molecules a UTR needs, over all its sites and cells.

    Returns:
        A copy of *adata* with the sites of those UTRs.
    """
    molecules = pd.Series(np.asarray(adata.X.sum(axis=0)).ravel())
    per_utr = molecules.groupby(adata.var["utr_name"].to_numpy(), observed=True).transform("sum")
    return adata[:, (per_utr >= minimum_molecules).to_numpy()].copy()


def counts_to_percentages(adata: anndata.AnnData, utr_name_column: str) -> anndata.AnnData:
    """Convert raw UMI counts to within-UTR percentages.

    For each UTR, divides count values by the row-wise sum so that each
    observation's counts within a UTR sum to 1. Cells with no counts in a
    UTR keep zeros for that UTR's columns.

    Implementation is vectorized: a single sparse matmul computes per-
    (observation, UTR) row sums, then division happens over the count
    matrix's nonzero entries directly. This avoids per-UTR view assignment
    to ``adata.X`` (deprecated under anndata's copy-on-write semantics) and
    drops runtime from O(n_utrs × n_obs × peaks_per_utr) to O(nnz).

    Sparsity is preserved end-to-end: zeros in the input remain zeros in
    the output.

    Args:
        adata: The AnnData object with raw counts in ``.X``.
        utr_name_column: Column in ``.var`` identifying UTR groups.

    Returns:
        A copy of *adata* with ``.X`` replaced by fractional values.
    """
    adata = adata.copy()
    n_vars = adata.n_vars

    utr_names = np.asarray(adata.var[utr_name_column].values)
    _, utr_codes = np.unique(utr_names, return_inverse=True)
    utr_codes = utr_codes.astype(np.int64)
    n_groups = int(utr_codes.max()) + 1 if n_vars > 0 else 0

    # Aggregator (n_vars x n_groups): one nonzero per column placed in its
    # UTR's group column. ``X @ aggregator`` yields per-(obs, UTR) row sums
    # in one matmul, replacing the per-UTR Python loop.
    aggregator = scipy.sparse.csc_matrix(
        (
            np.ones(n_vars, dtype=np.float64),
            (np.arange(n_vars, dtype=np.int64), utr_codes),
        ),
        shape=(n_vars, n_groups),
    )

    is_sparse = scipy.sparse.issparse(adata.X)
    X = adata.X if is_sparse else scipy.sparse.csr_matrix(adata.X)

    row_sums_per_utr = (X @ aggregator).toarray()

    coo = X.tocoo(copy=False)
    denom = row_sums_per_utr[coo.row, utr_codes[coo.col]]
    new_data = np.divide(
        coo.data.astype(np.float64),
        denom,
        out=np.zeros(coo.data.shape, dtype=np.float64),
        where=denom != 0,
    )
    new_X = scipy.sparse.csr_matrix(
        (new_data, (coo.row, coo.col)),
        shape=X.shape,
    )

    adata.X = new_X if is_sparse else new_X.toarray()
    return adata


def compute_utr_size_means(
    adata: anndata.AnnData, utr_name_column: str = "utr_name"
) -> anndata.AnnData:
    """Compute weighted mean UTR sizes per observation and UTR.

    For each (observation, UTR) pair the mean UTR size is calculated as the
    count-weighted average of individual peak sizes.

    Args:
        adata: The AnnData object with counts in ``.X`` and ``utr_size`` in
            ``.var``.
        utr_name_column: Column in ``.var`` identifying UTR groups.

    Returns:
        A new AnnData where each value is the mean UTR size for that
        observation/UTR pair.
    """
    obs_index_name = adata.obs.index.name
    adata.var = adata.var.set_index(utr_name_column)

    var_new = adata.var[~adata.var.index.duplicated(keep="first")].drop("utr_size", axis=1)

    df = adata.to_df()
    df_mult = df.mul(adata.var["utr_size"])

    df = (
        pd.melt(df, ignore_index=False)
        .groupby([obs_index_name, utr_name_column], observed=True)
        .sum()
    )
    df_mult_sum = (
        pd.melt(df_mult, ignore_index=False)
        .groupby([obs_index_name, utr_name_column], observed=True)
        .sum()
    )

    df = (df_mult_sum / df).reset_index()
    df = df.pivot(index=obs_index_name, columns=utr_name_column)["value"].reindex(adata.obs.index)

    # pivot sorts the UTRs: put their annotations in the same order
    adata = anndata.AnnData(df, obs=adata.obs, var=var_new.reindex(df.columns))
    return adata


def concatenate_adata(
    adata_list: list[anndata.AnnData],
) -> anndata.AnnData:
    """Concatenate multiple AnnData objects along the obs axis.

    Uses ``anndata.concat`` with sparse matrices throughout to avoid
    densifying large count matrices.  Observations are outer-joined so that
    all UTR peaks across inputs are represented (missing values filled
    with 0).

    Args:
        adata_list: AnnData objects sharing compatible ``.var`` schemas
            (must contain ``utr_name`` and ``utr_size`` columns).

    Returns:
        A single concatenated AnnData with sparse ``.X``.
    """
    # Build lightweight AnnData wrappers with canonical var_names
    # (utr_name|utr_size) and sparse X to avoid densifying the matrix.
    prepared = []
    var_dfs = []
    for adata in adata_list:
        canonical = (
            # astype(str): h5ad files store utr_name as categorical
            adata.var["utr_name"].astype(str) + "|" + adata.var["utr_size"].astype(int).astype(str)
        ).values
        X = adata.X
        if not scipy.sparse.issparse(X):
            X = scipy.sparse.csr_matrix(X)
        prepared.append(
            anndata.AnnData(
                X=X,
                obs=adata.obs,
                var=pd.DataFrame(index=canonical),
            )
        )
        var_dfs.append(adata.var)

    # Sparse-aware concatenation — keeps matrices sparse throughout
    adata_concat = anndata.concat(prepared, join="outer")
    del prepared

    # Reconstruct var metadata from originals
    all_var = pd.concat(var_dfs, axis=0).drop("utr_size", axis=1).drop_duplicates()
    del var_dfs

    # Parse canonical var_names back into (utr_name, utr_size)
    split = adata_concat.var_names.to_series().str.rsplit("|", n=1, expand=True)
    concat_cols_frame = pd.DataFrame(
        {
            "utr_name": split[0].values,
            "utr_size": split[1].astype(int).values,
        }
    )

    var_new = pd.merge(concat_cols_frame, all_var, on="utr_name", how="left")
    var_new.index = site_names(var_new)

    adata_concat.var = var_new

    return adata_concat


def utr_annotations(var: pd.DataFrame, utr_name_column: str = "utr_name") -> pd.DataFrame:
    """One row per UTR, with the site annotations that are the same for all its sites.

    Columns that differ between the sites of a UTR, such as ``utr_size``, are
    left out. The UTRs come in the order of their first site.

    Args:
        var: Site annotations with *utr_name_column*.
        utr_name_column: Column of *var* with the UTR names.

    Returns:
        The UTR annotations, indexed by UTR name, with ``sites``: the number
        of sites of each UTR.
    """
    grouped = var.groupby(utr_name_column, sort=False, observed=True)
    same = [
        column
        for column in var.columns
        if column != utr_name_column and (grouped[column].nunique(dropna=False) <= 1).all()
    ]
    utrs = grouped[same].first() if same else pd.DataFrame(index=grouped.size().index)
    utrs["sites"] = grouped.size()
    utrs.index = utrs.index.astype(str)
    utrs.index.name = None
    return utrs


def neighbour_graph(adata: anndata.AnnData, key: str) -> scipy.sparse.csr_matrix:
    """The graph of neighbouring cells in ``adata.obsp[key]``, such as scanpy's connectivities.

    Args:
        adata: The count matrix whose cells the graph connects.
        key: The key of the graph in ``adata.obsp``.

    Returns:
        The graph's weights, a row and a column per cell.

    Raises:
        ValueError: If ``adata.obsp`` has no *key*.
    """
    if key not in adata.obsp:
        known = ", ".join(adata.obsp.keys()) or "none"
        raise ValueError(
            f"The matrix has no graph {key!r} in .obsp (it has: {known}). Compute one with "
            "scanpy (sc.pp.neighbors stores obsp['connectivities']) and add it to the matrix."
        )
    return scipy.sparse.csr_matrix(adata.obsp[key], dtype=np.float64)


def smooth_counts(
    counts: scipy.sparse.spmatrix | np.ndarray, graph: scipy.sparse.spmatrix, rounds: int = 2
) -> scipy.sparse.csr_matrix:
    """Average each cell's counts over its neighbours, *rounds* times.

    Each round replaces the counts of a cell with the mean of its neighbours'
    counts, weighted by the graph. A cell without neighbours keeps its own.
    Smoothing shares molecules between similar cells, which steadies the
    measures of single cells (such as the length index) but makes them
    dependent: don't give p-values to correlations over smoothed cells.

    Args:
        counts: Counts, a row per cell.
        graph: Weights between cells, a row and a column per cell, such as
            scanpy's ``obsp['connectivities']``.
        rounds: How many times to average.

    Returns:
        The smoothed counts.

    Raises:
        ValueError: If *rounds* is negative, or the graph does not have a row
            and a column per cell.
    """
    if rounds < 0:
        raise ValueError(f"rounds {rounds}: smooth 0 or more times")
    if graph.shape != (counts.shape[0], counts.shape[0]):
        raise ValueError(
            f"The graph is {graph.shape[0]} x {graph.shape[1]}: it needs a row and a "
            f"column per cell ({counts.shape[0]})"
        )
    graph = scipy.sparse.csr_matrix(graph, dtype=np.float64)
    sums = np.asarray(graph.sum(axis=1)).ravel()
    isolated = sums == 0
    inverse = np.divide(1.0, sums, out=np.zeros_like(sums), where=~isolated)
    weights = scipy.sparse.diags(inverse) @ graph + scipy.sparse.diags(isolated.astype(float))
    smoothed = scipy.sparse.csr_matrix(counts, dtype=np.float64)
    for _ in range(rounds):
        smoothed = scipy.sparse.csr_matrix(weights @ smoothed)
    return smoothed


def length_index(
    adata: anndata.AnnData,
    utr_name_column: str = "utr_name",
    smooth_graph: str | None = None,
    smooth_rounds: int = 2,
) -> anndata.AnnData:
    """The length index and entropy of site use, per cell (or sample) and UTR.

    The length index is the mean scaled rank of a UTR's molecules' sites, from
    0 (all at its shortest site) to 1 (all at its longest); the entropy, from
    0 (one site) to 1 (equal use of all sites). See
    :mod:`utrcalling.core.stats.measures`. Only UTRs with two or more sites
    have them. Cells without molecules in a UTR get NaN.

    Args:
        adata: A count matrix of sites (peaks), cells or samples by sites.
        utr_name_column: Column of ``.var`` with the UTR names.
        smooth_graph: Key of a graph of neighbouring cells in ``.obsp``
            (such as scanpy's ``connectivities``): average the counts over it
            first, see :func:`smooth_counts`.
        smooth_rounds: How many times to average over *smooth_graph*.

    Returns:
        A matrix of cells by UTRs with the length index in ``.X`` and the
        entropy in ``.layers['entropy']``. ``.obs`` adds each cell's mean
        over its UTRs, ``length_index`` and ``entropy``, and how many UTRs it
        has molecules in, ``utrs_measured``. ``.var`` holds the UTRs'
        annotations and ``sites``.

    Raises:
        ValueError: If no UTR has two or more sites.
    """
    from utrcalling.core.stats import measures

    var = adata.var
    ranks = measures.site_ranks(var[utr_name_column].astype(str), var["utr_size"])
    multi_site = ~np.isnan(ranks)
    if not multi_site.any():
        raise ValueError("No UTR has two or more sites: there is no length index to compute.")
    counts = scipy.sparse.csr_matrix(adata.X, dtype=np.float64)[:, multi_site]
    if smooth_graph is not None:
        counts = smooth_counts(counts, neighbour_graph(adata, smooth_graph), smooth_rounds)
    sites = var[multi_site]
    codes, _ = pd.factorize(sites[utr_name_column].astype(str), sort=False)
    utrs = utr_annotations(sites, utr_name_column)
    n_sites, n_utrs = len(codes), len(utrs)
    columns = np.arange(n_sites)
    aggregator = scipy.sparse.csr_matrix(
        (np.ones(n_sites), (columns, codes)), shape=(n_sites, n_utrs)
    )
    ranked = scipy.sparse.csr_matrix((ranks[multi_site], (columns, codes)), shape=(n_sites, n_utrs))

    totals = (counts @ aggregator).toarray()
    measured = totals > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        index = np.where(measured, (counts @ ranked).toarray() / totals, np.nan)
    entries = counts.tocoo()
    shares = entries.data / totals[entries.row, codes[entries.col]]
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(shares > 0, -shares * np.log(shares), 0.0)
    entropy_sums = scipy.sparse.csr_matrix((terms, (entries.row, entries.col)), shape=counts.shape)
    entropy = (entropy_sums @ aggregator).toarray() / np.log(utrs["sites"].to_numpy(float))
    entropy = np.where(measured, entropy, np.nan)

    obs = adata.obs.copy()
    utrs_measured = measured.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        obs["length_index"] = np.nansum(index, axis=1) / utrs_measured
        obs["entropy"] = np.nansum(entropy, axis=1) / utrs_measured
    obs["utrs_measured"] = utrs_measured
    return anndata.AnnData(
        X=index.astype(np.float32),
        obs=obs,
        var=utrs,
        layers={"entropy": entropy.astype(np.float32)},
    )


def _bin_by_quantiles(values: pd.Series, bins: int) -> pd.Series:
    """Number the values 1 to *bins* by their quantiles (fewer bins if values repeat)."""
    numbers = pd.qcut(values, q=bins, labels=False, duplicates="drop")
    return (numbers + 1).astype("Int64")


def pseudobulk(
    adata: anndata.AnnData,
    by: list[str],
    bins: dict[str, int] | None = None,
    min_cells: int = 10,
) -> anndata.AnnData:
    """Sum the counts of cells into units: per sample, or per sample and bin.

    A unit is a group of cells that share their values of the *by* columns
    of ``.obs`` (such as the sample) and, with *bins*, their bin of a numeric
    column (such as pseudotime cut into equal-frequency bins over all cells).
    Units let statistics treat samples, not cells, as the replicates: build
    them within samples, never across.

    Each unit keeps the ``.obs`` columns whose value is the same for all its
    cells (such as the sample's condition or batch), and the mean of the
    other numeric columns (such as the mean pseudotime of a bin). Other
    columns are left out. Cells with a missing value in *by* or *bins* are
    in no unit.

    Args:
        adata: A count matrix, cells by sites (or raw sizes).
        by: Columns of ``.obs`` whose values make the units.
        bins: Numeric columns of ``.obs`` to cut into that many bins of equal
            numbers of cells: each unit is also one bin, numbered from 1 in
            the column ``<column>_bin``.
        min_cells: Units with fewer cells are left out.

    Returns:
        A count matrix of units by sites, named ``<value>|<value>...``, with
        ``n_cells`` in ``.obs``.

    Raises:
        ValueError: If a column is missing or a bins column is not numeric,
            or if no unit has *min_cells* cells.
    """
    from utrcalling.core.module_log import module_logger

    bins = dict(bins or {})
    obs = adata.obs.copy()
    missing = [column for column in [*by, *bins] if column not in obs.columns]
    if missing:
        raise ValueError(
            f"No column {', '.join(missing)} in the cell annotations (.obs): it has "
            f"{', '.join(map(str, obs.columns))}"
        )
    for column, number in bins.items():
        if not pd.api.types.is_numeric_dtype(obs[column]) or pd.api.types.is_bool_dtype(
            obs[column]
        ):
            raise ValueError(f"Column {column} is not numeric: it cannot be cut into bins")
        if number < 1:
            raise ValueError(f"{column}:{number}: cut into 1 or more bins")
        obs[f"{column}_bin"] = _bin_by_quantiles(obs[column], number)
        made = obs[f"{column}_bin"].nunique()
        if made < number:
            module_logger().warning(
                f"{column} has too many equal values for {number} bins: made {made}"
            )
    keys = [*by, *(f"{column}_bin" for column in bins)]
    in_a_unit = obs[keys].notna().all(axis=1).to_numpy()
    if not in_a_unit.all():
        module_logger().info(
            f"Left out {(~in_a_unit).sum()} cells with a missing {' or '.join(keys)}"
        )
    obs = obs[in_a_unit]
    grouped = obs.groupby(keys, sort=True, observed=True, dropna=True)
    unit = grouped.ngroup().to_numpy()
    n_cells = np.bincount(unit)
    kept = n_cells >= min_cells
    if not kept.any():
        raise ValueError(
            f"No unit has {min_cells} cells or more (the largest has {n_cells.max()}): "
            "lower --min-cells, or make larger units"
        )
    if not kept.all():
        module_logger().info(
            f"Left out {(~kept).sum()} of {len(kept)} units with fewer than {min_cells} cells"
        )

    membership = scipy.sparse.csr_matrix(
        (np.ones(len(unit), dtype=np.int64), (unit, np.arange(len(unit)))),
        shape=(len(n_cells), len(unit)),
    )[kept]
    counts = scipy.sparse.csr_matrix(adata.X)[in_a_unit]
    summed = scipy.sparse.csr_matrix(membership @ counts).astype(counts.dtype)

    first = grouped.first()
    units = first.reset_index()[keys]
    left_out = []
    for column in obs.columns.difference(keys, sort=False):
        constant = (grouped[column].nunique(dropna=False) <= 1).to_numpy()
        if constant.all():
            units[column] = first[column].reset_index(drop=True)
        elif pd.api.types.is_numeric_dtype(obs[column]) and not pd.api.types.is_bool_dtype(
            obs[column]
        ):
            units[column] = grouped[column].mean().reset_index(drop=True)
        else:
            left_out.append(str(column))
    if left_out:
        module_logger().info(
            f"Left out the columns that differ within units: {', '.join(left_out)}"
        )
    units["n_cells"] = n_cells
    units = units[kept]
    units.index = pd.Index(
        ["|".join(map(str, values)) for values in units[keys].itertuples(index=False)]
    )
    return anndata.AnnData(X=summed, obs=units, var=adata.var.copy())
