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
