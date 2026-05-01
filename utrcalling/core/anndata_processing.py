"""Post-processing utilities for AnnData objects produced by the utrcalling pipeline."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import anndata
import scipy.sparse

from tqdm import tqdm

from utrcalling.core import module_log


def generate_utr_site_names_from_gene_name(
    adata: anndata.AnnData,
    utr_name_column: str = "utr_name",
    gene_name_column: str = "gene_name",
) -> anndata.AnnData:
    """Generate human-readable UTR site names from gene names.

    Creates names in the format ``<gene_name>_<utr_number>-<utr_size>`` for
    each row in ``.var``, then replaces ``var_names`` with those new names.
    The original names/IDs are preserved in ``.var['UTR_site']``.

    When *gene_name* is missing (``NaN`` or ``"None"``), the first component
    of *utr_name* (split on ``-`` or ``_``) is used as fallback.  The UTR
    number is a 1-based sequential counter within each resolved gene name,
    preserving the original row order.

    Args:
        adata: The AnnData object whose ``var_names`` will be renamed.
        utr_name_column: Column in ``.var`` containing UTR identifiers.
        gene_name_column: Column in ``.var`` containing gene name annotations.

    Returns:
        A copy of *adata* with renamed ``var_names``.
    """
    adata = adata.copy()

    # Preserve original var_names
    adata.var["UTR_site"] = adata.var_names

    # Resolve gene names: use gene_name if present, else extract prefix from utr_name
    gene_col = adata.var[gene_name_column]
    missing = gene_col.isna() | (gene_col.astype(str) == "None")
    resolved = gene_col.astype(object)
    resolved[missing] = (
        adata.var.loc[missing, utr_name_column]
        .str.split(r"[-_]", regex=True)
        .str[0]
    )

    # Sequential numbering within each resolved gene (1-based), preserving row order
    utr_number = resolved.groupby(resolved, sort=False).cumcount() + 1

    # Build names: <gene_name>_<utr_number>-<utr_size>
    utr_size_str = adata.var["utr_size"].astype(int).astype(str)
    new_names = resolved + "_" + utr_number.astype(str) + "-" + utr_size_str

    # Handle any remaining duplicates (rare edge case)
    dup_mask = new_names.duplicated(keep=False)
    if dup_mask.any():
        suffix = new_names[dup_mask].groupby(new_names[dup_mask], sort=False).cumcount() + 1
        new_names[dup_mask] = new_names[dup_mask] + "_" + suffix.astype(str)

    adata.var_names = new_names.values
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

    return adata[
        :, adata.var.utr_name.isin(unique_df[unique_df.utr_size > 1].index)
    ].copy()


def counts_to_percentages(
    adata: anndata.AnnData, utr_name_column: str
) -> anndata.AnnData:
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
    n_obs, n_vars = adata.X.shape

    utr_names = np.asarray(adata.var[utr_name_column].values)
    _, utr_codes = np.unique(utr_names, return_inverse=True)
    utr_codes = utr_codes.astype(np.int64)
    n_groups = int(utr_codes.max()) + 1 if n_vars > 0 else 0

    # Aggregator (n_vars × n_groups): one nonzero per column placed in its
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

    var_new = adata.var[~adata.var.index.duplicated(keep="first")].drop(
        "utr_size", axis=1
    )

    df = adata.to_df()
    df_mult = df.mul(adata.var["utr_size"])

    df = (
        pd.melt(df, ignore_index=False).groupby([obs_index_name, utr_name_column]).sum()
    )
    df_mult_sum = (
        pd.melt(df_mult, ignore_index=False)
        .groupby([obs_index_name, utr_name_column])
        .sum()
    )

    df = (df_mult_sum / df).reset_index()
    df = df.pivot(index=obs_index_name, columns=utr_name_column)["value"].reindex(
        adata.obs.index
    )

    if not (df.columns == var_new.index).all():
        var_new = var_new.reindex(df.columns)
    adata = anndata.AnnData(df, obs=adata.obs, var=var_new)
    return adata


def subset_adata(
    adata: anndata.AnnData,
    annotations_keep: list[str],
    anno_types: list[str],
) -> list[anndata.AnnData]:
    """Subset an AnnData by external annotation CSV files.

    Each annotation file is loaded and used to filter either the ``obs`` or
    ``var`` axis of *adata*.

    Args:
        adata: The AnnData object to subset.
        annotations_keep: Paths to CSV files whose index values define the
            subset to keep.
        anno_types: Parallel list of ``"var"`` or ``"obs"`` indicating which
            axis each annotation file refers to.

    Returns:
        A list of AnnData views, one per annotation file.

    Raises:
        ValueError: If an *anno_type* is neither ``"var"`` nor ``"obs"``.
    """
    adata_list = []

    for annotation, anno_type in zip(annotations_keep, anno_types):
        annotation_df = pd.read_csv(annotation, index_col=0)

        if anno_type == "var":
            adata_list.append(adata[:, adata.var.index.isin(annotation_df.index)])
        elif anno_type == "obs":
            adata_list.append(adata[adata.obs.index.isin(annotation_df.index), :])
        else:
            raise ValueError(f"Annotation type {anno_type} not supported")

    return adata_list


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
            adata.var["utr_name"]
            + "|"
            + adata.var["utr_size"].astype(int).astype(str)
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
    all_var = (
        pd.concat(var_dfs, axis=0)
        .drop("utr_size", axis=1)
        .drop_duplicates()
    )
    del var_dfs

    # Parse canonical var_names back into (utr_name, utr_size)
    split = adata_concat.var_names.to_series().str.rsplit("|", n=1, expand=True)
    concat_cols_frame = pd.DataFrame({
        "utr_name": split[0].values,
        "utr_size": split[1].astype(int).values,
    })

    var_new = pd.merge(concat_cols_frame, all_var, on="utr_name", how="left")
    var_new["polyA"] = (
        var_new["utr_name"]
        + "-"
        + (var_new.groupby("utr_name").cumcount() + 1).astype(str)
    )
    var_new = var_new.set_index("polyA")

    adata_concat.var = var_new

    return adata_concat


def split_train_test_data(
    adata: anndata.AnnData, train_fraction: float, seed: int | None
) -> tuple[anndata.AnnData, anndata.AnnData]:
    """Split an AnnData into stratified train and test sets.

    Stratification is performed on UMI-count bins (auto-detected or derived
    via KDE) and, when present, a batch column.

    Args:
        adata: The AnnData object to split.
        train_fraction: Fraction of observations to assign to the training set
            (e.g. ``0.8``).
        seed: Random seed for reproducible sampling, or ``None``.

    Returns:
        A ``(train, test)`` tuple of AnnData views.
    """
    logger = module_log.module_logger()

    try:
        bins_columns = (
            adata.obs.columns.str.extract(r"(.*umi.*.*bin.*)", flags=re.IGNORECASE)
            .dropna()
            .values[0][0]
        )
    except IndexError:
        from statsmodels.nonparametric.bandwidths import bw_silverman
        from sklearn.neighbors import KernelDensity
        from scipy.signal import argrelextrema

        umis_per_id = np.sum(adata.X, axis=1)
        # Find optimal number of bins
        kde_bandwidth = bw_silverman(umis_per_id)
        a = umis_per_id.reshape(-1, 1)
        kde = KernelDensity(kernel="gaussian", bandwidth=kde_bandwidth).fit(a)
        s = np.linspace(min(a), max(a))
        e = kde.score_samples(s.reshape(-1, 1))
        mi, _ = argrelextrema(e, np.less)[0], argrelextrema(e, np.greater)[0]
        cuts = len(mi) + 1

        logger.info(
            f"Binning UTRs into {cuts} bins for stratified sampling. "
            "Bins can be found in .obs['UMI_bins']"
        )

        umi_bins = pd.qcut(umis_per_id, q=cuts, labels=range(1, cuts + 1))
        bins_columns = "UMI_bins"
        adata.obs[bins_columns] = umi_bins

    try:
        batch_columns = (
            adata.obs.columns.str.extract(r"(.*batch.*)", flags=re.IGNORECASE)
            .dropna()
            .values[0][0]
        )
    except IndexError:
        batch_columns = None

    groupby_cols = [c for c in [batch_columns, bins_columns] if c is not None]
    obs_train = adata.obs.groupby(
        groupby_cols, group_keys=False
    ).apply(lambda x: x.sample(frac=train_fraction, random_state=seed))

    obs_test = adata.obs.drop(labels=obs_train.index, axis=0)

    adata_train = adata[adata.obs_names.isin(obs_train.index), :]
    adata_test = adata[adata.obs_names.isin(obs_test.index), :]
    return adata_train, adata_test
