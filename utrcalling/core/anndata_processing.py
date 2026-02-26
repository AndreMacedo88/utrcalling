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
    unique_df = adata.var.groupby("utr_name").nunique()

    return adata[
        :, adata.var.utr_name.isin(unique_df[unique_df.utr_size > 1].index)
    ].copy()


def counts_to_percentages(
    adata: anndata.AnnData, utr_name_column: str
) -> anndata.AnnData:
    """Convert raw UMI counts to within-UTR percentages.

    For each UTR, divides count values by the row-wise sum so that each
    observation's counts within a UTR sum to 1.

    Args:
        adata: The AnnData object with raw counts in ``.X``.
        utr_name_column: Column in ``.var`` identifying UTR groups.

    Returns:
        The same *adata* with ``.X`` replaced by fractional values.
    """
    for utr in tqdm(
        np.unique(adata.var[utr_name_column]), colour="#ffb700", ascii=" ■"
    ):
        mask = adata.var[utr_name_column] == utr
        data_utr = adata[:, mask].X

        if scipy.sparse.issparse(data_utr):
            data_utr = data_utr.toarray()

        row_sums = np.sum(data_utr, axis=1).reshape(-1, 1)
        adata[:, mask].X = np.divide(
            data_utr, row_sums, out=np.zeros_like(data_utr, dtype=float), where=row_sums != 0
        )

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
    adata = anndata.AnnData(df, obs=adata.obs, var=var_new, dtype=df.values.dtype)
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

    Observations are outer-joined so that all UTR peaks across inputs are
    represented (missing values filled with 0).

    Args:
        adata_list: AnnData objects sharing compatible ``.var`` schemas
            (must contain ``utr_name`` and ``utr_size`` columns).

    Returns:
        A single concatenated AnnData.
    """
    # prepare X object
    adata_df_list = [adata.to_df() for adata in adata_list]
    for df, adata in zip(adata_df_list, adata_list):
        df.columns = pd.MultiIndex.from_frame(adata.var[["utr_name", "utr_size"]])

    adata_concat = pd.concat(adata_df_list, join="outer", axis=0, sort=False).fillna(0)

    # prepare var object
    adatas_var = (
        pd.concat([adata.var for adata in adata_list], axis=0)
        .drop("utr_size", axis=1)
        .drop_duplicates()
    )

    concat_cols_frame = adata_concat.columns.to_frame().reset_index(drop=True)
    var_new = pd.merge(concat_cols_frame, adatas_var, on="utr_name", how="left")
    var_new["polyA"] = (
        var_new["utr_name"]
        + "-"
        + (var_new.groupby("utr_name").cumcount() + 1).astype(str)
    )
    var_new = var_new.set_index("polyA")

    # final changes to X
    adata_concat.columns = (
        concat_cols_frame.utr_name
        + "-"
        + (concat_cols_frame.groupby("utr_name").cumcount() + 1).astype(str)
    )

    # prepare obs object
    obs_new = pd.concat([adata.obs for adata in adata_list], axis=0)

    adata_result = anndata.AnnData(X=adata_concat, obs=obs_new, var=var_new)

    return adata_result


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
