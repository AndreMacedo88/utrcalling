# import general purpose tools
from cmath import isnan
import warnings
import re
import joblib
# import data wrangling tools
import numpy as np
import pandas as pd
import anndata
# import stats tools
from statsmodels.nonparametric.bandwidths import bw_silverman
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema
# import visualization tools
from tqdm import tqdm
# import internal tools
from utrcalling.core import module_log, helpers


def generate_utr_site_names_from_gene_name(
        adata, utr_name_column="utr_name", gene_name_column="gene_name"):
    """
    Generates names for each distinct UTR size (UTR site) based on the gene name and UTR 
    number.
    It then changes the var_names variable of the AnnData object with those new names.
    The old names/IDs are kept in .var['UTR_site']

    Args:
        adata (anndata.AnnData): The AnnData object to change the var_names.
        out (str, optional): The path to write the output.
    """

    adata = adata.copy()

    # Keep old var_names in the var layer
    adata.var['UTR_site'] = adata.var_names

    # Append UTR_site number termination to gene names, making them UTR_site names
    new = []
    counter = 1
    utr = ""
    for row in adata.var.itertuples():
        utr_name = getattr(row, utr_name_column)
        gene_name = getattr(row, gene_name_column)

        if utr != utr_name:
            counter = 1
            utr = utr_name

        utr_info = utr.split("_")
        if len(utr_info) > 1:
            if pd.isnull(gene_name):
                name = f"{utr_info[0]}_UTR{utr_info[1]}_site{counter}"
            else:
                name = f"{gene_name}_UTR{utr_info[1]}_site{counter}"
        else:
            if pd.isnull(gene_name):
                name = f"{utr_info[0]}_site{counter}"
            else:
                name = f"{gene_name}_site{counter}"

        new.append(name)
        counter += 1

    adata.var['UTR_site_named'] = new

    # Deal with duplicates in the UTR site names
    dups = adata[:, adata.var.UTR_site_named.duplicated(keep='first')].var.UTR_site_named
    for dup in dups:
        counter = 1
        for row in adata[:, adata.var.UTR_site_named == dup].var.itertuples():
            adata.var.loc[row.Index, 'UTR_site_named'] = f"{dup}_{counter}"
            counter += 1

    # Assign the new UTR site names to the index, and drop the UTR site name column and index name
    adata.var_names = adata.var.UTR_site_named
    adata.var = adata.var.drop(['UTR_site_named'], axis=1)
    adata.var.index.name = None

    return adata


def select_alternative_transcription_events(adata):
    """
    Removes UTRs that have less than 2 alternative transcription sites, effectively 
    selecting alternative poly adenilation (APA) or alternative transcription 
    initiation (ATI) events.

    Args:
        adata (anndata.AnnData): The AnnData object to select APA events.

    Returns:
        anndata.AnnData: The modified AnnData.
    """
    unique_df = adata.var.groupby('utr_name').nunique()

    return adata[:, adata.var.utr_name.isin(
        unique_df[unique_df.utr_size > 1].index)].copy()


def counts_to_percentages(adata, utr_name_column):

    for utr in tqdm(np.unique(adata.var[utr_name_column]), colour="#ffb700", ascii=" ■"):
        data_utr = adata[:, adata.var[utr_name_column] == utr].X

        adata[:, adata.var[utr_name_column] == utr].X = np.divide(
            data_utr, np.sum(data_utr, axis=1).reshape(-1, 1))

    return adata


def compute_cell_means_and_var(adata, utr_name_column="utr_name"):
    def repeat_arr(row, sizes):
        sizes_repeated = np.repeat(sizes, row.astype(int))
        removed_zeros = np.where(sizes_repeated != 0, sizes_repeated, np.nan)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            means = np.nanmean(removed_zeros)

        data_to_norm = removed_zeros[np.isfinite(removed_zeros)]
        if data_to_norm.size > 2:
            norm_data = (data_to_norm - np.min(data_to_norm)) / np.ptp(data_to_norm)
            vars = np.nanvar(norm_data, ddof=1)
        else:
            vars = np.nan
        return [means, vars]

    def compute(utr):
        utr_data = adata[:, adata.var[utr_name_column] == utr]
        sizes = utr_data.var.utr_size.values
        mean_and_var = np.apply_along_axis(
            repeat_arr, 1, utr_data.X, sizes
        )
        number_umis = int(np.sum(utr_data.X))
        return utr, number_umis, mean_and_var

    cell_mean_sizes = pd.DataFrame(index=adata.obs_names)
    cell_var_sizes = pd.DataFrame(index=adata.obs_names)

    # parallelize the calculation of mean UTR site size per cell per UTR
    utrs = adata.var[utr_name_column].unique()
    utr_list, umi_list, mean_and_var_list = zip(*joblib.Parallel(n_jobs=-2)(
        joblib.delayed(compute)(utr=utr) for utr in
        tqdm(utrs, desc='Calculating mean UTR site size per cell per UTR. UTR no: ',
             colour="#ffb700", ascii=" ■")))

    # assign the parallelization results properly
    umis = {utr: umi for (utr, umi) in zip(utr_list, umi_list)}
    for utr, means_and_vars in zip(utr_list, mean_and_var_list):
        cell_mean_sizes[utr] = means_and_vars[:, 0]
        cell_var_sizes[utr] = means_and_vars[:, 1]

    var = adata.var[adata.var[utr_name_column].isin(
        cell_mean_sizes.columns)]
    var = var.drop_duplicates(subset=[utr_name_column])
    var = var.set_index(utr_name_column)
    var = var.drop(labels="utr_size", errors="ignore")
    var['UMIs'] = pd.Series(umis)

    adata = anndata.AnnData(cell_mean_sizes,
                            obs=adata.obs,
                            var=var,
                            dtype=cell_mean_sizes.values.dtype)
    adata.layers['variance'] = cell_var_sizes
    return adata


def subset_adata(adata, annotations_keep, anno_types):
    adata_list = []

    for annotation, anno_type in zip(annotations_keep, anno_types):
        annotation_df = pd.read_csv(annotation, index_col=0)

        if anno_type == "var":
            adata_list.append(
                adata[:, adata.var.index.isin(annotation_df.index)])
        elif anno_type == "obs":
            adata_list.append(
                adata[adata.obs.index.isin(annotation_df.index), :])
        else:
            raise AttributeError(f"Annotation type {anno_type} not supported")

    return adata_list


def concatenate_adata(adata_list):
    # prepare X object
    adata_df_list = [adata.to_df() for adata in adata_list]
    for df, adata in zip(adata_df_list, adata_list):
        df.columns = pd.MultiIndex.from_frame(adata.var[["utr_name", "utr_size"]])

    adata_concat = pd.concat(adata_df_list, join="outer", axis=0, sort=False).fillna(0)

    # prepare var object
    adatas_var = pd.concat([adata.var for adata in adata_list], axis=0).drop(
        "utr_size", axis=1).drop_duplicates()

    concat_cols_frame = adata_concat.columns.to_frame().reset_index(drop=True)
    var_new = pd.merge(concat_cols_frame, adatas_var, on="utr_name", how="left")
    var_new["polyA"] = var_new["utr_name"] + "-" + \
        (var_new.groupby("utr_name").cumcount() + 1).astype(str)
    var_new = var_new.set_index("polyA")

    # final changes to X
    adata_concat.columns = concat_cols_frame.utr_name + "-" + \
        (concat_cols_frame.groupby("utr_name").cumcount() + 1).astype(str)

    # prepare obs object
    obs_new = pd.concat([adata.obs for adata in adata_list], axis=0)

    adata_result = anndata.AnnData(X=adata_concat, obs=obs_new, var=var_new)

    return(adata_result)


def split_train_test_data(adata, train_fraction, seed):
    logger = module_log.module_logger()

    try:
        bins_columns = adata.obs.columns.str.extract(
            r"(.*umi.*.*bin.*)", flags=re.IGNORECASE).dropna().values[0][0]
    except IndexError:
        umis_per_id = np.sum(adata.X, axis=1)
        # Find optimal number of bins
        kde_bandwidth = bw_silverman(umis_per_id)
        a = umis_per_id.reshape(-1, 1)
        kde = KernelDensity(kernel='gaussian', bandwidth=kde_bandwidth).fit(a)
        s = np.linspace(min(a), max(a))
        e = kde.score_samples(s.reshape(-1, 1))
        mi, _ = argrelextrema(e, np.less)[0], argrelextrema(e, np.greater)[0]
        cuts = len(mi) + 1

        logger.info(f"Binning UTRs into {cuts} bins for stratified sampling. "
                    "Bins can be found in .obs['UMI_bins']")

        umi_bins = pd.qcut(umis_per_id, q=cuts, labels=range(1, cuts + 1))
        bins_columns = "UMI_bins"
        adata.obs[bins_columns] = umi_bins

    try:
        batch_columns = adata.obs.columns.str.extract(
            r"(.*batch.*)", flags=re.IGNORECASE).dropna().values[0][0]
    except IndexError:
        batch_columns = False

    obs_train = adata.obs.groupby(
        [batch_columns, bins_columns], group_keys=False
    ).apply(lambda x: x.sample(frac=train_fraction, random_state=seed))

    obs_test = adata.obs.drop(labels=obs_train.index, axis=0)

    adata_train = adata[adata.obs_names.isin(obs_train.index), :]
    adata_test = adata[adata.obs_names.isin(obs_test.index), :]
    return adata_train, adata_test
