import warnings
import numpy as np
import pandas as pd
import joblib
import anndata
from tqdm import tqdm
from utrcalling.core import helpers


def generate_friendly_utr_size_names(
        adata, out='', utr_name_column="utr_name", gene_name_column="gene_name"):
    """
    Generates names for each distinct UTR size based on the gene name and UTR number.
    It then changes the var_names variable of the AnnData object with those new names.

    Args:
        adata (anndata.AnnData): The AnnData object to change the var_names.
        out (str, optional): The path to write the output.
    """

    if not out:
        out = 'with_friendly_UTR_size_names.h5ad'
    out_filepath = helpers.check_file_exists_and_return_unique(out)

    adata = adata.copy()

    # Keep old var_names in the var layer
    adata.var['pA'] = adata.var_names

    # Append pA number termination to gene names, making them pA names
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
            if gene_name == "nan":
                name = f"{utr_info[0]}_UTR{utr_info[1]}_pA{counter}"
            else:
                name = f"{gene_name}_UTR{utr_info[1]}_pA{counter}"
        else:
            if gene_name == "nan":
                name = f"{utr_info[0]}_pA{counter}"
            else:
                name = f"{gene_name}_pA{counter}"

        new.append(name)
        counter += 1

    adata.var['pA_named'] = new

    # Deal with duplicates in the pA names
    dups = adata[:, adata.var.pA_named.duplicated(keep='first')].var.pA_named
    for dup in dups:
        counter = 1
        for row in adata[:, adata.var.pA_named == dup].var.itertuples():
            adata.var.loc[row.Index, 'pA_named'] = f"{dup}_{counter}"
            counter += 1

    # Assign the new pA names to the index, and drop the pA name column and index name
    adata.var_names = adata.var.pA_named
    adata.var = adata.var.drop(['pA_named'], axis=1)
    adata.var.index.name = None

    # Write the data to disk
    adata.write_h5ad(out_filepath)
    return


def potential_apa_events(adata):
    """
    Removes UTRs that have less than 2 pA sites, effectively selecting for alternative
    poly adenilation (APA) events.

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

    # parallelize the calculation of mean pA size per cell per UTR
    utrs = adata.var[utr_name_column].unique()
    utr_list, umi_list, mean_and_var_list = zip(*joblib.Parallel(n_jobs=-2)(
        joblib.delayed(compute)(utr=utr) for utr in
        tqdm(utrs, desc='Calculating mean pA usage per cell per UTR. UTR no: ',
             colour="#ffb700", ascii=" ■")))

    # assign the parallelization results properly
    umis = {utr: umi for (utr, umi) in zip(utr_list, umi_list)}
    for utr, means_and_vars in zip(utr_list, mean_and_var_list):
        cell_mean_sizes[utr] = means_and_vars[:, 0]
        cell_var_sizes[utr] = means_and_vars[:, 1]

    var = adata.var[adata.var[utr_name_column].isin(
        cell_mean_sizes.columns)].iloc[:, 1:-1]
    var = var.drop_duplicates(subset=[utr_name_column])
    var = var.set_index(utr_name_column)
    var['UMIs'] = pd.Series(umis)

    adata = anndata.AnnData(cell_mean_sizes, obs=adata.obs, var=var)
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
