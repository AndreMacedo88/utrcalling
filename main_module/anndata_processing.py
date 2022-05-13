from main_module import helpers


def friendly_polya_names(adata, out=''):
    """
    Generates PolyA site names from the gene name and UTR number.
    It then changes the var_names variable of the AnnData object with those new names.

    Args:
        adata (anndata.AnnData): The AnnData object to change the var_names.
        out (str, optional): The path to write the output.
    """

    if not out:
        out = 'with_friendly_pA_names.h5ad'
    out_filepath = helpers.check_file_exists_and_return_unique(out)

    adata = adata.copy()

    # Keep old var_names in the var layer
    adata.var['pA'] = adata.var_names

    # Append pA number termination to gene names, making them pA names
    new = []
    counter = 1
    utr = ""
    for row in adata.var.itertuples():
        if utr != row.utr_name:
            counter = 1
            utr = row.utr_name

        utr_info = utr.split("_")
        if len(utr_info) > 1:
            if row.gene_name == "nan":
                name = f"{utr_info[0]}_UTR{utr_info[1]}_pA{counter}"
            else:
                name = f"{row.gene_name}_UTR{utr_info[1]}_pA{counter}"
        else:
            if row.gene_name == "nan":
                name = f"{utr_info[0]}_pA{counter}"
            else:
                name = f"{row.gene_name}_pA{counter}"

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
