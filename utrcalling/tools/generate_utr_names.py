# import general tools
import argparse
# import internal tools
from utrcalling.core import anndata_processing as ap
from utrcalling.core import helpers


def generate_names(input_path, out_path, utr_name_column, gene_name_column):
    adata = helpers.check_if_path_and_open(input_path, method="h5ad")
    adata = ap.generate_utr_site_names_from_gene_name(
        adata=adata,
        utr_name_column=utr_name_column,
        gene_name_column=gene_name_column
    )

    # Write the data to disk
    out_filepath = helpers.check_file_exists_and_return_unique(out_path)
    adata.write_h5ad(out_filepath)


def main():
    args_all = parser.parse_args()

    args = vars(args_all).copy()
    del args["func"]

    args_all.func(**args)


parser = argparse.ArgumentParser(
    description="""
Generates names for each distinct UTR size based on the gene name and UTR number.
Then, changes the var_names of an AnnData object to contain the gene name.
New name format is as follows:<gene>_<utr number>_<UTR size number>. UTR number
is only present in genes with more than one annotated UTR.
""",
    epilog="""
Example use:
$ generate-names adata_old.h5ad adata_new_names.h5ad
""",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("input_path", type=str,
                    help="Path to input AnnData object file (h5ad format).")

parser.add_argument("out_path", type=str, help="Path where output will be stored.")

parser.add_argument("--utr_name_column", "-u", type=str, help="""
    Name of the column in the AnnData object that contains the UTR name.
    """, default="utr_name")

parser.add_argument("--gene_name_column", "-g", type=str, help="""
    Name of the column in the AnnData object that contains the gene name.
    """, default="gene_name")

# This allows for assigning different parsers to different functions:
parser.set_defaults(func=generate_names)

if __name__ == "__main__":
    main()
