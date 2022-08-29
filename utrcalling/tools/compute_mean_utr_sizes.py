# import general tools
import argparse
# import internal tools
from utrcalling.core import anndata_processing as ap
from utrcalling.core import helpers


def compute_mean_utr_sizes(input_path, out_path, utr_name_column="utr_name"):
    adata = helpers.check_if_path_and_open(input_path, method="h5ad")
    adata = ap.compute_cell_means_and_var(adata=adata, utr_name_column=utr_name_column)
    adata.write_h5ad(out_path)


def main():
    args_all = parser.parse_args()

    args = vars(args_all).copy()
    del args["func"]

    args_all.func(**args)


parser = argparse.ArgumentParser(
    description="""
Computes the variance and mean UTR size per row of the AnnData object provided
""",
    epilog="""
Example use:
$ compute-mean-utr-sizes adata.h5ad adata_means.h5ad -u utr_name
""",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("input_path", type=str,
                    help="Path to input AnnData object file (h5ad format).")
parser.add_argument("out_path", type=str,
                    help="Path where output will be stored.")
parser.add_argument("-u", "--utr_name_column", type=str, default="utr_name",
                    help="""
                    The name of the column in the anndata.var dataframe that 
                    contains the names of the UTRs.
                    """)

# This allows for assigning different parsers to different functions:
parser.set_defaults(func=compute_mean_utr_sizes)

if __name__ == "__main__":
    main()
