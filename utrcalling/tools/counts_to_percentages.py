# import general tools
import argparse
# import internal tools
from utrcalling.core import anndata_processing as ap
from utrcalling.core import helpers


def counts_to_percentages(input_path, out_path, utr_name_column="utr_name"):
    adata = helpers.check_if_path_and_open(input_path, method="h5ad")
    adata = ap.counts_to_percentages(adata=adata, utr_name_column=utr_name_column)
    adata.write_h5ad(out_path)


def main():
    args_all = parser.parse_args()

    args = vars(args_all).copy()
    del args["func"]

    args_all.func(**args)


parser = argparse.ArgumentParser(
    description="""
Converts the UMI counts per polyA site into percentage polyA site usage in relation
to the total sum of UMIs in the UTR.
""",
    epilog="""
Example use:
$ counts-to-percentages adata.h5ad adata_percentages -u utr_name
""",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("input_path", type=str,
                    help="Path to input AnnData object file (h5ad format).")
parser.add_argument("out_path", type=str,
                    help="""
                    Path where output will be stored. '_bw' plus the bandwidth used
                    will be appended to out_path.
                    """)
parser.add_argument("-u", "--utr_name_column", type=str, default="utr_name",
                    help="""
                    The name of the column in the anndata.var dataframe that 
                    contains the names of the UTRs.
                    """)

# This allows for assigning different parsers to different functions:
parser.set_defaults(func=counts_to_percentages)

if __name__ == "__main__":
    main()
