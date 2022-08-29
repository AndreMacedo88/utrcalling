# import general tools
import argparse
# import internal tools
from utrcalling.core import anndata_processing as ap
from utrcalling.core import helpers


def concatenate_experiments(input_paths, out_path):
    adata_list = [
        helpers.check_if_path_and_open(path_, method="h5ad") for path_ in input_paths
    ]
    adata_result = ap.concatenate_adata(adata_list=adata_list)
    adata_result.write_h5ad(out_path)


def main():
    args_all = parser.parse_args()

    args = vars(args_all).copy()
    del args["func"]

    args_all.func(**args)


parser = argparse.ArgumentParser(
    description="""
Concatenates 2 or more files containing raw 3'UTR sizes per UTR.
""",
    epilog="""
Example use:
$ concatenate-experiments -i adata_1.h5ad adata_2.h5ad adata_3.h5ad -o adata_concat.h5ad
""",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--input_paths", "-i", nargs="+", type=str, required=True,
                    help="Paths to the input AnnData object files (h5ad format).")
parser.add_argument("--out_path", "-o", type=str, required=True,
                    help="Path to write the result (h5ad format).")

# This allows for assigning different parsers to different functions:
parser.set_defaults(func=concatenate_experiments)

if __name__ == "__main__":
    main()
