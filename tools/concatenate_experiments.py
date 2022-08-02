# import general tools
import sys
import argparse
# import data processing tools
import scanpy as sc

sys.path.append('../')  # allows utrcalling to be imported from `python utrcalling.py run`
sys.path.append('./')  # allows utrcalling to be imported from this script
from main_module import anndata_processing as ap


def concatenate_experiments(adata_paths, out_path):
    adata_list = [sc.read_h5ad(adata_path) for adata_path in adata_paths]
    adata_result = ap.concatenate_adata(adata_list=adata_list)

    adata_result.write_h5ad(out_path)


def main():
    args_all = parser.parse_args()

    args = vars(args_all).copy()
    del args["func"]

    args_all.func(**args)


parser = argparse.ArgumentParser(
    description="""
Subsets an experiment file (h5ad), given a list of annotations to match one 
column of the var or obs.
The annotation files should be a CSV file with one column, named as in the h5ad file.
This column should have the values that will subset the h5ad experiment file.
""",
    epilog="""
Example use:
$ subset-experiments adata.h5ad -a anno_1 anno_2 anno_3 -t var obs var 
    -o adata_s1.h5ad adata_s2.h5ad adata_s3.h5ad
""",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--adata_paths", "-a", nargs="+", type=str, required=True,
                    help="Paths to the input AnnData object files (h5ad format).")
parser.add_argument("--out_path", "-o", type=str, required=True,
                    help="Path to write the result (h5ad format).")

# This allows for assigning different parsers to different functions:
parser.set_defaults(func=concatenate_experiments)

if __name__ == "__main__":
    main()
