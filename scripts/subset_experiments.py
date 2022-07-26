# import general tools
import sys
import argparse
# import data processing tools
import scanpy as sc

sys.path.append('../')  # allows utrcalling to be imported from `python utrcalling.py run`
sys.path.append('./')  # allows utrcalling to be imported from this script
from main_module import anndata_processing as ap


def subset_experiments(adata_path, annotations_keep, anno_types, out_paths):
    assert len(out_paths) == len(
        annotations_keep), f"Number of annotations does not match number of paths to write the output"

    adata = sc.read_h5ad(adata_path)
    adata_list = ap.subset_adata(
        adata=adata, annotations_keep=annotations_keep, anno_types=anno_types)

    for adata_new, out_path in zip(adata_list, out_paths):
        adata_new.write_h5ad(out_path)


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

parser.add_argument("adata_path", type=str,
                    help="Path to input AnnData object file (h5ad format).")
parser.add_argument("--annotations_keep", "-a", nargs="+", type=str, required=True,
                    help="Paths to the annotation files that describe what to subset.")
parser.add_argument("--anno_types", "-t", nargs="+", type=str, required=True,
                    help="Type of the annotation to use for subsetting.")
parser.add_argument("--out_paths", "-o", nargs="+", type=str, required=True,
                    help="Paths to write the subsets.")

# This allows for assigning different parsers to different functions:
parser.set_defaults(func=subset_experiments)

if __name__ == "__main__":
    main()
