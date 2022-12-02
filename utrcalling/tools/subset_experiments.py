# import general tools
import argparse
# import internal tools
from utrcalling.core import anndata_processing as ap
from utrcalling.core import helpers


def subset_experiments(input_path, annotations_keep, anno_types, out_paths):
    assert len(out_paths) == len(
        annotations_keep), "Number of annotations does not match number of paths to "
    "write the output"

    adata = helpers.check_if_path_and_open(input_path, method="h5ad")
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
The annotation files' first column will be used to subset the h5ad experiment file.
Make sure that this column values match the experiment file annotation index.
""",
    epilog="""
Example use:
$ subset-experiments adata.h5ad -a anno_1 anno_2 anno_3 -t var obs var 
    -o adata_s1.h5ad adata_s2.h5ad adata_s3.h5ad
""",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("input_path", type=str,
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
