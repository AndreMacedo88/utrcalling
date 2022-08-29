# import general tools
import argparse
# import data processing tools
import scanpy as sc

from utrcalling.core import peak_calling_anndata as pc
from utrcalling.core import helpers


def remove_peaks_below_count_threshold(input_path, out_path, threshold):
    adata = helpers.check_if_path_and_open(input_path, method="h5ad")
    adata_new = pc.remove_peaks_below_count(adata=adata, threshold=threshold)
    adata_new.write_h5ad(out_path)


def main():
    args_all = parser.parse_args()

    args = vars(args_all).copy()
    del args["func"]

    args_all.func(**args)


parser = argparse.ArgumentParser(
    description="""
Removes the UTR location peaks that have less than `threshold` counts in
the sum of all cells/samples/batches. This effectively curates the peaks, selecting the most common
and/or more prone to be true peaks.
""",
    epilog="""
Example use:
$ remove-peaks-below-count-threshold adata_old.h5ad adata_filtered.h5ad -t 10
""",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("input_path", type=str,
                    help="Path to input AnnData object file (h5ad format).")
parser.add_argument("out_path", type=str, help="Path where output will be stored.")
parser.add_argument("-t", "--threshold", type=int,
                    help="Count threshold to filter the data.", default=10)


# This allows for assigning different parsers to different functions:
parser.set_defaults(func=remove_peaks_below_count_threshold)

if __name__ == "__main__":
    main()
