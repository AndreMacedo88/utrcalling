# import general tools
import argparse

from utrcalling.core import peak_calling_anndata as pc
from utrcalling.core import helpers


def remove_peaks_below_count_threshold(input_path, out_path, threshold):
    adata = helpers.check_if_path_and_open(input_path, method="h5ad")
    adata_new = pc.remove_peaks_below_count(adata=adata, threshold=threshold)
    adata_new.write_h5ad(out_path)


def create_parser(subparsers):

    help = (
        "Removes the UTR location peaks that have less than `threshold` counts in "
        "the sum of all cells/samples/batches."
    )

    description = (
        f"{help} "
        "This effectively curates the peaks, selecting the most common "
        "and/or more prone to be true peaks."
    )

    parser = subparsers.add_parser(
        name="remove-peaks-below-threshold",
        help=help,
        description=f"DESCRIPTION: {description}",
        epilog="""
    Example use:
    $ utrcalling remove-peaks-below-threshold adata_old.h5ad adata_filtered.h5ad -t 10
    """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("input_path", type=str,
                        help="Path to input AnnData object file (h5ad format).")
    parser.add_argument("out_path", type=str, help="Path where output will be stored.")
    parser.add_argument("-t", "--threshold", type=int,
                        help="Count threshold to filter the data.", default=10)

    parser.set_defaults(func=remove_peaks_below_count_threshold)
