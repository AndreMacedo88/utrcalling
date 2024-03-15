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


def create_parser(subparsers):

    help = "Concatenates 2 or more files containing raw 3'UTR sizes per UTR."

    description = f"{help} "

    parser = subparsers.add_parser(
        name="concatenate-experiments",
        help=help,
        description=f"DESCRIPTION: {description}",
        epilog="""
    Example use:
    $ utrcalling concatenate-experiments -i adata_1.h5ad adata_2.h5ad adata_3.h5ad 
    -o adata_concat.h5ad
    """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input_paths",
        "-i",
        nargs="+",
        type=str,
        required=True,
        help="Paths to the input AnnData object files (h5ad format).",
    )
    parser.add_argument(
        "--out_path",
        "-o",
        type=str,
        required=True,
        help="Path to write the result (h5ad format).",
    )

    parser.set_defaults(func=concatenate_experiments)
