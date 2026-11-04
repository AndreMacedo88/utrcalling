# import general tools
from __future__ import annotations

import argparse


def compute_mean_utr_sizes(
    input_path: str, out_path: str, utr_name_column: str = "utr_name"
) -> None:
    from utrcalling.core import anndata_processing as ap
    from utrcalling.core import helpers

    adata = helpers.check_if_path_and_open(input_path, method="h5ad")
    adata = ap.compute_utr_size_means(adata=adata, utr_name_column=utr_name_column)
    adata.write_h5ad(out_path)


def create_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:

    help = "Computes the mean UTR size per row of the AnnData object provided."

    description = f"{help} "

    parser = subparsers.add_parser(
        name="compute-mean-utr-sizes",
        help=help,
        description=f"DESCRIPTION: {description}",
        epilog="""
    Example use:
    $ utrcalling compute-mean-utr-sizes adata.h5ad adata_means.h5ad -u utr_name
    """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "input_path", type=str, help="Path to input AnnData object file (h5ad format)."
    )
    parser.add_argument("out_path", type=str, help="Path where output will be stored.")
    parser.add_argument(
        "-u",
        "--utr_name_column",
        type=str,
        default="utr_name",
        help="""
                        The name of the column in the anndata.var dataframe that
                        contains the names of the UTRs.
                        """,
    )
    parser.set_defaults(func=compute_mean_utr_sizes)
