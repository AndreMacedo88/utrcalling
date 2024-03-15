# import general tools
import argparse

# import internal tools
from utrcalling.core import anndata_processing as ap
from utrcalling.core import helpers


def select_alternative_transcription_events(input_path, out_path):
    adata = helpers.check_if_path_and_open(input_path, method="h5ad")
    adata_new = ap.select_alternative_transcription_events(adata=adata)
    adata_new.write_h5ad(out_path)


def create_parser(subparsers):

    help = "Removes UTRs that have less than 2 alternative transcription sites."

    description = (
        f"{help} "
        "This selects alternative poly adenilation (APA) or "
        "alternative transcription initiation (ATI) events."
    )

    parser = subparsers.add_parser(
        name="select-alternative-transcription-events",
        help=help,
        description=f"DESCRIPTION: {description}",
        epilog="""
    Example use:
    $ utrcalling select-alternative-transcription-events adata_old.h5ad adata_new.h5ad
    """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "input_path", type=str, help="Path to input AnnData object file (h5ad format)."
    )
    parser.add_argument("out_path", type=str, help="Path where output will be stored.")

    parser.set_defaults(func=select_alternative_transcription_events)
