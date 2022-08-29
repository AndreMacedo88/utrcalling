# import general tools
import argparse
# import internal tools
from utrcalling.core import anndata_processing as ap
from utrcalling.core import helpers


def select_alternative_transcription_events(input_path, out_path):
    adata = helpers.check_if_path_and_open(input_path, method="h5ad")
    adata_new = ap.select_alternative_transcription_events(adata=adata)
    adata_new.write_h5ad(out_path)


def main():
    args_all = parser.parse_args()

    args = vars(args_all).copy()
    del args["func"]

    args_all.func(**args)


parser = argparse.ArgumentParser(
    description="""
Removes UTRs that have less than 2 alternative transcription sites, effectively 
selecting alternative poly adenilation (APA) or alternative transcription 
initiation (ATI) events.
""",
    epilog="""
Example use:
$ select-alternative-transcription-events adata_old.h5ad adata_new.h5ad
""",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("input_path", type=str,
                    help="Path to input AnnData object file (h5ad format).")
parser.add_argument("out_path", type=str, help="Path where output will be stored.")

# This allows for assigning different parsers to different functions:
parser.set_defaults(func=select_alternative_transcription_events)

if __name__ == "__main__":
    main()
