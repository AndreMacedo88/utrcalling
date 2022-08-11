# import general tools
import argparse
# import data processing tools
import scanpy as sc

from utrcalling.core import anndata_processing as ap


def main(adata_path, out_path):
    adata = sc.read_h5ad(adata_path)
    adata_new = ap.potential_apa_events(adata=adata)
    adata_new.write_h5ad(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
    Removes UTRs that have less than 2 pA sites, effectively selecting for alternative
    poly adenilation (APA) events.
    """,
        epilog="""
    Example use:
    $ remove_non_apa_utrs.py adata_old.h5ad adata_new.h5ad
    """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("input_path", type=str,
                        help="Path to input AnnData object file (h5ad format).")
    parser.add_argument("out_path", type=str, help="Path where output will be stored.")

    args = parser.parse_args()

    main(args.input_path, args.out_path)
