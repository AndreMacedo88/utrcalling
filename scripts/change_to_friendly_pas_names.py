# import general tools
import sys
import argparse
# import data processing tools
import scanpy as sc

sys.path.append('../')  # allows utrcalling to be imported from `python utrcalling.py run`
sys.path.append('./')  # allows utrcalling to be imported from this script
from utrcalling import utr_sizes_processing as up


def main(adata_path, out_path):
    adata = sc.read_h5ad(adata_path)
    up.friendly_polya_names(adata=adata, out=out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
    Changes the var_names of an AnnData object to contain the gene name.
    New name format is as follows:<gene>_<utr number>_<polyA site number>. UTR number
    is only present in genes with more than one annotated UTR.
    """,
        epilog="""
    Example use:
    $ python change_to_friendly_pas_names.py adata_old.h5ad adata_new_names.h5ad
    """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("input_path", type=str,
                        help="Path to input AnnData object file (h5ad format).")
    parser.add_argument("out_path", type=str, help="Path where output will be stored.")

    args = parser.parse_args()

    main(args.input_path, args.out_path)
