# import general tools
import sys
import argparse
# import data processing tools
import scanpy as sc

sys.path.append('../')  # allows utrcalling to be imported from `python utrcalling.py run`
sys.path.append('./')  # allows utrcalling to be imported from this script
from main_module import anndata_processing as ap


def main(adata_path, out_path, utr_name_column, gene_name_column):
    adata = sc.read_h5ad(adata_path)
    ap.friendly_polya_names(
        adata=adata, out=out_path, utr_name_column=utr_name_column,
        gene_name_column=gene_name_column)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
    Changes the var_names of an AnnData object to contain the gene name.
    New name format is as follows:<gene>_<utr number>_<polyA site number>. UTR number
    is only present in genes with more than one annotated UTR.
    """,
        epilog="""
    Example use:
    $ change_to_friendly_pas_names.py adata_old.h5ad adata_new_names.h5ad
    """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("input_path", type=str,
                        help="Path to input AnnData object file (h5ad format).")

    parser.add_argument("out_path", type=str, help="Path where output will be stored.")

    parser.add_argument("--utr_name_column", "-u", type=str, help="""
        Name of the column in the AnnData object that contains the UTR name.
        """, default="utr_name")

    parser.add_argument("--gene_name_column", "-g", type=str, help="""
        Name of the column in the AnnData object that contains the gene name.
        """, default="gene_name")

    args = parser.parse_args()

    main(args.input_path, args.out_path, args.utr_name_column, args.gene_name_column)
