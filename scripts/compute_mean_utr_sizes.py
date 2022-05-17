import sys
import argparse
import scanpy as sc

sys.path.append('../')  # allows utrcalling to be imported from `python utrcalling.py run`
sys.path.append('./')  # allows utrcalling to be imported from this script
from main_module import anndata_processing as ap


def main(adata_path, out_path, utr_name_column="utr_name"):

    adata = sc.read_h5ad(adata_path)

    adata = ap.compute_cell_means_and_var(adata=adata, utr_name_column=utr_name_column)
    
    adata.write_h5ad(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
    Calculates the variance and mean UTR size per row of the AnnData object provided
    """,
        epilog="""
    Example use:
    $ compute_mean_utr_sizes.py adata.h5ad adata_means.h5ad -u utr_name
    """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("input_path", type=str,
                        help="Path to input AnnData object file (h5ad format).")
    parser.add_argument("out_path", type=str,
                        help="Path where output will be stored.")
    parser.add_argument("-u", "--utr_name_column", type=str, default="utr_name",
                        help="""
                        The name of the column in the anndata.var dataframe that 
                        contains the names of the UTRs.
                        """)

    args = parser.parse_args()

    main(args.input_path, args.out_path, utr_name_column = args.utr_name_column)

