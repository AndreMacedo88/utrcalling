import sys
import argparse
import scanpy as sc

sys.path.append('../')  # allows utrcalling to be imported from `python utrcalling.py run`
sys.path.append('./')  # allows utrcalling to be imported from this script
from main_module import anndata_processing as ap


def main(adata_path, out_path, utr_name_column="utr_name"):

    adata = sc.read_h5ad(adata_path)

    adata = ap.counts_to_percentages(adata=adata, utr_name_column=utr_name_column)
    
    adata.write_h5ad(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
    Converts the UMI counts per polyA site into percentage polyA site usage in relation
    to the total sum of UMIs in the UTR.
    """,
        epilog="""
    Example use:
    $ counts_to_percentages.py adata.h5ad adata_percentages -u utr_name
    """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("input_path", type=str,
                        help="Path to input AnnData object file (h5ad format).")
    parser.add_argument("out_path", type=str,
                        help="""
                        Path where output will be stored. '_bw' plus the bandwidth used
                        will be appended to out_path.
                        """)
    parser.add_argument("-u", "--utr_name_column", type=str, default="utr_name",
                        help="""
                        The name of the column in the anndata.var dataframe that 
                        contains the names of the UTRs.
                        """)

    args = parser.parse_args()

    main(args.input_path, args.out_path, utr_name_column = args.utr_name_column)

