# import general tools
import argparse
# import data processing tools
import scanpy as sc

from utrcalling.core import peak_calling_anndata as pc


def main(adata_path, out_path, threshold):
    adata = sc.read_h5ad(adata_path)
    adata_new = pc.remove_peaks_below_count(adata=adata, threshold=threshold)
    adata_new.write_h5ad(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
    Removes the UTR location peaks that have less than `threshold` counts in
    the sum of all cells/samples/batches. This effectively curates the peaks, selecting the most common
    and/or more prone to be true peaks.
    """,
        epilog="""
    Example use:
    $ remove_peaks_below_count_threshold.py adata_old.h5ad adata_filtered.h5ad -t 10
    """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("input_path", type=str,
                        help="Path to input AnnData object file (h5ad format).")
    parser.add_argument("out_path", type=str, help="Path where output will be stored.")
    parser.add_argument("-t", "--threshold", type=int,
                        help="Count threshold to filter the data.", default=10)

    args = parser.parse_args()

    main(args.input_path, args.out_path, args.threshold)
