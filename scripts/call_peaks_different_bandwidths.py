import sys
import argparse
import scanpy as sc

sys.path.append('../')  # allows utrcalling to be imported from `python utrcalling.py run`
sys.path.append('./')  # allows utrcalling to be imported from this script
from main_module import peak_calling_anndata as pc


def main(adata_path, out_path, bandwidths):

    adata = sc.read_h5ad(adata_path)

    out_path = out_path.split(".h5ad")[0]
    for bandwidth in bandwidths:
        adata_peaks = pc.peak_call_anndata(adata, method="kde", bandwidth=bandwidth)
        adata_peaks.write_h5ad(
            f"{out_path}_bw{bandwidth}.h5ad"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
    Converts raw UMI sizes to consensus peaks, using a Gaussian KDE with the supplied
    bandwidths. One file per bandwidth is created, by appending "_bw<N>.h5ad" to the 
    supplied out_path.
    """,
        epilog="""
    Example use:
    $ call_peaks_different_bandwidths.py adata.h5ad adata -b 1 7 12
    """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("input_path", type=str,
                        help="Path to input AnnData object file (h5ad format).")
    parser.add_argument("out_path", type=str,
                        help="""
                        Path where output will be stored. '_bw' plus the bandwidth used
                        will be appended to out_path.
                        """)
    parser.add_argument("-b", "--bandwidths", nargs="+", type=int, default=[12],
                        help="One or more bandwidths to use in the gaussian kernel")

    args = parser.parse_args()

    main(args.input_path, args.out_path, args.bandwidths)
