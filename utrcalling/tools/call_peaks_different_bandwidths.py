# import general tools
import argparse
# import internal tools
from utrcalling.core import peak_calling_anndata as pc
from utrcalling.core import helpers


def call_peaks_different_bandwidths(input_path, out_path, bandwidths):
    adata = helpers.check_if_path_and_open(input_path, method="h5ad")
    out_path = out_path.split(".h5ad")[0]
    for bandwidth in bandwidths:
        adata_peaks = pc.peak_call_anndata(adata, method="kde", bandwidth=bandwidth)
        adata_peaks.write_h5ad(
            f"{out_path}_bw{bandwidth}.h5ad"
        )


def main():
    args_all = parser.parse_args()

    args = vars(args_all).copy()
    del args["func"]

    args_all.func(**args)


parser = argparse.ArgumentParser(
    description="""
Converts raw UMI sizes to consensus peaks, using a Gaussian KDE with the supplied
bandwidths. One file per bandwidth is created, by appending "_bw<N>.h5ad" to the 
supplied out_path.
""",
    epilog="""
Example use:
$ call-peaks-different-bandwidths adata.h5ad adata -b 1 7 12 15
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


# This allows for assigning different parsers to different functions:
parser.set_defaults(func=call_peaks_different_bandwidths)

if __name__ == "__main__":
    main()
