# import general tools
import argparse
import joblib
# import internal tools
from utrcalling.core import peak_calling_anndata as pc
from utrcalling.core import helpers


def estimate_best_bandwidth(input_path, out_path, bandwidths):
    adata = helpers.check_if_path_and_open(input_path, method="h5ad")

    fit_score = pc.estimate_best_bandwidth(
        adata=adata, hamming=1, bandwidths=bandwidths, return_distances_to_hex=False)

    scores = {bandwidth: score for bandwidth, score in zip(bandwidths, fit_score)}

    joblib.dump(scores, out_path)


def main():
    args_all = parser.parse_args()

    args = vars(args_all).copy()
    del args["func"]

    args_all.func(**args)


parser = argparse.ArgumentParser(
    description="""
Calculates a score per bandwidth. The score is the average number of UMIs 10-30 bps 
after the canonical PAS per UTR in the AnnData object.
""",
    epilog="""
Example use:
$ estimate-best-bandwidth adata.h5ad estimations_scores.joblib -b 1 7 12
""",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("input_path", type=str,
                    help="Path to input AnnData object file (h5ad format).")
parser.add_argument("out_path", type=str, help="Path where output will be stored.")
parser.add_argument("-b", "--bandwidths", nargs="+", type=int, default=[1, 7, 12, 15],
                    help="One or more bandwidths to use in the gaussian kernel")

# This allows for assigning different parsers to different functions:
parser.set_defaults(func=estimate_best_bandwidth)

if __name__ == "__main__":
    main()
