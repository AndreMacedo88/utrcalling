import argparse
import joblib
import scanpy as sc

from Utrcalling.core import peak_calling_anndata as pc


def main(adata_path, out_path, bandwidths):
    blood_data = sc.read_h5ad(adata_path)

    fit_score = pc.estimate_best_bandwidth(
        adata=blood_data, hamming=1, bandwidths=bandwidths, return_distances_to_hex=False)

    scores = {bandwidth: score for bandwidth, score in zip(bandwidths, fit_score)}

    joblib.dump(scores, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
    Calculates a score per bandwidth. The score is the average number of UMIs 10-30 bps 
    after the canonical PAS per UTR in the AnnData object.
    """,
        epilog="""
    Example use:
    $ estimate_best_bandwidth.py adata.h5ad estimations_scores.joblib -b 1 7 12
    """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("input_path", type=str,
                        help="Path to input AnnData object file (h5ad format).")
    parser.add_argument("out_path", type=str, help="Path where output will be stored.")
    parser.add_argument("-b", "--bandwidths", nargs="+", type=int, default=[1, 7, 12],
                        help="One or more bandwidths to use in the gaussian kernel")

    args = parser.parse_args()

    main(args.input_path, args.out_path, args.bandwidths)
