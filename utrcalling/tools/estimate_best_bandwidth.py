# import general tools
import argparse


def estimate_best_bandwidth(input_path, out_path, bandwidths, cores=None):
    import joblib
    from utrcalling.core import peak_calling_anndata as pc
    from utrcalling.core import helpers

    adata = helpers.check_if_path_and_open(input_path, method="h5ad")

    fit_score = pc.estimate_best_bandwidth(
        adata=adata,
        hamming=1,
        bandwidths=bandwidths,
        return_distances_to_hex=False,
        n_jobs=cores,
    )

    scores = {bandwidth: score for bandwidth, score in zip(bandwidths, fit_score)}

    joblib.dump(scores, out_path)


def create_parser(subparsers):

    help = "Calculates scores per bandwidth given."

    description = (
        f"{help} "
        "The score is the average number of UMIs 10-30 bps "
        "after the canonical PAS per UTR in the AnnData object."
    )

    parser = subparsers.add_parser(
        name="estimate-best-bandwidth",
        help=help,
        description=f"DESCRIPTION: {description}",
        epilog="""
    Example use:
    $ utrcalling estimate-best-bandwidth adata.h5ad estimations_scores.joblib -b 1 7 12
    """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "input_path", type=str, help="Path to input AnnData object file (h5ad format)."
    )
    parser.add_argument("out_path", type=str, help="Path where output will be stored.")
    parser.add_argument(
        "-b",
        "--bandwidths",
        nargs="+",
        type=int,
        default=[1, 7, 12, 15],
        help="One or more bandwidths to use in the gaussian kernel",
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=None,
        help=(
            "Max worker processes for the per-UTR parallel loops. Defaults to "
            "JOBLIB_DEFAULT_N_JOBS (-2 = all-but-one core) when omitted."
        ),
    )

    parser.set_defaults(func=estimate_best_bandwidth)
