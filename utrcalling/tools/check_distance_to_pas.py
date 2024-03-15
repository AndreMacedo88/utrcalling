# import general tools
import argparse
import joblib
import pybedtools

# import internal tools
from utrcalling.core import peak_calling_anndata as pc
from utrcalling.core import helpers


def check_distance_to_pas(input_path, out_path, genome_path, hamming):
    adata = helpers.check_if_path_and_open(input_path, method="h5ad")
    genome = pybedtools.BedTool(genome_path)
    distances_array = pc.pas_distance(adata=adata, genome=genome, hamming=hamming)

    joblib.dump(distances_array, out_path)


def create_parser(subparsers):

    help = (
        "Produces a vector with the distance of every mapped UMI to the canonical poly "
        "adenylation start site."
    )

    description = (
        f"{help} "
        "Degeneration of the canonical PAS can be accounted for by "
        "supplying hamming > 0."
    )

    parser = subparsers.add_parser(
        name="check-distance-to-pas",
        help=help,
        description=f"DESCRIPTION: {description}",
        epilog="""
    Example use:
    $ utrcalling check-distance-to-pas adata.h5ad distances_to_pas.joblib
        Homo_sapiens.GRCh38.dna.primary_assembly.fa -H 0
    """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "input_path", type=str, help="Path to input AnnData object file (h5ad format)."
    )
    parser.add_argument("out_path", type=str, help="Path where output will be stored.")
    parser.add_argument(
        "genome_path", type=str, help="Path to a genome file (fasta format)."
    )
    parser.add_argument(
        "-H",
        "--hamming",
        type=int,
        default=0,
        help="""
                        Number of differences to the canonical PAS hexamer (AATAAA)
                        that can be tolerated
                        """,
    )

    parser.set_defaults(func=check_distance_to_pas)
