import sys
import argparse
import joblib
import scanpy as sc
import pybedtools

sys.path.append('../')  # allows utrcalling to be imported from `python utrcalling.py run`
sys.path.append('./')  # allows utrcalling to be imported from this script
from main_module import peak_calling_anndata as pc


def main(adata_path, out_path, genome_path, hamming):
    adata = sc.read_h5ad(adata_path)
    genome = pybedtools.BedTool(genome_path)

    distances_array = pc.pas_distance(adata=adata, genome=genome, hamming=hamming)

    joblib.dump(distances_array, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
    Produces a vector with the distance of every mapped UMI to the canonical poly 
    adenylation start site.
    Degeneration of the canonical PAS can be accounted for by supplying hamming > 0.
    """,
        epilog="""
    Example use:
    $ check_distance_to_pas.py adata.h5ad distances_to_pas.joblib 
        Homo_sapiens.GRCh38.dna.primary_assembly.fa -H 0
    """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("input_path", type=str,
                        help="Path to input AnnData object file (h5ad format).")
    parser.add_argument("out_path", type=str, help="Path where output will be stored.")
    parser.add_argument("genome_path", type=str,
                        help="Path to a genome file (fasta format).")
    parser.add_argument("-H", "--hamming", type=int, default=0,
                        help="""
                        Number of differences to the canonical PAS hexamer (AATAAA)
                        that can be tolerated
                        """)

    args = parser.parse_args()

    main(args.input_path, args.out_path, args.genome_path, args.hamming)
