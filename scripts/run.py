import sys
sys.path.append('../')
import argparse

from utrcalling import core
from utrcalling.module_log import module_logger

def main(utr_annotation, genome, alignments, feature='three_prime_utr', 
single_cell_annotation=None, protocol_version=3, out="UTR_results.h5ad"):
    """
    Gets VEnCodes from an input file.
    Parameters
    ----------
    utr_annotation : str
    genome : str
    alignments : str
    feature : str
    single_cell_annotation : str
    protocol_version : int
    out : str
    """

    # Set up logging
    logger = module_logger(disable=False, store_log=True)

    logger.info('Preparing utr reference regions')
    utr_reference_regions = core.prepare_utr_reference_regions(utr_annotation, genome, 
    feature)

    logger.info('Mapping aligned reads to UTR reference')
    utr_sizes_counts = core.map_aligned_reads(
        alignments, utr_reference_regions,protocol_version=int(protocol_version))
    
    logger.info('Calling peaks from mapped data')
    count_matrix_anndata = core.call_peaks(
        utr_sizes_counts, utr_reference_regions, annotations=single_cell_annotation)
    
    count_matrix_anndata.write(filename=out)
    
    return

keyword_args = {"-a": "--alignments", "-f": "--feature", "-s": "--single_cell_annotation",
    "-p": "--protocol_version", "-o": "--out"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
        Runs the entire UTR calling pipeline.
        Exports the UTR sizes count matrix to a h5ad file, which you can name with --out.
        Example on how to use in command-line:
        $ python run.py mus_musculus.gtf mus_musculus.fasta 
            -a bowtie_align.bam bowtie_align_ko.bam -f three_prime_utr 
            -s experiment.csv -p 2 -o results.h5ad
        """)
    parser.add_argument("utr_annotation", type=str, 
        help="GTF file with the genome annotations")

    parser.add_argument("genome", type=str, help="File with the genomic sequences")
    
    parser.add_argument("--alignments", "-a", nargs='+', type=str, required=True, help="""
        File, or files with the read alignments. One alignment per sample should be 
        supplied if downstream applications involve comparisons between the samples.
        Alignments must be sorted and have an index file .bai in the same directory
        """)

    parser.add_argument("--feature", "-f", type=str, 
        help="Type of UTR that you want to map. three_prime_utr or five_prime_utr", 
        default='three_prime_utr')

    parser.add_argument("--single_cell_annotation", "-s", type=str, 
        help="File with your experiment's information, such as cell barcodes.", 
        default=None)
    
    parser.add_argument("--protocol_version", "-p", type=int, 
        help="Version of the 10x protocol used in library preparation.", default=2)

    parser.add_argument("--out", "-o", type=str, 
        help="Path or partial path to a folder to store the result files.", 
        default="UTR_results.h5ad")

    args = parser.parse_args()

    main(args.utr_annotation, args.genome, args.alignments, feature=args.three_prime_utr,
        single_cell_annotation=args.single_cell_annotation, 
        protocol_version=args.protocol_version,
        out=args.out
        )