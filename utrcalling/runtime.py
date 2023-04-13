import sys
import argparse
import ast
from os import path

from utrcalling.tools import run

"""
read the CLI version from the VERSION file
"""
# DIR_PATH = path.dirname(path.realpath(__file__))
# VERSION_FILE = path.join(DIR_PATH, './../VERSION')


def add(args):
    print(args.num1 + args.num2)


def subtract(args):
    print(args.num1 - args.num2)


"""
function to create the parser with subparsers
"""


def create_parser():
    # create the main parser object
    parser = argparse.ArgumentParser(
        description='Suite of tools to map reads to the UTR'
    )
    # version = open(VERSION_FILE).read().strip()
    # parser.add_argument('--version', '-V',
    #                     action='version', version=version)

    subparsers = parser.add_subparsers()

    # Run subparser
    parser_run = subparsers.add_parser(
        name='run',
        description="""
Runs the entire UTR calling pipeline.
Exports the UTR sizes count matrix to a h5ad file, which you can name with --out.
""",
        epilog="""
Example on how to use in command-line:
$ run.py mus_musculus.gtf
    -a bowtie_align.bam bowtie_align_ko.bam -f three_prime_utr -p 10x -v 2
    -i -A experiment.csv -o results.h5ad
""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser_run.add_argument("utr_annotation", type=str,
                            help="GTF file with the genome annotations")

    parser_run.add_argument("--alignments", "-a", nargs='+', type=str, required=True, help="""
        File, or files with the read alignments. One alignment per sample/batch should be
        supplied if downstream applications involve comparisons between the samples/batches.
        Alignments must be sorted and have an index file .bai in the same directory
        """)

    parser_run.add_argument("--feature", "-f", type=str,
                            help="Type of UTR that you want to map. three_prime_utr or \
                            five_prime_utr",
                            default='three_prime_utr')

    parser_run.add_argument("--annotations_file", "-A", type=str,
                            help="""
                        File with your experiment's information, such as cell barcodes.
                        """,
                            default=None)

    parser_run.add_argument("--protocol_platform", "-p", type=str,
                            help="Sequencing platform used.",
                            default='10x')

    parser_run.add_argument("--protocol_version", "-v", type=int,
                            help="Version of the protocol used in library preparation.",
                            default=2)

    parser_run.add_argument("--batch_merge_method", "-m", type=str,
                            help="""
                        Method used to deal with different UTR coverage in different
                        samples/batches. If `union`, then the tool outputs UTRs that are not
                        represented by all samples/batches. If `intersection` only the reads that
                        map to UTRs in common between all samples/batches will be used. Default
                        depends on the protocol used.
                        """,
                            default=None)

    parser_run.add_argument("--store_intermediate", "-i", action="store_true",
                            help="""
                        Stores intermediate files that can be used to
                        run quicker pipelines later on. Useful if something wrong
                        is happening in later stages of the pipeline.
                        """)

    parser_run.add_argument("--raw_sizes", "-r", action="store_true",
                            help="""
                        Produces a count matrix with the individual UMI UTR sizes, without
                        aggregating individual molecules' UTR sizes into consensus peaks.
                        """)

    parser_run.add_argument("--label_batches", "-L", type=ast.literal_eval,
                            help="""
                        If True, gets the batches from the annotation file and labels each
                        barcode/sample according to the batch. This prevents aggregation
                        of the results if barcodes are recycled between batches, or when
                        2 samples have the same ID.
                        """,
                            default=None)

    parser_run.add_argument("--log_level", "-l", type=str,
                            help="Level of logging messages to retrieve during the run.",
                            default="info")

    parser_run.add_argument("--out", "-o", type=str,
                            help="Path or partial path to a folder to store the result files.",
                            default="UTR_results.h5ad")

    # addition subparser
    add_subparser = subparsers.add_parser('add',
                                          help='add 2 numbers')
    add_subparser.add_argument('--num1', '-a',
                               help='first number in operation',
                               type=int, required=True)
    add_subparser.add_argument('--num2', '-b',
                               help='second number in operation',
                               type=int, required=True)
    add_subparser.set_defaults(func=add)

    # subtraction subparser
    subtract_subparser = subparsers.add_parser('subtract',
                                               help='subtract 2 numbers')
    subtract_subparser.add_argument('--num1', '-a',
                                    help='first number in operation',
                                    type=int, required=True)
    subtract_subparser.add_argument('--num2', '-b',
                                    help='second number in operation',
                                    type=int, required=True)
    subtract_subparser.set_defaults(func=subtract)

    return parser


"""
the main function creating the parser and executing all commands
"""


def main():
    parser = create_parser()
    args = parser.parse_args()
    if len(sys.argv) == 1:
        sys.exit(parser.print_usage())
    args.func(args)
