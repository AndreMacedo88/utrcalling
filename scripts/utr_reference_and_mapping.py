import sys
import argparse
import os

sys.path.append('../')  # allows utrcalling to be imported from `python utrcalling.py run`
sys.path.append('./')  # allows utrcalling to be imported from this script
from main_module import core, helpers, protocols
from main_module.module_log import module_logger


def main(utr_annotation, alignments, feature='three_prime_utr',
         protocol_platform="10x", protocol_version=3,
         log_level='info'):
    """
    Gets UTR sizes peak counts per cell/sample from input file/s.

    Args:
        utr_annotation (str): [description]
        alignments (str): [description]
        feature (str, optional): [description]. Defaults to 'three_prime_utr'.
        protocol_version (int, optional): [description]. Defaults to 3.
    """

    # Set up logging
    logger = module_logger(disable=False, store_log=True, level=log_level)
    for arg, value in locals().items():
        logger.info(f'Argument {arg}: {value}\n')

    # Supply the names of your samples, if applicable:
    # This next if clause is specific to my (Andre) workflow due to path. TODO: generalize it.
    if (not isinstance(alignments, list)) and (os.path.isdir(alignments)):
        alignments = helpers.get_paths_from_folders(alignments, suffix='.bam')
        sample_names = [helpers.get_parent_folders_names_from_file(
            file_path) for file_path in alignments]
    elif protocol_platform in ("smartseq", "smart-seq"):
        sample_names = [file_path.split(".")[0].split("/")[1]
                        for file_path in alignments]
    else:
        sample_names = None

    # Set up the protocol for this specific run
    protocol = protocols.Protocol(protocol_platform, int(protocol_version),
                                  sample_names=sample_names)

    # Get UTR reference data from the organisms GTF annotation file
    logger.info('Preparing utr reference regions')
    utr_reference_regions = core.prepare_utr_reference_regions(
        utr_annotation, feature, store=True)

    # Map to the reference UTR the reads that properly aligned to genome

    logger.info('Mapping aligned reads to UTR reference')
    core.map_aligned_reads(
        alignments, utr_reference_regions, protocol, store=True)

    return


keyword_args = {"-a": "--alignments", "-f": "--feature",
                "-p": "--protocol_platform", "-v": "--protocol_version",
                "-l": "--log_level"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Builds a UTR reference and then maps the supplied reads to generate raw UTR sizes.
        """,
        epilog="""
        Example on how to use in command-line:
        $ utr_reference_and_mapping.py mus_musculus.gtf 
            -a bowtie_align.bam bowtie_align_ko.bam -f three_prime_utr -p 10x -v 2
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("utr_annotation", type=str,
                        help="GTF file with the genome annotations")

    parser.add_argument("--alignments", "-a", nargs='+', type=str, required=True, help="""
        File, or files with the read alignments. One alignment per sample should be 
        supplied if downstream applications involve comparisons between the samples.
        Alignments must be sorted and have an index file .bai in the same directory
        """)

    parser.add_argument("--feature", "-f", type=str,
                        help="Type of UTR that you want to map. three_prime_utr or \
                            five_prime_utr",
                        default='three_prime_utr')

    parser.add_argument("--protocol_platform", "-p", type=str,
                        help="Sequencing platform used.",
                        default='10x')

    parser.add_argument("--protocol_version", "-v", type=int,
                        help="Version of the protocol used in library preparation.",
                        default=2)

    parser.add_argument("--log_level", "-l", type=str,
                        help="Level of logging messages to retrieve during the run.",
                        default="info")

    args = parser.parse_args()

    main(args.utr_annotation, args.alignments, feature=args.three_prime_utr,
         protocol_platform=args.protocol_platform,
         protocol_version=args.protocol_version,
         log_level=args.log_level
         )
