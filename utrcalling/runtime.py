import sys
import argparse
import re

from utrcalling.tools import run
from utrcalling.tools import call_peaks
from utrcalling.tools import prepare_reference_and_map_reads
from utrcalling.tools import remove_peaks_below_count_threshold
from utrcalling.tools import select_alternative_transcription_events
from utrcalling.tools import generate_utr_names
from utrcalling.tools import compute_mean_utr_sizes
from utrcalling.tools import counts_to_percentages
from utrcalling.tools import concatenate_experiments
from utrcalling.tools import subset_experiments
from utrcalling.tools import split_train_test
from utrcalling.tools import estimate_best_bandwidth
from utrcalling.tools import check_distance_to_pas
from utrcalling.tools import call_peaks_different_bandwidths

from utrcalling.tests import test_case

from utrcalling._metadata import __version__ as version

# read the version from the metadata file
# metadata_file = "utrcalling/_metadata.py"
# with open(metadata_file, "rt") as file:
#     metadata = file.read()
#     version_expression = r"^__version__ = ['\"]([^'\"]*)['\"]"
#     version_search = re.search(version_expression, metadata, re.M)
#     if version_search:
#         version = version_search.group(1)
#     else:
#         raise RuntimeError("Unable to find version string in {}.".format(metadata))


# Function to create the parser with subparsers
def create_parser():
    # create the main parser object
    parser = argparse.ArgumentParser(
        description="Suite of tools to map reads to the UTR."
    )
    parser.add_argument(
        '--version', '-V',
        action='version',
        version=f"Software version: {version}"
    )

    # start adding all the subparsers/commands
    subparsers = parser.add_subparsers()

    # add main commands
    run.create_parser(subparsers=subparsers)
    call_peaks.create_parser(subparsers=subparsers)
    prepare_reference_and_map_reads.create_parser(subparsers=subparsers)

    # add the command to run the tests
    test_case.create_parser(subparsers=subparsers)

    # add other tools
    remove_peaks_below_count_threshold.create_parser(subparsers=subparsers)
    select_alternative_transcription_events.create_parser(subparsers=subparsers)
    generate_utr_names.create_parser(subparsers=subparsers)
    compute_mean_utr_sizes.create_parser(subparsers=subparsers)
    counts_to_percentages.create_parser(subparsers=subparsers)
    concatenate_experiments.create_parser(subparsers=subparsers)
    subset_experiments.create_parser(subparsers=subparsers)
    split_train_test.create_parser(subparsers=subparsers)
    estimate_best_bandwidth.create_parser(subparsers=subparsers)
    check_distance_to_pas.create_parser(subparsers=subparsers)
    call_peaks_different_bandwidths.create_parser(subparsers=subparsers)

    return parser


# The main function creating the parser and executing all commands
def main():
    parser = create_parser()
    args = parser.parse_args()
    if len(sys.argv) == 1:
        sys.exit(parser.print_usage())

    args_prepared = vars(args).copy()
    del args_prepared["func"]

    args.func(**args_prepared)
