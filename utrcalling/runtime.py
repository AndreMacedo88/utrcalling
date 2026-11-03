import argparse
import sys

from utrcalling._metadata import __version__ as version


# Function to create the parser with subparsers
def create_parser():
    from utrcalling.tests import test_case
    from utrcalling.tools import (
        call_peaks,
        call_peaks_different_bandwidths,
        check_distance_to_pas,
        compute_mean_utr_sizes,
        concatenate_experiments,
        counts_to_percentages,
        estimate_best_bandwidth,
        generate_utr_names,
        prepare_reference_and_map_reads,
        remove_peaks_below_count_threshold,
        run,
        select_alternative_transcription_events,
        split_train_test,
        subset_experiments,
    )

    # create the main parser object
    parser = argparse.ArgumentParser(description="Suite of tools to map reads to the UTR.")
    parser.add_argument("--version", "-V", action="version", version=f"Software version: {version}")

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
    from utrcalling import show_logo

    show_logo()

    parser = create_parser()
    args = parser.parse_args()
    if len(sys.argv) == 1:
        sys.exit(parser.print_usage())

    args_prepared = vars(args).copy()
    del args_prepared["func"]

    args.func(**args_prepared)


if __name__ == "__main__":
    main()
