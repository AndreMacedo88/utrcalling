"""The utrcalling command line: `utrcalling COMMAND`, or `utrcalling GROUP COMMAND`."""

import sys
from collections.abc import Callable

from cyclopts import App, Group, Parameter

from utrcalling._metadata import __version__ as version
from utrcalling.tools import (
    call_peaks,
    call_peaks_different_bandwidths,
    check_distance_to_pas,
    compute_mean_utr_sizes,
    concatenate_experiments,
    counts_to_percentages,
    estimate_best_bandwidth,
    prepare_reference_and_map_reads,
    remove_peaks_below_count_threshold,
    run,
    select_alternative_transcription_events,
)

PIPELINE = Group("Pipeline", sort_key=0)
COMMAND_GROUPS = Group("Command groups", sort_key=1)

app = App(
    name="utrcalling",
    help="Count the molecules at each UTR site, per cell or sample, from RNA-seq reads.",
    version=f"Software version: {version}",
    version_flags=["--version", "-V"],
    # Flags only switch things on: no --no-raw-sizes and the like.
    default_parameter=Parameter(negative=()),
    # Return to main() instead of exiting, so that tests can run commands in-process.
    result_action="return_value",
)
app["--help"].group = "Help"
app["--version"].group = "Help"

app.command(run.run, name="run", group=PIPELINE, sort_key=0)
app.command(
    prepare_reference_and_map_reads.generate_utr_reference_and_map_reads,
    name="prepare-reference-and-map-reads",
    group=PIPELINE,
    sort_key=1,
)
app.command(call_peaks.call_peaks, name="call-peaks", group=PIPELINE, sort_key=2)

process = App(
    name="process",
    help="Process count matrices: filter sites, keep alternative sites, convert counts.",
    group=COMMAND_GROUPS,
)
# In the order of a typical workflow.
process.command(
    remove_peaks_below_count_threshold.remove_peaks_below_count_threshold,
    name="filter",
    sort_key=0,
)
process.command(
    select_alternative_transcription_events.select_alternative_transcription_events,
    name="select-alternative",
    sort_key=1,
)
process.command(counts_to_percentages.counts_to_percentages, name="to-fractions", sort_key=2)
process.command(compute_mean_utr_sizes.compute_mean_utr_sizes, name="mean-sizes", sort_key=3)
process.command(concatenate_experiments.concatenate_experiments, name="concatenate", sort_key=4)
app.command(process)

qc = App(
    name="qc",
    help="Quality control: distances of molecules to polyadenylation signals.",
    group=COMMAND_GROUPS,
)
qc.command(check_distance_to_pas.check_distance_to_pas, name="pas-distance")
app.command(qc)

bandwidth = App(
    name="bandwidth",
    help="Choose the KDE bandwidth of peak calling: score bandwidths, call peaks with several.",
    group=COMMAND_GROUPS,
)
bandwidth.command(estimate_best_bandwidth.estimate_best_bandwidth, name="score", sort_key=0)
bandwidth.command(
    call_peaks_different_bandwidths.call_peaks_different_bandwidths, name="call-peaks", sort_key=1
)
app.command(bandwidth)

# The commands' names before version 4.0.0, which the paper's workflows use. They
# keep working, but the help lists only the names above.
OLD_NAMES: dict[str, Callable[..., None]] = {
    "remove-peaks-below-threshold": (
        remove_peaks_below_count_threshold.remove_peaks_below_count_threshold
    ),
    "select-alternative-transcription-events": (
        select_alternative_transcription_events.select_alternative_transcription_events
    ),
    "counts-to-percentages": counts_to_percentages.counts_to_percentages,
    "compute-mean-utr-sizes": compute_mean_utr_sizes.compute_mean_utr_sizes,
    "concatenate-experiments": concatenate_experiments.concatenate_experiments,
    "check-distance-to-pas": check_distance_to_pas.check_distance_to_pas,
    "estimate-best-bandwidth": estimate_best_bandwidth.estimate_best_bandwidth,
    "call-peaks-different-bandwidths": (
        call_peaks_different_bandwidths.call_peaks_different_bandwidths
    ),
}
for old_name, command in OLD_NAMES.items():
    app.command(command, name=old_name, show=False)


def hyphenate_long_options(tokens: list[str]) -> list[str]:
    """Spell long options with hyphens: `--kde_bandwidth` becomes `--kde-bandwidth`.

    Before version 4.0.0 they were spelled with underscores, as in the paper's
    workflows. Tokens after `--` are values, and stay as they are.
    """
    hyphenated: list[str] = []
    for position, token in enumerate(tokens):
        if token == "--":
            return hyphenated + tokens[position:]
        if token.startswith("--"):
            option, equals, value = token.partition("=")
            token = option.replace("_", "-") + equals + value
        hyphenated.append(token)
    return hyphenated


def main() -> None:
    from utrcalling import show_logo

    show_logo()
    app(hyphenate_long_options(sys.argv[1:]))


if __name__ == "__main__":
    main()
