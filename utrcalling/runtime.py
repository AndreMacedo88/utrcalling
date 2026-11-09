"""The utrcalling command line: `utrcalling COMMAND`, or `utrcalling GROUP COMMAND`."""

import sys
from collections.abc import Callable

from cyclopts import App, Group, Parameter

from utrcalling._metadata import __version__ as version
from utrcalling.commands import call_peaks, count, map_reads, run
from utrcalling.commands.process import concatenate, filter_sites, keep_multi_site, rename_sites
from utrcalling.commands.qc import (
    batches,
    cells,
    known_sites,
    pas_distance,
    peaks,
    report,
    score_bandwidths,
)
from utrcalling.commands.tools import mean_sizes, to_fractions

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

# `run` does what `map-reads` and then `count` do; `call-peaks` continues `run -r`.
app.command(run.run, name="run", group=PIPELINE, sort_key=0)
app.command(map_reads.map_reads, name="map-reads", group=PIPELINE, sort_key=1)
app.command(count.count, name="count", group=PIPELINE, sort_key=2)
app.command(call_peaks.call_peaks, name="call-peaks", group=PIPELINE, sort_key=3)

# The groups come in the order of an analysis, like scanpy's pp and tl: process
# the count matrix, then compute the measures to analyse from it.
process = App(
    name="process",
    help="Prepare the site count matrix: filter sites, keep multi-site UTRs, join experiments.",
    group=COMMAND_GROUPS,
    sort_key=0,
)
process.command(filter_sites.filter_sites, name="filter-sites", sort_key=0)
process.command(keep_multi_site.keep_multi_site, name="keep-multi-site", sort_key=1)
process.command(rename_sites.rename_sites, name="rename-sites", sort_key=2)
process.command(concatenate.concatenate, name="concatenate", sort_key=3)
app.command(process)

tools = App(
    name="tools",
    help="Compute the measures to analyse from site counts: site usage, mean UTR sizes.",
    group=COMMAND_GROUPS,
    sort_key=1,
)
tools.command(to_fractions.to_fractions, name="to-fractions", sort_key=0)
tools.command(mean_sizes.mean_sizes, name="mean-sizes", sort_key=1)
app.command(tools)

qc = App(
    name="qc",
    help="Quality control: tables that describe sites, cells and samples, at any step.",
    group=COMMAND_GROUPS,
    sort_key=2,
)
# From the whole picture to the details: the matrices, cells, groups of cells,
# and sites, then the choice of bandwidth.
qc.command(report.report, name="report", sort_key=0)
qc.command(cells.cells, name="cells", sort_key=1)
qc.command(batches.batches, name="batches", sort_key=2)
qc.command(known_sites.known_sites, name="known-sites", sort_key=3)
qc.command(pas_distance.pas_distance, name="pas-distance", sort_key=4)
qc.command(peaks.peaks, name="peaks", sort_key=5)
qc.command(score_bandwidths.score_bandwidths, name="score-bandwidths", sort_key=6)
app.command(qc)

# The commands' names before version 4.0.0, which the paper's workflows use. They
# keep working, but the help lists only the names above. (`call-peaks` with -c,
# now `count`, is rewritten by update_old_command_lines.)
OLD_NAMES: dict[str, Callable[..., None]] = {
    "prepare-reference-and-map-reads": map_reads.map_reads,
    "call-peaks-different-bandwidths": call_peaks.call_peaks,
    "remove-peaks-below-threshold": filter_sites.filter_sites,
    "select-alternative-transcription-events": keep_multi_site.keep_multi_site,
    "generate-names": rename_sites.rename_sites,
    "concatenate-experiments": concatenate.concatenate,
    "counts-to-percentages": to_fractions.to_fractions,
    "compute-mean-utr-sizes": mean_sizes.mean_sizes,
    "check-distance-to-pas": pas_distance.pas_distance,
    "estimate-best-bandwidth": score_bandwidths.score_bandwidths,
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


def update_old_command_lines(tokens: list[str]) -> list[str]:
    """Rewrite a command line written for an earlier version as today's.

    Long options get hyphens (see hyphenate_long_options). Before version 5.0.0,
    `call-peaks` counted the sites from the read mappings that `-c` gives: that
    is `count` now, and `call-peaks` takes a raw-sizes count matrix.
    """
    tokens = hyphenate_long_options(tokens)
    old_call_peaks = tokens[:1] == ["call-peaks"] and any(
        token == "-c" or token.partition("=")[0] == "--utr-sizes-counts" for token in tokens
    )
    if old_call_peaks:
        tokens = ["count", *tokens[1:]]
    return tokens


def main() -> None:
    from utrcalling import show_logo

    show_logo()
    app(update_old_command_lines(sys.argv[1:]))


if __name__ == "__main__":
    main()
