"""Command-line options shared by the pipeline commands.

`run`, `map-reads` and `count` take many of the same
options. Each alias below carries the option's flags, help panel and help text,
so the three commands describe them the same way.
"""

from collections.abc import Sequence
from typing import Annotated, Literal, get_args

from cyclopts import Group, Parameter, Token

from utrcalling.core.peak_calling.settings import (
    CANDIDATE_BANDWIDTHS,
    OPTIMIZE,
    PEAK_CALLERS,
    SCORES,
    Bandwidth,
    Kernel,
    check_kernel,
    parse_bandwidth,
)

# Help panels of the pipeline commands, in this order.
INPUTS = Group("Inputs", sort_key=0)
PROTOCOL = Group("Protocol and samples", sort_key=1)
PEAK_CALLING = Group("Peak calling", sort_key=2)
OUTPUT = Group("Output and resources", sort_key=3)


def truth_value(type_: object, tokens: Sequence[Token]) -> bool:
    """Read true/false, yes/no or 1/0 in any case, as in `-L True`."""
    text = tokens[0].value
    if text.lower() in ("true", "yes", "1"):
        return True
    if text.lower() in ("false", "no", "0"):
        return False
    raise ValueError(f"expected true or false, got {text!r}")


def bandwidth(type_: object, tokens: Sequence[Token]) -> Bandwidth:
    """Read a KDE bandwidth: a number of nt, or a rule such as scott:0.5."""
    return parse_bandwidth(tokens[0].value)


def bandwidths(type_: object, tokens: Sequence[Token]) -> tuple[Bandwidth, ...]:
    """Read one or more KDE bandwidths, where 0 keeps the raw sizes."""
    return tuple(parse_bandwidth(token.value, allow_zero=True) for token in tokens)


def candidate_bandwidths(type_: object, tokens: Sequence[Token]) -> tuple[Bandwidth, ...]:
    """Read one or more KDE bandwidths to call peaks with (not 0)."""
    return tuple(parse_bandwidth(token.value) for token in tokens)


def kernels(type_: object, tokens: Sequence[Token]) -> tuple[str, ...]:
    """Read one or more KDE kernels, in any case."""
    return tuple(dict.fromkeys(check_kernel(token.value.lower()) for token in tokens))


def peak_caller(type_: object, tokens: Sequence[Token]) -> str:
    """Read the name of a peak caller, in any case."""
    return PEAK_CALLERS.check(tokens[0].value)


def scores(type_: object, tokens: Sequence[Token]) -> tuple[str, ...]:
    """Read the names of one or more scores, in any case."""
    return tuple(dict.fromkeys(SCORES.check(token.value) for token in tokens))


def bandwidth_or_optimize(type_: object, tokens: Sequence[Token]) -> Bandwidth:
    """Read a KDE bandwidth, or optimize."""
    text = tokens[0].value
    return OPTIMIZE if text.strip().lower() == OPTIMIZE else parse_bandwidth(text)


def bandwidths_or_optimize(type_: object, tokens: Sequence[Token]) -> tuple[Bandwidth, ...]:
    """Read one or more KDE bandwidths (0 keeps the raw sizes), or optimize alone."""
    if any(token.value.strip().lower() == OPTIMIZE for token in tokens):
        if len(tokens) > 1:
            raise ValueError("optimize chooses among the --candidates: give it alone")
        return (OPTIMIZE,)
    return bandwidths(type_, tokens)


def lowercase_choice(type_: object, tokens: Sequence[Token]) -> str:
    """Read one of the Literal *type_*'s values in any case, as in `-s UTR`."""
    text = tokens[0].value
    choices = get_args(type_)
    if text.lower() not in choices:
        raise ValueError(f"expected one of {', '.join(choices)}, got {text!r}")
    return text.lower()


Alignments = Annotated[
    list[str],
    Parameter(
        name=("--alignments", "-a"),
        consume_multiple=1,
        group=INPUTS,
        help=(
            "BAM files with the aligned reads, sorted and indexed (with a .bai file next "
            "to each). Give one per sample or batch if you will compare them later."
        ),
    ),
]
AnnotationsFile = Annotated[
    str,
    Parameter(
        name=("--annotations-file", "-A"),
        group=INPUTS,
        help=(
            "Table (.csv) with your experiment's information, such as cell barcodes, "
            "sample IDs and batches."
        ),
    ),
]
Feature = Annotated[
    str,
    Parameter(
        name=("--feature", "-f"),
        group=INPUTS,
        help="GTF feature type of the UTRs to map: three_prime_utr or five_prime_utr.",
    ),
]

ProtocolPlatform = Annotated[
    str,
    Parameter(
        name=("--protocol-platform", "-p"),
        group=PROTOCOL,
        help=(
            "Sequencing protocol: 10x, smartseq, smart-seq_cap, smart-seq_cap-readname, "
            "5prime10x, standard_10x (Cell Ranger BAM), standard_smartseq, truseq "
            "(stranded bulk RNA-seq, e.g. TruSeq Stranded mRNA) or bulk_pe (unstranded "
            "paired-end bulk RNA-seq; one observation per fragment, at the fragment's "
            "polyA-distal base)."
        ),
    ),
]
ProtocolVersion = Annotated[
    int,
    Parameter(
        name=("--protocol-version", "-v"),
        group=PROTOCOL,
        help="Version of the protocol used in library preparation.",
    ),
]
Strandedness = Annotated[
    Literal["reverse", "forward", "unstranded"] | None,
    Parameter(
        group=PROTOCOL,
        help=(
            "Library strandedness, for stranded bulk protocols such as truseq: reverse "
            "(the default for truseq, right for TruSeq Stranded mRNA), forward or "
            "unstranded. Protocols with a fixed strand convention ignore it."
        ),
    ),
]
# `str` makes the option take a value, as in `-L True`: a plain bool is a flag.
LabelBatches = Annotated[
    bool | str | None,
    Parameter(
        name=("--label-batches", "-L"),
        converter=truth_value,
        metavar="TRUE|FALSE",
        group=PROTOCOL,
        help=(
            "Whether to label each barcode or sample with its batch from the annotation "
            "file, so recycled barcodes or repeated sample IDs stay apart. The default "
            "depends on the protocol."
        ),
    ),
]
BatchMergeMethod = Annotated[
    Literal["union", "intersection"] | None,
    Parameter(
        name=("--batch-merge-method", "-m"),
        group=PROTOCOL,
        help=(
            "How to combine samples or batches that cover different UTRs: union keeps "
            "every UTR, intersection only the UTRs that all of them cover. The default "
            "depends on the protocol."
        ),
    ),
]

SummarizationLevel = Annotated[
    Literal["utr", "gene"],
    Parameter(
        name=("--summarization-level", "-s"),
        converter=lowercase_choice,
        group=PEAK_CALLING,
        help="Count sites per (merged) UTR, or per gene, joining the UTRs of each gene.",
    ),
]
RawSizes = Annotated[
    bool,
    Parameter(
        name=("--raw-sizes", "-r"),
        group=PEAK_CALLING,
        show_default=False,
        help=(
            "Count each molecule at its own UTR size instead of collapsing the sizes "
            "into peaks. The KDE peaks then go to a second file, `<out>_kde.h5ad`."
        ),
    ),
]
# The help of the bandwidth options: what a rule is.
BANDWIDTH_RULES_HELP = (
    "A rule sets a bandwidth per UTR from the spread of its sizes, narrower as its "
    "molecules grow in number: scott (SD x n^-1/5) or silverman (0.9 x min(SD, "
    "IQR/1.34) x n^-1/5, which several sites in a UTR inflate less), scaled as in "
    "scott:0.5."
)
# The help of the options that -b optimize needs.
OPTIMIZE_HELP = (
    "Or optimize: score the --candidates with each --kde-kernel by the sites that look "
    "true (the polyadenylation signal 10-30 nt before 3' sites, the initiator at 5' "
    "sites, or the --known-sites; see --score and qc score-bandwidths), and call the "
    "peaks with the best. Their scores go to <out>_bandwidth_scores.tsv."
)
KdeBandwidth = Annotated[
    Bandwidth,
    Parameter(
        name=("--kde-bandwidth", "-b"),
        converter=bandwidth_or_optimize,
        metavar="NT|RULE|optimize",
        group=PEAK_CALLING,
        help=(
            "Bandwidth of the KDE that collapses UTR sizes into peaks: a number of nt, "
            "or a rule. With the Gaussian kernel, each molecule spreads into a bell "
            "curve whose standard deviation is the bandwidth: with 12, two equally used "
            "sites stay apart only from 25 nt on, and less equal ones from further. 12 "
            "suits 3'-end-enriched assays, where reads pile up within a "
            "few bp of the polyadenylation site. Randomly fragmented libraries spread "
            "more per site and need a wider kernel, or the spread splits into many close "
            "pseudo-peaks. For 5' data, 12 joins start sites a few nt apart: try "
            f"--kde-kernel linear with -b optimize. {BANDWIDTH_RULES_HELP} {OPTIMIZE_HELP}"
        ),
    ),
]
# The help of the kernel options: what the kernels do.
KERNELS_HELP = (
    "The Gaussian smooths the most and suits 3' data. The linear kernel keeps the peaks "
    "on the sizes where the molecules are, as transcription start sites need (5' data). "
    "For all kernels but the Gaussian and the exponential, the bandwidth is the "
    "kernel's half-width."
)
# No help panel: the commands add theirs.
KdeKernel = Annotated[
    Kernel,
    Parameter(converter=lowercase_choice, help=f"Kernel of the KDE. {KERNELS_HELP}"),
]
KdeKernels = Annotated[
    tuple[Kernel, ...],
    Parameter(
        name="--kde-kernel",
        consume_multiple=1,
        converter=kernels,
        show_default=lambda kernels: " ".join(kernels),
        help=(
            f"Kernel of the KDE. {KERNELS_HELP} With -b optimize, give several to choose "
            "among them too, as in --kde-kernel gaussian linear."
        ),
    ),
]
# No help panel: the commands add theirs.
PeakCaller = Annotated[
    str,
    Parameter(
        converter=peak_caller,
        metavar="NAME",
        help=(
            "Method that collapses each UTR's sizes into sites: kde (the kernel density "
            "estimate of --kde-kernel and the bandwidth), or one that a plugin adds "
            "(see docs/extending.md)."
        ),
    ),
]

Out = Annotated[
    str,
    Parameter(
        name=("--out", "-o"),
        group=OUTPUT,
        help="File to write the count matrix to (.h5ad). The log file goes next to it.",
    ),
]
StoreIntermediate = Annotated[
    bool,
    Parameter(
        name=("--store-intermediate", "-i"),
        group=OUTPUT,
        show_default=False,
        help=(
            "Also store the intermediate files (UTR reference, read mappings) in "
            "./intermediate/, from which `count` can redo the last steps quickly."
        ),
    ),
]
LogLevel = Annotated[
    str,
    Parameter(
        name=("--log-level", "-l"),
        group=OUTPUT,
        help="Level of the messages to show: debug, info, warning or error.",
    ),
]
NoLogFile = Annotated[
    bool,
    Parameter(group=OUTPUT, show_default=False, help="Don't write a log file."),
]
# No help panel: the pipeline commands add theirs, `Annotated[Cores, Parameter(group=OUTPUT)]`.
Cores = Annotated[
    int | None,
    Parameter(
        help=(
            "Maximum number of worker processes. By default, and for 0 or less, all "
            "cores but one. Inside Snakemake, pass the rule's {threads}."
        ),
    ),
]

# Options of the commands that process count matrices, which have no help panels.
UtrNameColumn = Annotated[
    str,
    Parameter(
        name=("--utr-name-column", "-u"),
        help="Column of the count matrix's .var table with the UTR names.",
    ),
]
# The help of the options of bandwidths to score: their defaults.
CANDIDATES_HELP = (
    f"By default, for 3' data {' '.join(map(str, CANDIDATE_BANDWIDTHS['three_prime']))}; "
    f"for 5' data {' '.join(map(str, CANDIDATE_BANDWIDTHS['five_prime']))}. In bash, "
    "{2..20..2} gives 2 4 ... 20."
)
CandidateBandwidths = Annotated[
    tuple[Bandwidth, ...] | None,
    Parameter(
        consume_multiple=1,
        converter=candidate_bandwidths,
        metavar="NT|RULE",
        show_default=False,
        help=(
            "The bandwidths that -b optimize chooses from: numbers of nt, or rules. "
            f"{CANDIDATES_HELP}"
        ),
    ),
]
ScoredBandwidths = Annotated[
    tuple[Bandwidth, ...] | None,
    Parameter(
        name=("--bandwidths", "-b"),
        consume_multiple=1,
        converter=bandwidths,
        metavar="NT|RULE",
        show_default=False,
        help=(
            "The bandwidths to score: numbers of nt, or rules; 0 scores the raw sizes. "
            f"{CANDIDATES_HELP}"
        ),
    ),
]
Bandwidths = Annotated[
    tuple[Bandwidth, ...],
    Parameter(
        name=("--bandwidths", "-b"),
        consume_multiple=1,
        converter=bandwidths_or_optimize,
        metavar="NT|RULE|optimize",
        show_default=lambda bandwidths: " ".join(map(str, bandwidths)),
        help=(
            "One or more bandwidths of the KDE: numbers of nt, or rules. "
            f"{BANDWIDTH_RULES_HELP} {OPTIMIZE_HELP}"
        ),
    ),
]
# The help of the score options: the scores.
SCORES_HELP = (
    "pas (the default for 3' data): molecules at sites 10-30 nt after a "
    "polyadenylation signal, AATAAA. initiator (the default for "
    "5' data): molecules at sites with C or T at -1 and A or G at +1, where RNA "
    "polymerase II starts. known-sites: molecules at sites near the --known-sites, "
    "which rewards smoothing up to the width of the database's clusters. pas and "
    "initiator read the --genome."
)
# No help panel: the commands add theirs.
Scores = Annotated[
    tuple[str, ...] | None,
    Parameter(
        name="--score",
        consume_multiple=1,
        converter=scores,
        metavar="NAME",
        show_default=False,
        help=(
            "How -b optimize scores the candidates, by the sites that look true: "
            f"{SCORES_HELP} With several, the first chooses."
        ),
    ),
]
# No help panel: the pipeline commands add theirs.
Genome = Annotated[
    str | None,
    Parameter(
        help=(
            "Genome sequence (FASTA) of the annotation, for -b optimize with the pas or "
            "initiator score; chromosomes match with or without chr."
        ),
    ),
]
# No help panel: the commands add theirs.
KnownSites = Annotated[
    str | None,
    Parameter(
        help=(
            "Known sites (BED, plain or gzipped) of the annotation's genome assembly, for "
            "the known-sites score: polyadenylation sites such as PolyASite's for 3' "
            "data, transcription start sites such as refTSS's for 5' data."
        ),
    ),
]
