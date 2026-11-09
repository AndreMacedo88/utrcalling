"""Command-line options shared by the pipeline commands.

`run`, `map-reads` and `count` take many of the same
options. Each alias below carries the option's flags, help panel and help text,
so the three commands describe them the same way.
"""

from collections.abc import Sequence
from typing import Annotated, Literal, get_args

from cyclopts import Group, Parameter, Token

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
KdeBandwidth = Annotated[
    int,
    Parameter(
        group=PEAK_CALLING,
        help=(
            "Bandwidth of the Gaussian KDE that collapses UTR sizes into peaks. 12 suits "
            "3'-end-enriched assays, where reads pile up within a few bp of the "
            "polyadenylation site. Randomly fragmented libraries spread more per site "
            "and need a wider kernel, or the spread splits into many close pseudo-peaks."
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
Bandwidths = Annotated[
    tuple[int, ...],
    Parameter(
        name=("--bandwidths", "-b"),
        consume_multiple=1,
        show_default=lambda bandwidths: " ".join(map(str, bandwidths)),
        help="One or more bandwidths of the Gaussian KDE.",
    ),
]
