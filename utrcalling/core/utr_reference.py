"""The UTRs of a GTF annotation, merged into the UTR reference that reads are mapped to.

Reads GTF files and merges overlapping regions with pandas, exactly as
utrcalling did with gtfparse and BEDTools (``bedtools merge -s``) before
version 5.1.0: the same UTRs, names and columns.
"""

from __future__ import annotations

import gzip
import heapq
import io

import numpy as np
import pandas as pd

from utrcalling.core.module_log import module_logger

# The columns of a GTF file before its attributes.
GTF_COLUMNS = ["seqname", "source", "feature", "start", "end", "score", "strand", "frame"]
# The attributes of each UTR that the reference keeps.
UTR_ATTRIBUTES = ("gene_id", "transcript_id", "gene_name")
# The columns of a merged region, as BED files name them (see merge_regions).
BED_COLUMNS = ["chrom", "start", "end", "name", "score", "strand", "thickStart", "thickEnd"]


def read_gtf(gtf_file: str, feature: str | None = None) -> pd.DataFrame:
    """The rows of a GTF file, with each attribute as a column, as gtfparse 1.x read them.

    Args:
        gtf_file: Path to a GTF annotation file (plain or gzipped).
        feature: Only the rows of this feature type, such as ``three_prime_utr``:
            much faster on whole-genome annotations, whose other lines are
            skipped unread.

    Returns:
        A row per feature, with the GTF columns (``seqname``, ``source``,
        ``feature``, ``start`` and ``end`` (1-based, inclusive), ``score``
        (float32), ``strand`` and ``frame`` (0 for ".")), then a column per
        attribute in the order they first appear, as text: "" for rows
        without it, the values of an attribute given twice joined by ",".
    """
    logger = module_logger()
    logger.info("Reading the GTF file")
    opener = gzip.open if str(gtf_file).endswith(".gz") else open
    with opener(gtf_file, "rt") as gtf:
        if feature is None:
            text = gtf.read()
        else:
            marker = f"\t{feature}\t"
            text = "".join(line for line in gtf if marker in line)
    table = pd.read_csv(
        io.StringIO(text),
        sep="\t",
        comment="#",
        names=[*GTF_COLUMNS, "attribute"],
        skipinitialspace=True,
        skip_blank_lines=True,
        dtype={"start": np.int64, "end": np.int64, "score": np.float32, "seqname": str},
        na_values=".",
        converters={"frame": lambda frame: 0 if frame == "." else int(frame)},
    )
    for column in ("seqname", "source", "strand", "frame"):
        table[column] = [str(value) for value in table[column]]
    if feature is not None:
        table = table[table["feature"] == feature].reset_index(drop=True)
    attributes = _expand_attributes(table["attribute"])
    table = table.drop(columns="attribute")
    for name, values in attributes.items():
        table[name] = values
    logger.info("Finished reading the GTF file")
    return table


def _expand_attributes(attribute_strings: pd.Series) -> dict[str, list[str]]:
    """Each attribute's values, from GTF attribute strings such as 'gene_id "G1"; gene_name "A";'.

    As gtfparse 1.x expands them: fields split at ";", a field's name and value
    are its first two words, quotes around the value are dropped, a missing
    attribute is "", and an attribute given twice in a row has its values
    joined by ",". (gtfparse also repairs ';"' and ';-' inside values, from
    some Ensembl releases.)
    """
    columns: dict[str, list[str]] = {}
    count = len(attribute_strings)
    for row, text in enumerate(attribute_strings):
        text = text.replace(';"', '"').replace(";-", "-")
        for field in text.split(";"):
            parts = field.strip().split(" ", 2)[:2]
            if len(parts) != 2:
                continue
            name, value = parts
            if value.startswith('"'):
                value = value.replace('"', "")
            if name not in columns:
                columns[name] = [""] * count
            column = columns[name]
            column[row] = value if column[row] == "" else f"{column[row]},{value}"
    return columns


def open_gtf(gtf_file: str) -> pd.DataFrame:
    """Parse a GTF annotation file to a pandas DataFrame (see :func:`read_gtf`).

    Args:
        gtf_file: Path to a GTF annotation file (plain or gzipped).

    Returns:
        A DataFrame with all features from the GTF file.
    """
    return read_gtf(gtf_file)


def extract_feature(gtf_df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """Extract rows matching a specific feature type from a GTF DataFrame.

    Args:
        gtf_df: GTF annotation DataFrame.
        feature: Feature type to filter (e.g., 'three_prime_utr', 'five_prime_utr').

    Returns:
        A filtered DataFrame containing only rows with the specified feature.
    """
    logger = module_logger()
    logger.info("Filtering GTF reference file to retrieve only the UTR annotations")

    utr_df = gtf_df[gtf_df["feature"] == feature]
    return utr_df


def prepare_annotations_index(
    annotations_extra: list[str],
) -> tuple[list[str], list[int]]:
    """Build column list and index positions for BED merge operations.

    Args:
        annotations_extra: List of additional annotation column names beyond
            the standard seqname/start/end.

    Returns:
        A (columns, indices) tuple where columns is the full column list and
        indices are 1-based positions for the extra annotations.
    """
    main = ["seqname", "start", "end"]
    annotations_columns = main + annotations_extra
    annotations_index = list(range(len(main) + 1, len(annotations_columns) + 1))
    return annotations_columns, annotations_index


def merge_regions(regions: pd.DataFrame) -> pd.DataFrame:
    """Merge the overlapping regions of each strand, exactly as ``bedtools merge -s -o distinct`` does.

    A port of the merge of BEDTools 2.31 (FileRecordMergeMgr), which built
    utrcalling's UTR reference before version 5.1.0, so the reference stays
    the same. The regions are read in order, as BED records: chromosome,
    start and end (here the GTF's 1-based start, as utrcalling always wrote
    it, and its end), then annotations, the sixth being the strand. A region
    whose start equals its end counts as 1 nt wider on each side. Each merged
    region begins with the record of the lowest start among those set aside
    so far (from the other strand, or past the merged region before), or else
    with the next record; it takes the following records of its strand that
    start at or before its furthest end, so records that only touch join too.
    Records without a strand (".") are left out.

    Args:
        regions: The regions, sorted by chromosome (by name, as text) and
            start, as bedtools needs them, with chrom, start and end first, the
            strand sixth, and other annotations.

    Returns:
        A merged region per row, in the order bedtools writes them, with the
        columns of BED_COLUMNS: chromosome, the first record's start, the
        furthest end (a lone region keeps its own coordinates), then each
        annotation as the comma-separated distinct values of its records,
        sorted as text. The columns' types are those pandas reads from text,
        as utrcalling read BEDTools' output before.
    """
    module_logger().info("Merging overlapping UTR regions")
    # The values as text, as in a BED file, with "." for missing ones.
    text = regions.to_csv(sep="\t", header=False, na_rep=".", index=False)
    bed = pd.read_csv(io.StringIO(text), sep="\t", header=None, dtype=str, keep_default_na=False)
    chroms = bed[0].tolist()
    strands = bed[5].tolist()
    starts = bed[1].astype(np.int64).tolist()
    ends = bed[2].astype(np.int64).tolist()
    zero_length = [start == end for start, end in zip(starts, ends, strict=True)]
    wide_starts = [start - zero for start, zero in zip(starts, zero_length, strict=True)]
    wide_ends = [end + zero for end, zero in zip(ends, zero_length, strict=True)]
    chromosome_order: dict[str, int] = {}
    for chrom in chroms:
        chromosome_order.setdefault(chrom, len(chromosome_order))

    # Records set aside, per strand: heaps by chromosome, start and end.
    stored: dict[str, list[tuple[int, int, int, int]]] = {"+": [], "-": []}

    def store(record: int) -> None:
        if strands[record] in stored:  # records without a strand are dropped
            key = (chromosome_order[chroms[record]], wide_starts[record], wide_ends[record], record)
            heapq.heappush(stored[strands[record]], key)

    def take_stored(strand: str | None = None) -> int | None:
        """The stored record of the lowest position, of *strand* or of either."""
        heaps = [stored[strand]] if strand is not None else [stored["+"], stored["-"]]
        heaps = [heap for heap in heaps if heap]
        if not heaps:
            return None
        lowest = min(heaps, key=lambda heap: heap[0][:3])  # "+" first at equal positions
        return heapq.heappop(lowest)[3]

    unread = iter(range(len(bed)))
    groups: list[list[int]] = []
    while True:
        first = take_stored()
        while first is None:
            first = next(unread, None)
            if first is None:
                break
            if strands[first] not in stored:
                first = None
        if first is None:
            break
        group, furthest = [first], wide_ends[first]
        while True:
            record = take_stored(strands[first])
            from_storage = record is not None
            if record is None:
                record = next(unread, None)
            if record is None:
                break
            if chroms[record] != chroms[first] or wide_starts[record] > furthest:
                store(record)
                break
            if strands[record] not in stored:
                continue
            if not from_storage and strands[record] != strands[first]:
                store(record)
                continue
            group.append(record)
            furthest = max(furthest, wide_ends[record])
        groups.append(group)

    annotations = [bed[column].tolist() for column in range(3, bed.shape[1])]
    rows = []
    for group in groups:
        first = group[0]
        if len(group) == 1:
            row = [chroms[first], starts[first], ends[first]]
        else:
            row = [chroms[first], wide_starts[first], max(wide_ends[record] for record in group)]
        for values in annotations:
            row.append(",".join(sorted({values[record] for record in group})))
        rows.append(row)
    text = pd.DataFrame(rows).to_csv(sep="\t", header=False, index=False)
    return pd.read_csv(io.StringIO(text), sep="\t", names=BED_COLUMNS[: bed.shape[1]])


def generate_utr_name(utr_df: pd.DataFrame) -> pd.DataFrame:
    """Generate unique UTR names by appending occurrence counts.

    For UTRs with the same gene-level name, appends '_2', '_3', etc.
    to distinguish them. Sets 'utr_name' as the DataFrame index.

    Uses vectorized groupby+cumcount instead of row-by-row iteration.

    Args:
        utr_df: A UTR DataFrame with a 'name' column.

    Returns:
        The DataFrame with a new 'utr_name' index column.

    Raises:
        TypeError: If input is not a DataFrame.
    """
    if not isinstance(utr_df, pd.DataFrame):
        raise TypeError(f"Please turn your {type(utr_df)} object to a pandas DataFrame")
    # sort by name so that name incrementation is handled correctly later on.
    utr_df = utr_df.sort_values(by=["chrom", "start", "end"])

    # Vectorized: compute occurrence number within each name group
    occurrence = utr_df.groupby("name", sort=False).cumcount()
    # First occurrence (0) gets no suffix; subsequent ones get _2, _3, ...
    suffix = occurrence.map(lambda x: f"_{x + 1}" if x > 0 else "")
    utr_df["utr_name"] = utr_df["name"].astype(str) + suffix

    utr_df = utr_df.set_index("utr_name")
    return utr_df


def correct_column_names(utr_df: pd.DataFrame) -> pd.DataFrame:
    """Rename BED thickStart/thickEnd columns to transcript_id/gene_name.

    Args:
        utr_df: A UTR DataFrame from BED parsing.

    Returns:
        The DataFrame with renamed columns.
    """
    utr_df = utr_df.rename(columns={"thickStart": "transcript_id", "thickEnd": "gene_name"})
    return utr_df


def subset_reference_data(
    utr_sizes: pd.DataFrame, utr_reference_regions: pd.DataFrame
) -> pd.DataFrame:
    """Subset the UTR reference to only UTRs present in the read data.

    Args:
        utr_sizes: Flat DataFrame with a 'utr_name' column.
        utr_reference_regions: Full UTR reference DataFrame indexed by utr_name.

    Returns:
        A deduplicated subset of the reference containing only matching UTRs.
    """
    logger = module_logger()
    logger.info("Subsetting UTR reference data")

    utr_names = utr_sizes["utr_name"].unique()
    sub_utr_reference = utr_reference_regions[utr_reference_regions.index.isin(utr_names)]
    sub_utr_reference_unique = sub_utr_reference.drop_duplicates()

    return sub_utr_reference_unique
