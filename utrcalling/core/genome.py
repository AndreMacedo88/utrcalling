"""Where the sites of a count matrix lie in the genome, and the sequence around them.

Positions are 1-based. Sequences are read in transcript orientation, the
reverse complement on the minus strand, around each site's own base: offset 0
is the site (the first transcribed base for 5' sites, the last for 3' sites),
-1 the base before it in the transcript's direction, +1 the base after it.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import pysam

from utrcalling.core import provenance

if TYPE_CHECKING:
    import anndata
    import numpy.typing as npt

UTR_ENDS = ("three_prime", "five_prime")

# A genome FASTA: its path, or the file opened with pysam. (Objects with the
# path as `fn`, such as the pybedtools BedTool of earlier versions, work too.)
GenomeFasta = str | os.PathLike[str] | pysam.FastaFile
# The complement of each base and IUPAC code, keeping lower case (soft-masked
# repeats), as BEDTools complements them.
IUPAC_COMPLEMENT = str.maketrans("ACGTRYKMBVDHSWNacgtrykmbvdhswn", "TGCAYRMKVBHDSWNtgcayrmkvbhdswn")


@contextmanager
def open_genome(genome: GenomeFasta) -> Iterator[pysam.FastaFile]:
    """The genome FASTA, opened with pysam: closed after use, unless it came open."""
    if isinstance(genome, pysam.FastaFile):
        yield genome
        return
    with pysam.FastaFile(str(getattr(genome, "fn", genome))) as fasta:
        yield fasta


def region_sequence(genome: GenomeFasta, chrom: str, start: int, end: int, strand: str) -> str:
    """The genome from *start* to *end* (1-based, inclusive), in the strand's orientation.

    As ``bedtools getfasta -s`` reads it: the reverse complement on the minus
    strand, with the case of the FASTA kept.
    """
    with open_genome(genome) as fasta:
        sequence = fasta.fetch(str(chrom), start - 1, end)
    if strand == "-":
        sequence = sequence.translate(IUPAC_COMPLEMENT)[::-1]
    return sequence


def site_positions(var: pd.DataFrame, utr_end: str) -> npt.NDArray[np.int64]:
    """The genome position (1-based) of each site.

    UTR sizes count from the UTR's start in transcript orientation for 3'
    protocols (``utr_end`` ``three_prime``) and from its end for 5' protocols
    (``five_prime``), as utr_mapping measures them.

    Args:
        var: Site annotations with ``utr_size`` and the UTR's ``start``,
            ``end`` (1-based, inclusive) and ``strand``.
        utr_end: ``three_prime`` or ``five_prime``.

    Returns:
        The positions, in the order of *var*.

    Raises:
        ValueError: If *utr_end* is neither, or the UTR coordinates are not
            numbers, as in gene-level matrices.
    """
    if utr_end not in UTR_ENDS:
        raise ValueError(f"utr_end {utr_end!r}: expected three_prime or five_prime")
    for column in ("start", "end"):
        if not pd.api.types.is_integer_dtype(var[column]):
            raise ValueError(
                "The sites have no genome positions: their UTR coordinates are not "
                "numbers (gene-level matrices join the coordinates of a gene's UTRs)."
            )
    size = var["utr_size"].to_numpy(dtype=np.int64)
    start = var["start"].to_numpy(dtype=np.int64)
    end = var["end"].to_numpy(dtype=np.int64)
    plus = (var["strand"].astype(str) == "+").to_numpy()
    if utr_end == "three_prime":
        return np.where(plus, start + size, end - size)
    return np.where(plus, end - size + 1, start + size - 1)


def positioned_utr_end(adata: anndata.AnnData, utr_end: str | None) -> str:
    """Which end of the UTR the sizes count from: *utr_end*, or what the matrix records.

    Raises:
        ValueError: If neither tells, or if the matrix is at gene level.
    """
    record = provenance.recorded(adata)
    utr_end = utr_end or record.get("utr_end")
    if utr_end is None:
        raise ValueError(
            "The matrix does not record which end of the UTR its sizes count from "
            "(it predates utrcalling 5.0.0): give it, three_prime or five_prime."
        )
    if record.get("summarization_level") == "gene":
        raise ValueError(
            "Gene-level sites have no genome positions: their sizes are positions "
            "along the gene's joined UTRs. Use a UTR-level matrix (-s utr)."
        )
    return utr_end


def site_table(adata: anndata.AnnData, positions: npt.NDArray[np.int64]) -> pd.DataFrame:
    """A row per site, indexed by ``site``: where it lies and its molecules in all cells."""
    var = adata.var
    table = var[[c for c in ("gene_name", "utr_name", "utr_size") if c in var]].copy()
    table["chrom"] = var["chrom"].astype(str).to_numpy()
    table["position"] = positions
    table["strand"] = var["strand"].astype(str).to_numpy()
    table["molecules"] = np.asarray(adata.X.sum(axis=0)).ravel()
    table.index.name = "site"
    return table


def chromosome_name(name: object) -> str:
    """Chromosome names compared without a leading "chr" (chr1 and 1 match)."""
    text = str(name)
    return text[3:] if text.lower().startswith("chr") else text


def chromosome_names(genome: str | Path, chroms: Iterable[str]) -> dict[str, str | None]:
    """The genome's name for each chromosome: the same, or with or without "chr".

    Args:
        genome: Genome FASTA.
        chroms: The chromosomes, as the count matrix names them.

    Returns:
        The genome's name of each chromosome, or None if the genome lacks it.

    Raises:
        ValueError: If the genome has none of them.
    """
    with pysam.FastaFile(str(genome)) as fasta:
        names = {
            chrom: next(
                (name for name in (chrom, f"chr{chrom}", chromosome_name(chrom)) if name in fasta),
                None,
            )
            for chrom in chroms
        }
    if all(name is None for name in names.values()):
        raise ValueError(
            "No chromosome of the matrix is in the genome: check that both come "
            "from the same genome assembly."
        )
    return names


def read_sequences(
    genome: str | Path, sites: pd.DataFrame, start: int, stop: int
) -> tuple[npt.NDArray[np.str_], npt.NDArray[np.bool_]]:
    """The genome from offset *start* up to *stop* (not included) around each site.

    Offsets count in transcript orientation from the site's own base (offset
    0); read_sequences(genome, sites, -1, 1) gives each site's base and the
    one before it, as the transcript reads them.

    Args:
        genome: Genome FASTA (plain or bgzip-compressed; indexed if it has no
            ``.fai``). Chromosomes match with or without "chr".
        sites: Table with ``chrom``, ``position`` (1-based) and ``strand``.
        start: The first offset to read.
        stop: The offset after the last one to read.

    Returns:
        The sequences in upper case, with N for other letters and beyond the
        chromosome's ends (all N on chromosomes that the genome lacks), and
        whether the genome has each site's chromosome.

    Raises:
        ValueError: If *stop* is not after *start*, or if the genome has no
            chromosome of the sites.
    """
    if stop <= start:
        raise ValueError(f"offsets {start} to {stop}: read at least 1 nt")
    sequences, found = read_stretches(
        genome, sites, np.full(len(sites), start), np.full(len(sites), stop)
    )
    return np.array(sequences, dtype=f"<U{stop - start}"), found


def read_stretches(
    genome: str | Path,
    sites: pd.DataFrame,
    starts: npt.NDArray[np.integer[Any]],
    stops: npt.NDArray[np.integer[Any]],
) -> tuple[list[str], npt.NDArray[np.bool_]]:
    """The genome from each site's own offsets *starts* up to *stops* (see :func:`read_sequences`).

    Args:
        genome: Genome FASTA (see :func:`read_sequences`).
        sites: Table with ``chrom``, ``position`` (1-based) and ``strand``.
        starts: Each site's first offset to read.
        stops: Each site's offset after the last one to read: at its start,
            the stretch is empty.

    Returns:
        The sequences, as :func:`read_sequences` gives them, and whether the
        genome has each site's chromosome.

    Raises:
        ValueError: If the genome has no chromosome of the sites.
    """
    complement = str.maketrans("ACGT", "TGCA")
    names = chromosome_names(genome, sites["chrom"].unique())
    sequences = []
    with pysam.FastaFile(str(genome)) as fasta:
        rows = sites[["chrom", "position", "strand"]].itertuples(index=False)
        for (chrom, position, strand), start, stop in zip(rows, starts, stops, strict=True):
            length = int(max(stop - start, 0))
            name = names[chrom]
            if name is None:
                sequences.append("N" * length)
                continue
            # 0-based coordinates of the stretch; the minus strand reads leftwards.
            first = int(position - 1 + start if strand != "-" else position - stop)
            last = first + length
            size = fasta.get_reference_length(name)
            # Within the chromosome; N for the bases before its start and after its end.
            within_first = min(max(first, 0), size)
            within_last = max(min(last, size), within_first)
            sequence = fasta.fetch(name, within_first, within_last).upper()
            sequence = "N" * min(max(-first, 0), length) + sequence
            sequence += "N" * (length - len(sequence))
            sequence = re.sub("[^ACGT]", "N", sequence)
            if strand == "-":
                sequence = sequence.translate(complement)[::-1]
            sequences.append(sequence)
    found = sites["chrom"].map(names).notna().to_numpy()
    return sequences, found


def site_sequences(
    adata: anndata.AnnData,
    genome: str | Path,
    *,
    upstream: int | Literal["utr"] = 50,
    downstream: int | Literal["utr"] = 50,
    utr_end: str | None = None,
) -> pd.DataFrame:
    """The genome sequence around each site, for motif and sequence analyses.

    Positions count from the transcript's end, as each field numbers them.
    For a 5' site (a transcription start site), +1 is the site and -1 the base
    before it, outside the transcript. For a 3' site (a cleavage and
    polyadenylation site), -1 is the site, the transcript's last base, and +1
    the base after it. ``upstream`` holds -*upstream* to -1, and
    ``downstream`` +1 to +*downstream*, in transcript orientation.

    ``utr`` reads to the end of the site's UTR (as utrcalling merged it from
    the annotation's UTR features, one stretch of the genome) on that side.
    So the UTR that each site makes is ``downstream="utr"`` for 5' sites (the
    start site to the end of the 5' UTR) and ``upstream="utr"`` for 3' sites
    (the start of the 3' UTR to the cleavage site); the other side gives the
    part of the annotated UTR that the site leaves out. A UTR spliced into
    several exons is several UTRs here, one per stretch (gene, gene_2, ...).

    Args:
        adata: Count matrix at UTR level (sites or raw sizes), with
            ``utr_name``, ``utr_size``, ``chrom``, ``start``, ``end`` and
            ``strand`` in ``.var``.
        genome: Genome FASTA of the annotation (see :func:`read_sequences`).
        upstream: How many nt to read before the transcript's end, or ``utr``.
        downstream: How many nt to read after it, or ``utr``.
        utr_end: Where the UTR sizes count from (see :func:`site_positions`).
            By default, what the matrix records.

    Returns:
        The columns of :func:`site_table`, then ``upstream`` and
        ``downstream``, missing for sites on chromosomes that the genome lacks.
        With ``utr``, empty where the site lies at or beyond the UTR's end.

    Raises:
        ValueError: If a length is negative or both are 0, if the matrix holds
            fractions, mean sizes or length indices, or for the reasons of
            :func:`positioned_utr_end` and :func:`read_stretches`.
    """
    lengths = [length for length in (upstream, downstream) if length != "utr"]
    if any(length < 0 for length in lengths) or lengths == [0, 0]:
        raise ValueError(
            f"upstream {upstream} and downstream {downstream}: read 0 nt or more on "
            "each side, and 1 or more in all"
        )
    matrix = provenance.recorded(adata).get("matrix")
    if matrix in provenance.MEASURES:
        raise ValueError(
            f"This matrix holds {matrix.replace('_', ' ')}: give the count matrix it comes from."
        )
    utr_end = positioned_utr_end(adata, utr_end)
    positions = site_positions(adata.var, utr_end)
    table = site_table(adata, positions)
    # Offsets count from the site's base; +1 is offset 0 for 5' sites, 1 for 3' sites.
    plus_one = 0 if utr_end == "five_prime" else 1
    # The offsets of the UTR's first and last base, in transcript orientation.
    var = adata.var
    plus = (var["strand"].astype(str) == "+").to_numpy()
    utr_first = np.where(
        plus, var["start"].to_numpy() - positions, positions - var["end"].to_numpy()
    )
    utr_last = np.where(
        plus, var["end"].to_numpy() - positions, positions - var["start"].to_numpy()
    )
    # The offsets to read: from first up to stop, not included.
    if upstream == "utr":
        first = np.minimum(utr_first, plus_one)
    else:
        first = np.full(len(table), plus_one - upstream)
    if downstream == "utr":
        stop = np.maximum(utr_last + 1, plus_one)
    else:
        stop = np.full(len(table), plus_one + downstream)
    sequences, found = read_stretches(genome, table, first, stop)
    # Each sequence splits at +1, plus_one - first characters in.
    splits = plus_one - first
    for column, parts in (
        ("upstream", [sequence[:split] for sequence, split in zip(sequences, splits, strict=True)]),
        (
            "downstream",
            [sequence[split:] for sequence, split in zip(sequences, splits, strict=True)],
        ),
    ):
        values = pd.array(parts, dtype="string")
        values[~found] = pd.NA
        table[column] = values
    return table
