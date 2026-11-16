"""Where the sites of a count matrix lie in the genome, and the sequence around them.

Positions are 1-based. Sequences are read in transcript orientation, the
reverse complement on the minus strand, around each site's own base: offset 0
is the site (the first transcribed base for 5' sites, the last for 3' sites),
-1 the base before it in the transcript's direction, +1 the base after it.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pysam

from utrcalling.core import provenance

if TYPE_CHECKING:
    import anndata
    import numpy.typing as npt

UTR_ENDS = ("three_prime", "five_prime")


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
    length = stop - start
    complement = str.maketrans("ACGT", "TGCA")
    with pysam.FastaFile(str(genome)) as fasta:
        names = {
            chrom: next(
                (name for name in (chrom, f"chr{chrom}", chromosome_name(chrom)) if name in fasta),
                None,
            )
            for chrom in sites["chrom"].unique()
        }
        if all(name is None for name in names.values()):
            raise ValueError(
                "No chromosome of the matrix is in the genome: check that both come "
                "from the same genome assembly."
            )
        sequences = []
        for chrom, position, strand in sites[["chrom", "position", "strand"]].itertuples(
            index=False
        ):
            name = names[chrom]
            if name is None:
                sequences.append("N" * length)
                continue
            # 0-based coordinates of the stretch; the minus strand reads leftwards.
            first = position - 1 + start if strand != "-" else position - stop
            last = first + length
            size = fasta.get_reference_length(name)
            sequence = fasta.fetch(name, max(first, 0), min(last, size)).upper()
            sequence = "N" * max(-first, 0) + sequence + "N" * max(last - size, 0)
            sequence = re.sub("[^ACGT]", "N", sequence)
            if strand == "-":
                sequence = sequence.translate(complement)[::-1]
            sequences.append(sequence)
    found = sites["chrom"].map(names).notna().to_numpy()
    return np.array(sequences, dtype=f"<U{length}"), found
