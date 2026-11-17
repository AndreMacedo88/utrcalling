"""`utrcalling tools site-sequences`: the genome sequence around each site."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Literal

from cyclopts import Parameter, Token

# File endings that mean FASTA; any other gives a table.
FASTA_SUFFIXES = (".fa", ".fasta", ".fna")


def length_or_utr(type_: object, tokens: Sequence[Token]) -> int | Literal["utr"]:
    """Read a number of nt, or utr."""
    text = tokens[0].value.strip().lower()
    if text == "utr":
        return "utr"
    try:
        return int(text)
    except ValueError:
        raise ValueError(f"expected a number of nt or utr, got {tokens[0].value!r}") from None


# A number of nt to read on one side of the site, or utr: to the end of its UTR.
Length = Annotated[int | Literal["utr"], Parameter(converter=length_or_utr, metavar="NT|utr")]


def site_sequences(
    input_path: str,
    genome_path: str,
    out_path: str,
    *,
    upstream: Length = 50,
    downstream: Length = 50,
    utr_end: Literal["three_prime", "five_prime"] | None = None,
) -> None:
    """Write the genome sequence around each site, for motif and sequence analyses.

    Positions count from the transcript's end, as each field numbers them. For
    5' data, +1 is the transcription start site and -1 the base before it, so
    --upstream reads the promoter and --downstream the start of the 5' UTR.
    For 3' data, -1 is the cleavage site, the last base of the transcript, so
    --upstream reads the end of the 3' UTR (where the polyadenylation signal
    lies, 10-30 nt before the site) and --downstream the genome after it.
    Sequences are in transcript orientation.

    utr instead of a number reads to the end of the site's UTR on that side.
    The UTR that each site makes, from its start to its end, is --upstream 0
    --downstream utr for 5' sites (the start site to the end of the 5' UTR,
    before the start codon), and --upstream utr --downstream 0 for 3' sites
    (after the stop codon to the cleavage site). The other side gives the part
    of the annotated UTR that the site leaves out. A UTR is one stretch of the
    genome, as utrcalling merged it from the annotation: a UTR spliced into
    several exons is several UTRs (gene, gene_2, ...).

    Writes FASTA if out_path ends in .fa, .fasta or .fna, for motif tools such
    as the MEME suite or HOMER: a record per site, named after it, with the
    upstream and downstream sequence joined (sites with neither are left out).
    Otherwise writes a tab-separated table with a row per site: its gene, UTR,
    UTR size, genome position, molecules, and the upstream and downstream
    sequences apart.

    Works on any UTR-level count matrix, best on sites (peaks), filtered to
    those to study: raw sizes give a row per size.

    For example:

        utrcalling tools site-sequences sites.h5ad GRCm39.fa tss.fa --upstream 10 --downstream 10
        utrcalling tools site-sequences sites.h5ad GRCm39.fa utrs.fa --upstream 0 --downstream utr

    Args:
        input_path: Count matrix (.h5ad) at UTR level.
        genome_path: Genome sequence (FASTA, plain or bgzip-compressed) of the
            annotation the UTRs come from; chromosomes match with or without "chr".
        out_path: Where to write the sequences: .fa, .fasta or .fna for FASTA,
            anything else (such as .tsv) for a table.
        upstream: How many nt to read before the transcript's end (-1 and
            before), or utr.
        downstream: How many nt to read after it (+1 and after), or utr.
        utr_end: Which end of the UTR the sizes count from: three_prime for 3'
            protocols, five_prime for 5' ones. Only for matrices of utrcalling
            before 5.0.0: later ones record it.
    """
    from utrcalling.core import genome, helpers, storage
    from utrcalling.core.module_log import module_logger

    logger = module_logger()
    adata = helpers.check_if_path_and_open(input_path, method="h5ad")
    table = genome.site_sequences(
        adata, genome_path, upstream=upstream, downstream=downstream, utr_end=utr_end
    )
    found = table["upstream"].notna().to_numpy()
    written = found.sum()
    if out_path.lower().endswith(FASTA_SUFFIXES):
        with_sequence = found & (table["upstream"] + table["downstream"]).str.len().gt(0)
        records = (
            (
                f"{site} {row.chrom}:{row.position}({row.strand}) "
                f"upstream={len(row.upstream)} downstream={len(row.downstream)}",
                f"{row.upstream}{row.downstream}",
            )
            for site, row in table[with_sequence.to_numpy()].iterrows()
        )
        storage.write_fasta(records, out_path)
        written = with_sequence.sum()
        if written < found.sum():
            logger.warning(
                f"Left out {found.sum() - written} sites with no sequence: at the end of their UTR"
            )
    else:
        storage.write_table(table, out_path)
    logger.info(f"Wrote the sequences around {written} sites to {out_path}")
    if not found.all():
        logger.warning(
            f"No sequence for {(~found).sum()} sites on chromosomes that the genome lacks"
        )
