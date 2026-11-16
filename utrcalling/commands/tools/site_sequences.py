"""`utrcalling tools site-sequences`: the genome sequence around each site."""

from __future__ import annotations

from typing import Literal

# File endings that mean FASTA; any other gives a table.
FASTA_SUFFIXES = (".fa", ".fasta", ".fna")


def site_sequences(
    input_path: str,
    genome_path: str,
    out_path: str,
    *,
    upstream: int = 50,
    downstream: int = 50,
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

    Writes FASTA if out_path ends in .fa, .fasta or .fna, for motif tools such
    as the MEME suite or HOMER: a record per site, named after it, with the
    upstream and downstream sequence joined. Otherwise writes a tab-separated
    table with a row per site: its gene, UTR, UTR size, genome position,
    molecules, and the upstream and downstream sequences apart.

    Works on any UTR-level count matrix, best on sites (peaks), filtered to
    those to study: raw sizes give a row per size.

    For example:

        utrcalling tools site-sequences sites.h5ad GRCm39.fa tss.fa --upstream 10 --downstream 10

    Args:
        input_path: Count matrix (.h5ad) at UTR level.
        genome_path: Genome sequence (FASTA, plain or bgzip-compressed) of the
            annotation the UTRs come from; chromosomes match with or without "chr".
        out_path: Where to write the sequences: .fa, .fasta or .fna for FASTA,
            anything else (such as .tsv) for a table.
        upstream: How many nt to read before the transcript's end (-1 and before).
        downstream: How many nt to read after it (+1 and after).
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
    if out_path.lower().endswith(FASTA_SUFFIXES):
        records = (
            (
                f"{site} {row.chrom}:{row.position}({row.strand}) "
                f"upstream={upstream} downstream={downstream}",
                f"{row.upstream}{row.downstream}",
            )
            for site, row in table[found].iterrows()
        )
        storage.write_fasta(records, out_path)
    else:
        storage.write_table(table, out_path)
    logger.info(f"Wrote the sequences around {found.sum()} sites to {out_path}")
    if not found.all():
        logger.warning(
            f"No sequence for {(~found).sum()} sites on chromosomes that the genome lacks"
        )
