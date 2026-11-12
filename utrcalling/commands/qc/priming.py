"""`utrcalling qc priming`: sites that the genome around them explains as priming artefacts."""

from __future__ import annotations

from typing import Annotated, Literal

from cyclopts import Parameter


def priming(
    input_path: str,
    genome_path: str,
    out_path: str,
    *,
    window: Annotated[int, Parameter(name=("--window", "-w"))] = 20,
    tso: str | None = None,
    max_edit_distance: Annotated[int, Parameter(name=("--max-edit-distance", "-e"))] = 4,
    utr_end: Literal["three_prime", "five_prime"] | None = None,
) -> None:
    """Flag the sites that the genome around them explains as priming artefacts.

    3' data, internal priming: the oligo-dT primer can bind A's that the genome
    encodes inside the UTR, instead of the poly(A) tail. Those molecules end
    just before the A's, like a shorter UTR. A site is flagged when the genome's
    --window nt after it hold 6 A's in a row, or 7 A's in 10 nt (the rule of
    Gautheret et al. 1998).

    5' data, strand invasion: the template-switching oligo (TSO) can bind inside
    the RNA where it resembles the TSO's 3' end, instead of at the cap. Those
    molecules start just after that sequence. A site is flagged when the 9 nt
    before it are within --max-edit-distance of the TSO's last 9 nt, and 2 of
    the 3 nt before it are G's (the rule of Tang et al. 2013).

    Writes a tab-separated table with a row per site: its gene, UTR, UTR size,
    genome position, molecules, the genome sequence checked (on the site's
    strand), the measures and the flag. Also logs the share of sites and of
    molecules flagged.

    A flag means that the genome could explain the site as an artefact, not that
    it is one: weigh it with the known sites (qc known-sites) and the PAS
    hexamer (qc pas-distance). The flags need sites at the molecules' ends: the
    standard_10x, standard_smartseq, truseq and bulk_pe protocols place sites
    where the reads stop, short of the ends, so there the flags say little.

    For example:

        utrcalling qc priming peaks.h5ad GRCh38.fa priming.tsv

    Args:
        input_path: Count matrix (.h5ad) at UTR level.
        genome_path: Genome sequence (FASTA, plain or bgzip-compressed) of the
            annotation the UTRs come from; chromosomes match with or without "chr".
        out_path: Where to write the table (.tsv).
        window: How many nt after a 3' site to look for A's.
        tso: The TSO, or its 3' end, for 5' data: write random bases (a UMI) as
            N, which match any base, and riboguanosines as G. By default, the
            TSO of the matrix's protocol: TTTCTTATATGGG for 5' 10x,
            AGAGACAGATTGCGCAATGNNNNNNNNGGG for Smart-seq3. With a random UMI
            next to the G's, as Smart-seq3 has, any sequence before a site
            resembles some TSO of the pool, so only the G's tell.
        max_edit_distance: Substitutions, insertions and deletions up to which
            the genome before a 5' site resembles the TSO.
        utr_end: Which end of the UTR the sizes count from: three_prime for 3'
            protocols, five_prime for 5' ones. Only for matrices of utrcalling
            before 5.0.0: later ones record it.
    """
    from utrcalling.core import helpers, qc, storage
    from utrcalling.core.module_log import module_logger

    logger = module_logger()
    adata = helpers.check_if_path_and_open(input_path, method="h5ad")
    table = qc.priming_artefacts(
        adata,
        genome_path,
        utr_end=utr_end,
        window=window,
        tso=tso,
        max_edit_distance=max_edit_distance,
    )
    storage.write_table(table, out_path)

    flag = table["internal_priming" if "internal_priming" in table else "strand_invasion"]
    checked = flag.notna().to_numpy()
    flagged = flag.fillna(False).to_numpy(dtype=bool)
    molecules = table.molecules.to_numpy()
    logger.info(
        f"Flagged for {str(flag.name).replace('_', ' ')}: {flagged.sum() / checked.sum():.1%} "
        f"of the sites, {molecules[flagged].sum() / molecules[checked].sum():.1%} of the molecules"
    )
    if not checked.all():
        logger.warning(
            f"Not checked: {(~checked).sum()} sites on chromosomes that the genome lacks"
        )
