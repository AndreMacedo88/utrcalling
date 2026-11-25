"""The bridge to DRIMSeq, in R: the same test with DRIMSeq's estimate of the precision.

DRIMSeq (Nowicka and Robinson 2016, F1000Research 5:1356) fits the same
Dirichlet-multinomial model, with the precision estimated from the Cox-Reid
adjusted profile likelihood and moderated by weighted likelihood: well tried
on few samples. It is a Bioconductor package under the GPL (version 3 or
later), which users install themselves; utrcalling only runs it, with
Rscript, as a separate program: it neither includes nor copies its code.

utrcalling writes the counts and the design matrices it built from the
formulas to a temporary directory, runs ``drimseq.R`` on it, and reads the
results back, so both engines fit exactly the same models.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

SCRIPT = Path(__file__).with_name("drimseq.R")
INSTALL = "conda install -c conda-forge -c bioconda r-base bioconductor-drimseq"


def check(rscript: str = "Rscript") -> str:
    """Check that R and DRIMSeq can run, before any work.

    Args:
        rscript: The Rscript program: a name on the PATH, or a path (such as
            the Rscript of another conda environment).

    Returns:
        DRIMSeq's version.

    Raises:
        RuntimeError: If Rscript cannot be found, or DRIMSeq not loaded.
    """
    program = shutil.which(rscript)
    if program is None:
        raise RuntimeError(
            f"Could not find R's {rscript}. The drimseq engine runs DRIMSeq in R: install "
            f"both in this environment with `{INSTALL}`, or pass --rscript with the path of "
            "an Rscript that has DRIMSeq."
        )
    loaded = subprocess.run(
        [program, "-e", 'cat(as.character(packageVersion("DRIMSeq")))'],
        capture_output=True,
        text=True,
        check=False,
    )
    if loaded.returncode != 0:
        raise RuntimeError(
            f"R ({program}) could not load DRIMSeq. Install it with `{INSTALL}`, or "
            "pass --rscript with an Rscript that has it."
        )
    return loaded.stdout.strip()


@dataclass(frozen=True)
class DrimseqResult:
    """What DRIMSeq found.

    Attributes:
        genes: Per UTR: ``statistic`` (the likelihood ratio), ``df``,
            ``p_value`` and ``precision``; NaN where DRIMSeq could not test it.
        sites: Per site: ``statistic`` and ``p_value`` of DRIMSeq's
            beta-binomial test of the site against the others of its UTR.
        versions: The versions of R and DRIMSeq.
    """

    genes: pd.DataFrame
    sites: pd.DataFrame
    versions: dict[str, str]


def run(
    utrs: Sequence[str],
    sites: Sequence[Sequence[str]],
    counts: Sequence[npt.NDArray[np.float64]],
    full: npt.NDArray[np.float64],
    reduced: npt.NDArray[np.float64],
    *,
    moderation: str = "trended",
    cores: int = 1,
    seed: int = 0,
    rscript: str = "Rscript",
) -> DrimseqResult:
    """Test the UTRs with DRIMSeq.

    Args:
        utrs: The UTRs to test.
        sites: The sites of each UTR.
        counts: The counts of each UTR, units x its sites (every unit of the
            design, with or without molecules).
        full: The full design matrix, with linearly independent columns.
        reduced: The reduced design matrix, likewise.
        moderation: DRIMSeq's moderation of the precision: trended, common or none.
        cores: Worker processes for DRIMSeq.
        seed: R's random seed.
        rscript: The Rscript program.

    Returns:
        DRIMSeq's results, by UTR and site name.

    Raises:
        RuntimeError: If DRIMSeq fails.
    """
    from utrcalling.core.module_log import module_logger

    n_units = full.shape[0]
    units = [f"u{position}" for position in range(n_units)]
    gene_ids = np.repeat([f"g{position}" for position in range(len(utrs))], [len(s) for s in sites])
    feature_ids = [f"f{position}" for position in range(len(gene_ids))]
    table = pd.DataFrame(np.vstack([block.T for block in counts]).astype(np.int64), columns=units)
    table.insert(0, "feature_id", feature_ids)
    table.insert(0, "gene_id", gene_ids)
    with tempfile.TemporaryDirectory(prefix="utrcalling-drimseq-") as folder:
        directory = Path(folder)
        table.to_csv(directory / "counts.tsv", sep="\t", index=False)
        pd.DataFrame({"sample_id": units}).to_csv(directory / "samples.tsv", sep="\t", index=False)
        for name, matrix in (("full", full), ("reduced", reduced)):
            columns = [f"c{position}" for position in range(matrix.shape[1])]
            pd.DataFrame(matrix, columns=columns).to_csv(
                directory / f"{name}.tsv", sep="\t", index=False
            )
        pd.DataFrame(
            {"key": ["moderation", "cores", "seed"], "value": [moderation, cores, seed]}
        ).to_csv(directory / "settings.tsv", sep="\t", index=False)
        module_logger().info(f"Running DRIMSeq on {len(utrs)} UTRs and {n_units} units")
        finished = subprocess.run(
            [rscript, str(SCRIPT), str(directory)], capture_output=True, text=True, check=False
        )
        if finished.returncode != 0:
            raise RuntimeError(f"DRIMSeq failed:\n{finished.stderr[-3000:]}")
        genes = pd.read_csv(directory / "genes.tsv", sep="\t")
        features = pd.read_csv(directory / "features.tsv", sep="\t")
        precision = pd.read_csv(directory / "precision.tsv", sep="\t")
        versions = pd.read_csv(directory / "versions.tsv", sep="\t", dtype=str)

    utr_of = dict(zip((f"g{position}" for position in range(len(utrs))), utrs, strict=True))
    site_of = dict(zip(feature_ids, (site for names in sites for site in names), strict=True))
    by_gene = genes.set_index("gene_id").rename(index=utr_of)
    gene_table = pd.DataFrame(
        {
            "statistic": by_gene["lr"],
            "df": by_gene["df"],
            "p_value": by_gene["pvalue"],
            "precision": precision.set_index("gene_id")["genewise_precision"].rename(index=utr_of),
        }
    ).reindex(list(utrs))
    by_site = features.set_index("feature_id").rename(index=site_of)
    site_table = pd.DataFrame({"statistic": by_site["lr"], "p_value": by_site["pvalue"]})
    return DrimseqResult(
        gene_table, site_table, dict(zip(versions["name"], versions["version"], strict=True))
    )
