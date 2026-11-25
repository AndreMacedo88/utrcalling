"""Simulated site counts for the statistics tests: units drawn from a Dirichlet-multinomial."""

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse


def site_counts(
    n_units=400,
    n_samples=4,
    n_utrs=12,
    changing=4,
    effect="pseudotime",
    strength=2.0,
    depth=2.0,
    precision=8.0,
    seed=0,
):
    """Site counts of units from *n_samples* samples; the first *changing* UTRs change.

    Each UTR has 2 or 3 sites (alternately). Its units' site proportions
    follow a softmax of per-site baselines plus, for the changing UTRs, a
    per-site slope along ``pseudotime`` (uniform on 0-1) or a shift in the
    ``treated`` condition (the second half of the samples). Units draw their
    proportions from a Dirichlet of that mean and *precision*, then a
    Poisson(*depth*) number of molecules (plus one).

    Returns:
        An AnnData of units x sites, with ``.obs`` columns ``sample``,
        ``batch`` (the sample's number), ``pseudotime``, ``condition`` and
        ``family`` (pairs of samples), and ``.var`` columns ``utr_name``,
        ``utr_size`` and ``gene_name``.
    """
    rng = np.random.default_rng(seed)
    sample = np.repeat(np.arange(n_samples), int(np.ceil(n_units / n_samples)))[:n_units]
    pseudotime = rng.random(n_units)
    treated = sample >= n_samples // 2
    blocks, rows = [], []
    for utr in range(n_utrs):
        n_sites = 2 + utr % 2
        linear = np.tile(rng.normal(0, 0.5, n_sites), (n_units, 1))
        if utr < changing:
            slope = np.linspace(-strength, strength, n_sites)
            driver = pseudotime if effect == "pseudotime" else treated.astype(float)
            linear += np.outer(driver, slope)
        shares = np.exp(linear - linear.max(axis=1, keepdims=True))
        shares /= shares.sum(axis=1, keepdims=True)
        totals = rng.poisson(depth, n_units) + 1
        counts = np.array(
            [
                rng.multinomial(t, rng.dirichlet(precision * p))
                for t, p in zip(totals, shares, strict=True)
            ]
        )
        blocks.append(counts)
        for site in range(n_sites):
            rows.append(
                {"utr_name": f"UTR{utr}", "utr_size": 100 * (site + 1), "gene_name": f"G{utr}"}
            )
    var = pd.DataFrame(rows)
    var.index = [
        f"G{row.utr_name[3:]}_{position}-{row.utr_size}" for position, row in var.iterrows()
    ]
    obs = pd.DataFrame(
        {
            "sample": [f"s{number}" for number in sample],
            "batch": sample,
            "pseudotime": pseudotime,
            "condition": np.where(treated, "treated", "control"),
            "family": [f"f{number % (n_samples // 2)}" for number in sample],
        },
        index=[f"unit{position}" for position in range(n_units)],
    )
    return ad.AnnData(
        X=scipy.sparse.csr_matrix(np.hstack(blocks).astype(np.int32)), obs=obs, var=var
    )


FAKE_RSCRIPT = '''#!{python}
"""A stand-in for Rscript running utrcalling's drimseq.R: writes results shaped like DRIMSeq's."""
import sys
from pathlib import Path

import pandas as pd

mode = "{mode}"
if sys.argv[1] == "-e":
    if mode == "no_drimseq":
        sys.exit("Error: there is no package called 'DRIMSeq'")
    print("1.34.0", end="")
    sys.exit(0)
if mode == "fail":
    sys.exit("Error in dmPrecision: something went wrong")
directory = Path(sys.argv[2])
counts = pd.read_csv(directory / "counts.tsv", sep="\\t")
full = pd.read_csv(directory / "full.tsv", sep="\\t")
reduced = pd.read_csv(directory / "reduced.tsv", sep="\\t")
genes = counts.groupby("gene_id", sort=False).size()
df = (genes - 1) * (full.shape[1] - reduced.shape[1])
lr = [float(3 + 4 * position) for position in range(len(genes))]
pvalues = [10.0 ** -position for position in range(len(genes))]
# The second UTR could not be tested, and the third had no precision.
pvalues[1] = float("nan")
precision = [5.0] * len(genes)
if len(genes) > 2:
    precision[2] = float("nan")
pd.DataFrame(
    {{"gene_id": genes.index, "lr": lr, "df": df.values, "pvalue": pvalues, "adj_pvalue": pvalues}}
).to_csv(directory / "genes.tsv", sep="\\t", index=False)
pd.DataFrame(
    {{
        "gene_id": counts.gene_id,
        "feature_id": counts.feature_id,
        "lr": 1.0,
        "df": 1,
        "pvalue": 0.25,
        "adj_pvalue": 0.5,
    }}
).to_csv(directory / "features.tsv", sep="\\t", index=False)
pd.DataFrame({{"gene_id": genes.index, "genewise_precision": precision}}).to_csv(
    directory / "precision.tsv", sep="\\t", index=False
)
(directory / "settings_seen.tsv").write_text((directory / "settings.tsv").read_text())
pd.DataFrame({{"name": ["R", "DRIMSeq"], "version": ["4.4.3", "1.34.0"]}}).to_csv(
    directory / "versions.tsv", sep="\\t", index=False
)
'''


def fake_rscript(directory, mode="ok"):
    """Write an executable stand-in for Rscript in *directory*; *mode* ok, fail or no_drimseq."""
    import stat
    import sys
    from pathlib import Path

    path = Path(directory) / "Rscript"
    path.write_text(FAKE_RSCRIPT.format(python=sys.executable, mode=mode))
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def real_rscript():
    """An Rscript with DRIMSeq, from $UTRCALLING_TEST_RSCRIPT or the PATH, or None."""
    import os
    import shutil
    import subprocess

    program = os.environ.get("UTRCALLING_TEST_RSCRIPT") or shutil.which("Rscript")
    if program is None:
        return None
    loaded = subprocess.run(
        [program, "-e", "library(DRIMSeq)"], capture_output=True, text=True, check=False
    )
    return program if loaded.returncode == 0 else None
