"""`utrcalling stats site-usage`: which UTRs change the use of their sites, and how much."""

from __future__ import annotations

from typing import Annotated, Literal

from cyclopts import Group, Parameter

from utrcalling.commands.options import Cores, UtrNameColumn, lowercase_choice

MODEL = Group("Model", sort_key=0)
REPLICATES = Group("Replicates and permutations", sort_key=1)
EFFECTS = Group("Effect sizes", sort_key=2)
FILTERS = Group("UTRs and precision", sort_key=3)
RESOURCES = Group("Resources", sort_key=4)


def site_usage(
    input_path: str,
    out_path: str,
    *,
    full: Annotated[str, Parameter(group=MODEL, metavar="FORMULA")],
    reduced: Annotated[str, Parameter(group=MODEL, metavar="FORMULA")] = "~ 1",
    engine: Annotated[
        Literal["dirmult", "drimseq"], Parameter(group=MODEL, converter=lowercase_choice)
    ] = "dirmult",
    replicates: Annotated[str | None, Parameter(group=REPLICATES, metavar="COLUMN")] = None,
    permute: Annotated[bool, Parameter(group=REPLICATES, show_default=False)] = False,
    strata: Annotated[str | None, Parameter(group=REPLICATES, metavar="COLUMN")] = None,
    permutations: Annotated[
        tuple[int, ...],
        Parameter(
            group=REPLICATES,
            consume_multiple=1,
            metavar="N",
            show_default=lambda stages: " ".join(map(str, stages)),
        ),
    ] = (49, 199, 999),
    seed: Annotated[int, Parameter(group=REPLICATES)] = 0,
    effect_of: Annotated[str | None, Parameter(group=EFFECTS, metavar="COLUMN")] = None,
    effect_range: Annotated[
        tuple[float, float],
        Parameter(group=EFFECTS, metavar="PERCENTILE", show_default=lambda r: f"{r[0]:g} {r[1]:g}"),
    ] = (10, 90),
    contrast: Annotated[
        tuple[str, str] | None, Parameter(group=EFFECTS, metavar="LEVEL REFERENCE")
    ] = None,
    smooth_graph: Annotated[str | None, Parameter(group=EFFECTS, metavar="KEY")] = None,
    smooth_rounds: Annotated[int, Parameter(group=EFFECTS, metavar="N")] = 2,
    min_units: Annotated[int, Parameter(group=FILTERS, metavar="N")] = 10,
    min_counts: Annotated[int, Parameter(group=FILTERS, metavar="N")] = 10,
    precision_moderation: Annotated[
        Literal["trended", "common", "none"],
        Parameter(group=FILTERS, converter=lowercase_choice),
    ] = "trended",
    utr_name_column: Annotated[UtrNameColumn, Parameter(group=FILTERS)] = "utr_name",
    rscript: Annotated[str, Parameter(group=RESOURCES, metavar="PATH")] = "Rscript",
    cores: Annotated[Cores, Parameter(group=RESOURCES)] = None,
) -> None:
    """Test which UTRs change the use of their sites with a variable, with effect sizes.

    For each UTR with two or more sites, fits a Dirichlet-multinomial
    regression of its site counts on the --full model and on the --reduced
    one, and tests their difference with a likelihood-ratio test. Models are
    R-style formulas of columns of the cell (or sample) annotations: the
    variables the full model adds are tested, any kind (categories, numbers,
    splines such as bs(pseudotime, df=5), interactions). Write C(column) for
    numbers that are categories, such as batch numbers.

    Give site counts (peaks), after removing weak sites (process
    filter-sites): raw sizes have too many sites with too few molecules.

    Cells of one sample are not independent replicates: name the replicate
    of each unit with --replicates. A variable that is the same for all
    cells of a sample (a condition, a genotype) is then tested on one unit
    per sample (process pseudobulk), or with --permute, which swaps it
    between samples. A variable that varies within samples (pseudotime) can
    be tested on cells: --permute then shuffles it within each sample and
    gives a p-value that does not rest on the model's assumptions, while a
    test of per-sample bins (process pseudobulk --bins) checks that the
    change repeats across samples. Which test answers which question, and
    how to prepare the data: https://utrcalling.readthedocs.io/en/latest/site-usage.html

    Writes a tab-separated table with a row per multi-site UTR: its status
    (ok, or why it was not tested), statistic, df, p_value and p_adjusted
    (Benjamini-Hochberg), precision, permutation p-values, and effect sizes
    read from the fitted full model: the length index (from 0, all molecules
    at the shortest site, to 1, all at the longest) at the start and end of
    the contrast, and its change, delta_length_index. `<out>_sites.tsv` gives
    each site's proportions, and `<out>_settings.json` how the test ran.

    For example:

        utrcalling stats site-usage cells.h5ad pseudotime.tsv --full "~ C(batch) + bs(pseudotime, df=5)" --reduced "~ C(batch)" --replicates sample --permute
        utrcalling stats site-usage samples.h5ad condition.tsv --full "~ batch + condition" --reduced "~ batch" --contrast treated control
        utrcalling stats site-usage pairs.h5ad pairs.tsv --full "~ family + diagnosis" --reduced "~ family" --permute --strata family
        utrcalling stats site-usage samples.h5ad drimseq.tsv --full "~ batch + condition" --reduced "~ batch" --engine drimseq

    Args:
        input_path: Count matrix of sites (.h5ad), cells or samples by sites.
        out_path: Where to write the table of UTRs (.tsv).
        full: The full model, as in "~ C(batch) + bs(pseudotime, df=5)".
        reduced: The reduced model, nested in the full one, as in "~ C(batch)".
        engine: dirmult fits the model in utrcalling and suits any number of
            units, cells included. drimseq runs DRIMSeq in R, whose estimate
            of the precision is well tried on few samples; install it with
            `conda install -c conda-forge -c bioconda r-base bioconductor-drimseq`.
        replicates: Column naming each unit's biological replicate, such as
            the sample of each cell.
        permute: Also give a permutation p-value (dirmult engine), which
            shuffles the tested variables within replicates (when they vary
            within them) or between replicates (when they don't), refitting
            everything each time.
        strata: Column within which replicates swap the tested variables,
            such as the family of paired samples.
        permutations: Totals of random permutations after which to stop once
            the observed statistic was reached 5 times. With no more distinct
            permutations than the last total, all are used: the p-value is
            exact.
        seed: Seed of the permutations: the same seed gives the same p-values.
        effect_of: The variable whose effects to measure, by default the one
            the full model adds.
        effect_range: For a numeric variable, the percentiles of its values
            between which effects are read.
        contrast: For a categorical variable, the level to compare and the
            reference level, as in --contrast treated control (by default
            the last level and the first).
        smooth_graph: Key of a graph of neighbouring cells in .obsp, such as
            scanpy's connectivities: the descriptive correlations (not the
            test) average each cell's counts over its neighbours first.
        smooth_rounds: How many times to average over --smooth-graph.
        min_units: Units with molecules that a UTR needs to be tested (for
            cells, 50 or more gives steadier fits).
        min_counts: Molecules that a UTR needs, over all units, to be tested.
        precision_moderation: Shrink each UTR's precision (how much units
            vary around the model) toward the trend over UTRs of similar
            depth, toward their common value, or not at all.
        rscript: R's Rscript program, for --engine drimseq: a name on the
            PATH or a path, such as the Rscript of another conda environment.
    """
    import json
    from pathlib import Path

    from utrcalling.core import helpers, storage
    from utrcalling.core.stats import site_usage as tests

    adata = helpers.check_if_path_and_open(input_path, method="h5ad")
    result = tests.site_usage(
        adata,
        full,
        reduced,
        engine=engine,
        replicates=replicates,
        strata=strata,
        permute=permute,
        permutations=permutations,
        seed=seed,
        effect_of=effect_of,
        effect_range=effect_range,
        contrast=contrast,
        min_units=min_units,
        min_counts=min_counts,
        precision_moderation=precision_moderation,
        smooth_graph=smooth_graph,
        smooth_rounds=smooth_rounds,
        rscript=rscript,
        utr_name_column=utr_name_column,
        n_jobs=cores,
    )
    stem = Path(out_path).with_suffix("")
    storage.write_table(result.genes, out_path)
    storage.write_table(result.sites, f"{stem}_sites.tsv")
    settings = {"input": input_path, **result.settings}
    Path(f"{stem}_settings.json").write_text(json.dumps(settings, indent=2) + "\n")
