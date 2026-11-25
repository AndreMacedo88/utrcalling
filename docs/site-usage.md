# Differential site usage

Which UTRs change the use of their sites (polyadenylation sites of 3' UTRs,
transcription start sites of 5' UTRs) between conditions, along a trajectory
or between groups of cells, and by how much? This page explains the measures
utrcalling computes, which test answers which question, what each test
assumes, and how to prepare the data for it.

Three commands work together:

- `process pseudobulk` sums the counts of cells into units, one per sample or
  per sample and bin, so that samples, not cells, are the replicates.
- `tools length-index` describes the site use of every cell or sample and
  UTR: its length index and entropy.
- `stats site-usage` tests which UTRs change their site use, and measures how
  much (effect sizes).

The usual name in the literature for differences in the use of isoforms is
differential transcript usage (DTU), a term from alternative splicing.
Here the features are the sites of each UTR, read from the molecules' ends,
so utrcalling calls it differential site usage. It covers alternative
polyadenylation (differential 3' UTR length) and alternative transcription
start sites alike.

## The measures: length index and entropy

A UTR's sites are ranked by UTR size and their ranks scaled from 0 (the
shortest) to 1 (the longest). The **length index** of a cell or sample in a
UTR is the mean scaled rank of its molecules: 0 when all of them are at the
shortest site, 1 when all are at the longest. It is the Weighted UTR Index of
scUTRquant [^fansler2024], after the psi of LABRAT [^goering2021]. For 3' data
a UTR size counts from the stop codon, for 5' data from the start codon, so in
both a larger index means longer UTRs: more distal polyadenylation sites, or
more upstream start sites.

The **entropy** of site use is the Shannon entropy of the sites' shares
divided by its largest value, log K for K sites: 0 when one site takes every
molecule, 1 when all sites take the same share. The two measures tell
different things: a UTR whose molecules move from its middle site to both
ends keeps its length index but gains entropy.

```bash
utrcalling tools length-index sites.h5ad length_index.h5ad
```

This writes a matrix of cells (or samples) by UTRs with the length index in
`.X` and the entropy in `.layers['entropy']`; cells without molecules in a UTR
get NaN. `.obs` adds each cell's mean over its UTRs, `length_index` and
`entropy`: a global length index, which can show cell types or states that
use longer or shorter UTRs overall. Only UTRs with two or more sites have
these measures.

Single cells have one to a few molecules per UTR, so their indices jump
between 0 and 1. `--smooth-graph connectivities` averages each cell's counts
over its neighbours first, in a graph that scanpy's `pp.neighbors` computed
from gene expression, and steadies them for plots:

```bash
utrcalling tools length-index sites.h5ad smoothed.h5ad --smooth-graph connectivities --smooth-rounds 2
```

Smoothed cells share molecules and are not independent: describe them, but
never give p-values to correlations over them. The tests below work on the
counts.

## Choosing the test

The test depends on the **unit of replication**, not on whether the variable
is categorical or numeric. Cells of one sample share the sample's genotype,
environment, dissociation and library: they are not independent replicates
of anything that differs between samples. Ask first:

**Is the tested variable the same for all cells of a sample?** A condition, a
genotype, a diagnosis, a treatment, an age: yes. A pseudotime, a cell state
or a cell type: no, it varies among the cells of each sample.

- **Yes, the same within samples.** Only samples can replicate it. Test one
  unit per sample: sum each sample's cells with `process pseudobulk --by
  sample`, or use bulk data, and run `stats site-usage` on the samples. With
  few samples, `--engine drimseq` estimates the precision with methods made
  for small numbers. Add `--permute` for a p-value that does not rest on the
  model: the samples swap their values (all distinct swaps when they are few,
  which gives an exact p-value). Testing such a variable on cells treats
  thousands of cells as replicates and gives p-values far too small:
  `stats site-usage` refuses it when `--replicates` names the sample, unless
  `--permute` is given, whose permutations then swap the samples whole.
- **No, it varies within samples.** Cells can be tested, with the sample in
  the model (`~ C(sample) + ...`) and named with `--replicates`. `--permute`
  then shuffles the variable among the cells of each sample: its p-value
  needs no assumption about how the cells vary. A trend found this way could
  still come from one sample: check that it repeats across samples by testing
  per-sample bins of the variable (`process pseudobulk --by sample --bins
  pseudotime:3`), which treats samples as replicates.

The tests answer different questions, and agreeing results mean more than
one small p-value:

| Test | Units | A small p-value means | Use it for | It needs |
|---|---|---|---|---|
| Likelihood ratio, chi-squared | cells or samples | the site proportions depend on the tested variables, given the others | screening and ranking UTRs; it is fast | independent units given the model; enough units per parameter |
| `--permute`, within replicates | cells | the proportions follow the variable within samples, without assuming how units vary around the model | checking the chi-squared p-values of cells, whose spread the model may misjudge | the other variables of the model the same within each replicate (such as the sample itself), or the p-value is approximate |
| Likelihood ratio on samples, or `--engine drimseq` | samples, or samples x bins | the change repeats across samples | variables that are the same within samples; checking a trend found in cells | about 5 units per free parameter or more |
| `--permute`, between replicates | samples (or cells, moved by sample) | as above, without the chi-squared approximation | few samples; paired designs with `--strata` | samples exchangeable within strata; enough samples for small p-values (4 samples of two conditions give 6 swaps: a p-value of 1/6 at best) |

### Paired and blocked designs

When samples come in pairs or blocks (two siblings of a family, a patient
before and after treatment, samples processed together), put the block in
both models, `--full "~ family + condition" --reduced "~ family"`, and swap
the values only within blocks with `--permute --strata family`. Nine pairs
give 2^9 = 512 swaps, all of which are used.

### Cell types

A cell type varies within samples, but the cells of a type in one sample are
still not independent. Compare cell types on units of sample x cell type
(`process pseudobulk --by sample cell_type`), with the sample in the model:
`--full "~ sample + cell_type" --reduced "~ sample"`.

## Models as formulas

The models are R-style formulas of columns of the annotations (`.obs`),
read by formulaic [^formulaic]. The variables that the full model adds are
the tested ones; the reduced model must be nested in the full one.

| Formula | Meaning |
|---|---|
| `~ condition` | a level per condition (text columns are categories) |
| `~ C(batch)` | batches numbered 1, 2, 3 are categories, not a slope; without `C()`, a column of a few whole numbers is a number (utrcalling warns) |
| `~ bs(pseudotime, df=5)` | a smooth change: a B-spline of 5 degrees of freedom, as R's `splines::bs` (identical values) |
| `~ pseudotime` | a linear change of the log-odds |
| `~ C(batch) + condition * bs(pseudotime, df=3)` | the trajectory's shape per condition; against `~ C(batch) + condition + bs(pseudotime, df=3)` it tests whether conditions change the trajectory |
| `~ 1` | no variable: the reduced model of a test without covariates |

A spline remembers the knots it placed on all units, so permutations and
effect sizes evaluate the same functions. Units with a missing value in a
variable of the models are left out (such as cells outside the lineage that
has a pseudotime). The degrees of freedom are (columns of the full model -
columns of the reduced one) x (sites - 1), counted for each UTR on its own
units: a batch without molecules in a UTR drops out of its fit.

## Preparing the data

**Do:**

- Test the **site counts** of a peak-called matrix (`call-peaks`, or `run`
  without `-r`), as molecule counts. Raw sizes give each UTR hundreds of
  sizes with a molecule or two each: there is nothing to estimate proportions
  from, and `stats site-usage` refuses them. Fractions, mean sizes and length
  indices are refused too: the model needs counts.
- Remove weak sites first (`process filter-sites`, such as `-t 0.05 -m 10`),
  and keep the multi-site UTRs (`process keep-multi-site`). Sites with a few
  molecules add parameters but no information.
- Call the sites once, on the molecules of all samples you compare:
  concatenate the raw-sizes matrices (`process concatenate`), then call the
  peaks, so that every sample has the same sites.
- Put batch, or the sample, in both models when it is not the tested
  variable.
- Keep enough units for the parameters: each UTR's fit has (columns of the
  full model) x (sites - 1) coefficients. As a rule of thumb, aim for 5 units
  or more per coefficient; for cells, raise `--min-units` to 50.

**Don't:**

- Don't test a variable computed from the same site counts, such as clusters
  or a pseudotime built from site usage: the test is then circular. Compute
  pseudotimes and clusters from gene expression.
- Don't take chi-squared p-values over cells for variables that are the same
  within samples (see above).
- Don't build pseudobulk units that mix samples: include the sample in
  `--by`.
- Don't give p-values to correlations over smoothed cells; the
  `rho_*` columns have none.
- Don't read the sites of standard (read-position) protocols as exact
  cleavage or start sites: their reads end around the site, so sites are
  broader, and neighbouring ones may merge. Their changes in the length index
  are still comparable between conditions.

## Running the test

```bash
# Along a trajectory, in single cells: the sample in the model and as replicate
utrcalling stats site-usage cells.h5ad trajectory.tsv \
    --full "~ C(sample) + bs(pseudotime, df=5)" --reduced "~ C(sample)" \
    --replicates sample --permute --min-units 50 --cores 16

# ... and the check that the trend repeats across samples: per-sample terciles
utrcalling process pseudobulk cells.h5ad terciles.h5ad --by sample --bins pseudotime:3 --min-cells 30
utrcalling stats site-usage terciles.h5ad terciles.tsv \
    --full "~ C(sample) + C(pseudotime_bin)" --reduced "~ C(sample)" --engine drimseq

# Condition against control, on one unit per sample
utrcalling process pseudobulk cells.h5ad samples.h5ad --by sample
utrcalling stats site-usage samples.h5ad condition.tsv \
    --full "~ C(batch) + condition" --reduced "~ C(batch)" --contrast treated control --permute

# Paired samples: two siblings per family
utrcalling stats site-usage samples.h5ad siblings.tsv \
    --full "~ family + diagnosis" --reduced "~ family" --permute --strata family

# Alternative start sites (5' data) between cell types, per sample
utrcalling process pseudobulk cells_5prime.h5ad celltypes.h5ad --by sample cell_type
utrcalling stats site-usage celltypes.h5ad celltypes.tsv \
    --full "~ sample + cell_type" --reduced "~ sample" --contrast neuron progenitor
```

In Python, `utrcalling.core.stats.site_usage.site_usage` takes the same
options and returns the tables (see [](api.md)):

```python
import anndata as ad
from utrcalling.core.stats.site_usage import site_usage

cells = ad.read_h5ad("cells.h5ad")
result = site_usage(
    cells,
    full="~ C(sample) + bs(pseudotime, df=5)",
    reduced="~ C(sample)",
    replicates="sample",
    permute=True,
)
result.genes.sort_values("p_adjusted").head()
```

From R, read the tables (`read.delim("trajectory.tsv")`), or call the Python
function through reticulate.

## The results

`stats site-usage` writes three files. The table you name has a row per UTR
with two or more sites:

| Column | Meaning |
|---|---|
| `gene_name` | the UTR's gene |
| `n_sites`, `n_units`, `total_counts` | sites with molecules, units with molecules, molecules |
| `status` | `ok`; `not_converged` (the fit did not settle, often because a site has no molecules in part of the data: the p-value is less sure, left out of the adjustment, and not permuted); `optimization_failed`; or why it was not tested: `single_site`, `too_few_units`, `too_few_counts`, `not_identifiable` (the UTR's units cannot tell the models apart, such as all in one condition), `not_tested` (DRIMSeq could not test it) |
| `statistic`, `df`, `p_value` | the likelihood ratio, its degrees of freedom, the chi-squared p-value |
| `p_adjusted` | Benjamini-Hochberg adjusted, over the UTRs with status `ok`; use this list of UTRs, all tested ones, as the background of enrichment analyses |
| `precision`, `precision_target` | the precision of the fit (how closely units follow the model: the dispersion is 1 / (1 + precision)) and the value it was shrunk toward |
| `permutation_p_value`, `permutation_p_adjusted`, `permutations`, `permutation_exact` | with `--permute` |
| `length_index_from`, `length_index_to`, `delta_length_index` | the length index at the start and end of the contrast, and its change, from -1 to 1 |
| `length_index_range`, `min_at`, `max_at`, `non_monotonicity` | for a numeric variable: how far the index moves along it (signed: positive when its highest point comes after its lowest), where, and how far from monotonic the change is (0 monotonic, near 1 when it rises and returns) |
| `tvd`, `max_abs_delta_proportion` | the total variation distance between the start and end proportions (half the sum of the changes of all sites), and the largest change of one site |
| `rho_length_index`, `rho_entropy` | for a numeric variable: Spearman correlation of each unit's length index and entropy with it, from the counts (or `--smooth-graph`), without p-values |
| `observed_delta_length_index`, `observed_delta_entropy` | for a categorical variable: the difference of the units' mean length index and entropy, without a model |
| `max_variance_ratio`, `min_variance_ratio` | how much more the units spread around the fit than the precision allows (near 1 for a good fit; much above 1, the chi-squared p-value is too small) |

`<out>_sites.tsv` gives each site's UTR, size, scaled rank, and its
proportions at the start and end of the contrast; with `--engine drimseq`,
also DRIMSeq's test of the site against the rest of its UTR.
`<out>_settings.json` records the version, the command line, the models, the
tested variables, the contrast, the units, and how the permutations ran.

### Effect sizes

The effect sizes come from the fitted full model, the model that is tested.
At each point of the contrast (two levels of a category; 50 points between
two percentiles of a numeric variable, by default its 10th and 90th) every
unit of the UTR gets that value, the model predicts its proportions, and
they are averaged over the units: the proportions the UTR's units would have
with that value, the other variables staying as they are. `--effect-of`
chooses the variable when the full model adds several, `--contrast LEVEL
REFERENCE` the levels compared (by default the last level against the first),
and `--effect-range` the percentiles.

## How the test works

For each UTR with two or more sites, the counts of its K sites in a unit
follow a Dirichlet-multinomial: a multinomial whose proportions vary between
units around their mean with a spread set by the precision. The mean
proportions follow the design through a softmax link (log-odds of each site
against the longest). The model is fitted by maximum likelihood with analytic
gradients; the full model estimates the precision, and the reduced model is
fitted at the same precision, as DRIMSeq does [^nowicka2016]. The likelihood
ratio is chi-squared with the degrees of freedom above.

The precision of UTRs with few or shallow units is unsure. It is shrunk
toward the trend of moment estimates against depth over all UTRs
(`--precision-moderation trended`), toward their common value (`common`),
or not at all (`none`), by a penalty that fades as units accumulate.

Permutations refit everything for each shuffle, the precision included: the
p-value is that of the same statistic. Random permutations come in stages
(by default 49, then 199, then 999 in all), stopping once the observed
statistic has been reached 5 times, so UTRs with clearly large p-values stop
early; p = (b + 1) / (B + 1) for b exceedances in B permutations
[^phipson2010]. Each UTR has its own random numbers, from `--seed`, so results
do not depend on `--cores`.

### The engines

- `dirmult` (the default) is utrcalling's own fit. It works on any units,
  cells included, and on all cores: it tests about 8,000 UTRs over 14,000
  cells in under a minute on 30 cores (more with permutations, which refit
  hundreds of times per UTR).
- `drimseq` runs DRIMSeq [^nowicka2016] in R, on the design matrices that
  utrcalling builds from the formulas, so both engines fit the same models.
  DRIMSeq estimates the precision from the Cox-Reid adjusted profile
  likelihood with weighted-likelihood moderation, well tried on few samples.
  Its fits are slow on many units: use it for samples, not cells. The effect
  sizes come from utrcalling's fit of the full model at DRIMSeq's precision.
  The sites table adds DRIMSeq's beta-binomial test of each site.

DRIMSeq is an optional dependency. Install R and DRIMSeq in your environment:

```bash
conda install -c conda-forge -c bioconda r-base bioconductor-drimseq
```

or point `--rscript` at the `Rscript` of another environment that has it,
such as `--rscript ~/miniforge3/envs/r/bin/Rscript`. utrcalling checks for it
before any work. DRIMSeq is free software under the GPL (version 3 or later);
utrcalling only runs it as a separate program and neither includes nor copies
its code.

## References

[^fansler2024]: Fansler MM, Mitschka S, Mayr C (2024). Quantifying 3' UTR
    length from scRNA-seq data reveals changes independent of gene expression.
    Nat Commun 15:4050.

[^goering2021]: Goering R, Engel KL, Gutierrez-Gonzalez A, Taliaferro JM (2021).
    LABRAT reveals association of alternative polyadenylation with
    transcript localization, RNA binding protein expression, transcription
    speed, and cancer survival. BMC Genomics 22:476.

[^formulaic]: Wardrop M. formulaic: a high-performance implementation of
    Wilkinson formulas for Python. <https://matthewwardrop.github.io/formulaic/>

[^nowicka2016]: Nowicka M, Robinson MD (2016). DRIMSeq: a Dirichlet-multinomial
    framework for multivariate count outcomes in genomics. F1000Research 5:1356.

[^phipson2010]: Phipson B, Smyth GK (2010). Permutation p-values should never be
    zero: calculating exact p-values when permutations are randomly drawn.
    Stat Appl Genet Mol Biol 9:39.

Pseudobulk testing of single-cell data: Squair JW et al. (2021). Confronting
false discoveries in single-cell differential expression. Nat Commun 12:5692.
Permutations with exchangeability blocks: Winkler AM et al. (2015). Multi-level
block permutation. NeuroImage 123:253-268.
