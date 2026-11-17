# Changelog

All notable changes to utrcalling. Versions follow [semantic versioning](https://semver.org):
a new major version can change results or require changes to your command lines.

## [Unreleased]

### Changed (breaking)

- **Commands renamed and regrouped**, so that each name says what the command
  does. The pipeline commands are `run`, `map-reads` and `count` (`run` in two
  steps) and `call-peaks`. `process` prepares the count matrix and the new
  `tools` group computes the measures to analyse from it, in the order of an
  analysis, like scanpy's `pp` and `tl`:

  | 4.0.0 | Now |
  | --- | --- |
  | `prepare-reference-and-map-reads` | `map-reads` |
  | `call-peaks` | `count` |
  | `bandwidth call-peaks` | `call-peaks` |
  | `process filter` | `process filter-sites` |
  | `process select-alternative` | `process keep-multi-site` |
  | `process rename` | `process rename-sites` |
  | `process to-fractions` | `tools to-fractions` |
  | `process mean-sizes` | `tools mean-sizes` |
  | `bandwidth score` | `qc score-bandwidths` |

  The names of 4.0.0 no longer work. The names from before 4.0.0 still do,
  as in the paper's workflows: `call-peaks` with read mappings (`-c`) runs
  `count`.
- The command modules moved from `utrcalling.tools` to `utrcalling.commands`,
  with a subpackage per group.
- `qc pas-distance` and `qc score-bandwidths` write tab-separated tables
  instead of joblib files, which only Python could open. `pas-distance` has a
  row per site: its molecules, nearest hexamer and distance (weight the
  distances by the molecules for their distribution over molecules).
  `score-bandwidths` has a row per bandwidth: its kernel, score, sites and
  sites per UTR, and takes `-H/--hamming` (1 by default, as before). By
  default it scores 4, 6, 8, 10, 12, 15 and 20 nt for 3' data (before: 1, 7,
  12 and 15). In Python,
  `peak_calling_anndata.pas_distance_table()` and `score_bandwidths()` return
  these tables.

### Added

- **Peak calling for 5' data.** Transcription start sites are exact to the
  nucleotide, and conditions can shift them by a few nt, which the 3' default
  (a Gaussian of 12 nt) joins. The README describes how to choose the peak
  calling of 5' data by the initiator at its peaks, and the settings that
  Eugenia Guerrero Aruffo found best for interferon-treated neural stem cells
  in her master's thesis (2026): the linear kernel, half of Scott's bandwidth
  per UTR, UTRs with 500 molecules or more, and sites filtered per treatment.
  The pieces follow.
- **Kernels for peak calling.** `--kde-kernel` chooses the kernel of the KDE
  that calls the peaks, in `run`, `count`, `call-peaks`, `qc peaks` and
  `qc score-bandwidths`: `gaussian` (the default, as before), `linear`,
  `epanechnikov`, `exponential`, `tophat` or `cosine`, as scikit-learn defines
  them. The linear kernel keeps the peaks on the sizes where the molecules
  are, as transcription start sites need: Eugenia Guerrero Aruffo chose it for
  5' Cap-seq data in her master's thesis (2026). For all kernels but the
  Gaussian and the exponential, the bandwidth is the kernel's half-width. The
  count matrices record the kernel, and `qc report` shows it.
- **Bandwidth rules.** Besides a number of nt, the bandwidth of `run` and
  `count` (`--kde-bandwidth`, now also `-b`) and of `call-peaks`, `qc peaks`
  and `qc score-bandwidths` (`-b`) can be a rule that sets each UTR's from the
  spread of its molecules' sizes, narrower as they grow in number: `scott`
  (SD x n^-1/5, the default of SciPy's gaussian_kde) or `silverman` (0.9 x
  min(SD, IQR/1.34) x n^-1/5, as R's `bw.nrd0`, which several sites in a UTR
  inflate less), optionally scaled, as in `scott:0.5`. The sites keep their
  UTR's bandwidth in `.var["bandwidth"]`, and `qc peaks` shows it. `call-peaks`
  names its files after the rule (`peaks_bwscott0.5.h5ad`), and takes
  bandwidths that are not whole numbers (`-b 7.5`). For 5' Cap-seq data of
  neural stem cells treated with interferon, Eugenia Guerrero Aruffo found
  `scott:0.5` with the linear kernel best.
- **A bandwidth score for 5' data.** `qc score-bandwidths` scores the
  peaks of 5' data by the initiator: per UTR, the share of molecules at
  frequent peaks with a pyrimidine at -1 and a purine at +1 (the start site),
  where RNA polymerase II starts most transcripts. Smoothing that moves a peak
  off the start site loses it: on 5' Cap-seq data, 42% of the raw molecules
  start at such a pair, 22% of the Gaussian peaks of 12 nt, and 60% of the
  linear peaks of about 2 nt. Eugenia Guerrero Aruffo proposed this measure
  in her master's thesis (2026). The command picks the score by the matrix's
  UTR end (`--utr-end` for matrices before 5.0.0, which count as 3'), scores
  1 2 3 5 8 12 nt and Scott's and Silverman's rules by default for 5' data,
  and takes `--kde-kernel`. For 3' data it now searches the PAS hexamers of
  each UTR once, not once per bandwidth, and matches chromosomes with or
  without "chr", as the 5' score does; UTRs on chromosomes that the genome
  lacks no longer crash it, but are left out.
- **`-b optimize`** in `run`, `count` and `call-peaks` chooses the bandwidth
  for the data: it scores the `--candidates` (by default those of
  `qc score-bandwidths` for the data's UTR end) against the `--genome`, calls
  the peaks with the best, and writes the scores to
  `<out>_bandwidth_scores.tsv`; the matrix records them. `run` checks the
  genome before mapping any read. The default bandwidth stays 12 nt.
- **`process filter-sites -g COLUMN`** counts per group of cells (a column
  of the annotations, such as a treatment) and keeps a site that passes the
  threshold in at least one group: a site that a condition raises stays,
  though it is a small share of all counts. Eugenia Guerrero Aruffo filtered
  5' sites so, by IFN treatment, with `-t 0.1 -m 10`. Without `-g`, nothing
  changes.
- **`process filter-utrs -m N`** keeps the UTRs with at least N molecules
  over all cells, whose sites are surer. Peak calling keeps every molecule of
  a UTR, so it keeps the same UTRs before and after `call-peaks`. Eugenia
  Guerrero Aruffo kept the 5' UTRs with 500 molecules or more.
- **`tools site-sequences`** writes the genome sequence around each site,
  as FASTA for motif tools (for a `.fa`, `.fasta` or `.fna` output) or as a
  table, for `--upstream` and `--downstream` nt (50 each by default).
  Positions count from the transcript's end, as each field numbers them: for
  5' data, +1 is the transcription start site; for 3' data, -1 is the
  cleavage site. In Python, `genome.site_sequences()` returns the table.
- **Quality control commands**, which write tables and leave the count
  matrices unchanged:
  - `qc report`: a row per count matrix of any step, with what it holds and
    the steps that made it, its cells, sites, UTRs and molecules, per-cell
    medians, multi-site UTRs, the depth correlation and the mapping summary.
    `--multiqc` adds the header that MultiQC reads.
  - `qc cells`: per cell, its molecules, sites, measured and multi-site UTRs,
    and its relative UTR size; the log gives the correlation of the relative
    size with the molecules, which shows whether UTR sizes follow depth.
  - `qc batches -g COLUMN`: per group of cells (batch, sample), the sites no
    other group has; per pair of groups, the correlation and difference of
    their site usage.
  - `qc known-sites`: each site's distance to the nearest known site of a BED
    file, such as the PolyASite 2.0 clusters or refTSS (a first line of column
    names, as refTSS's files have, is skipped).
  - `qc priming`: flags the sites that the genome around them could explain
    as priming artefacts. For 3' data, internal priming: 6 A's in a row, or 7
    in 10 nt, in the genome after the site. For 5' data, strand invasion: the
    genome before the site like the template-switching oligo's 3' end (the
    rule of Tang et al. 2013), with the oligo of the matrix's protocol or
    `--tso`.
  - `qc peaks`: the peaks a bandwidth calls from raw sizes, with their
    molecules, share of their UTR and the spread of the raw sizes they collect.
- **Mapping summary.** `run` and `count` write `<out>_mapping_summary.tsv`:
  per BAM file, the alignments in the UTR regions that each check skipped
  (wrong strand, outside the UTR, no barcode, ...) and the reads and molecules
  left; in total, the molecules left after merging the samples, the size
  filter and the cell list, and those counted. `map-reads` writes the BAM rows
  to `intermediate/mapping_summary.tsv`, and `count` reads them back from the
  stored read mappings. Each matrix keeps the summary (`process concatenate`
  joins those of the experiments). The log's mapping statistics use the same
  names.
- Every count matrix records how it was made, in `.uns["utrcalling"]`: the
  utrcalling version, the commands that made it, what it holds (raw sizes,
  peaks, fractions or mean sizes), the KDE bandwidth, the protocol, the
  summarization level and the end of the UTR its sizes count from. The QC
  commands read it.

### Fixed

- **Peak calling on densities with flat stretches.** Each peak took the
  molecules below the next local minimum in order, which assumes that maxima
  and minima alternate. Where the density was flat (a flat top, or a step on a
  slope), they did not: molecules went to the wrong peak, some to none, and
  peaks fell at sizes without molecules. Peaks are now the local maxima (a flat
  top once, at its middle; a step is not a peak), and two neighbouring peaks
  split at the lowest point between them. The Gaussian density of earlier
  versions practically never has flat stretches: its peaks are unchanged
  (identical on every UTR of the 3' and 5' data sets we checked). The new
  kernels have many, as Eugenia Guerrero Aruffo found in her master's thesis
  (2026).
- `call-peaks` (`call-peaks-different-bandwidths` before) and the bandwidth
  scores crashed on a matrix with a UTR without molecules, for example after
  keeping only some of the cells. Such a UTR now has no sites.

## [4.0.0] - 2026-11-05

Site names and PAS distances change, three commands are gone, and the command
line has a new structure. Other command lines written for 3.x keep working.

### Changed (breaking)

- **Site names.** Every output names its sites `<gene>_<n>-<size>`, for example
  `TSPAN6_2-816`: raw sizes, KDE peaks, and the outputs of `bandwidth call-peaks`
  and `process concatenate`. Genes without a name use their Ensembl ID, and names
  are always unique. Before, the same site had different names depending on the
  command (`ENSG00000000003-816`, `TSPAN6_pA2`), and genes without a name gave
  duplicate names. To find sites in code, use the `utr_name`, `utr_size` and
  `gene_name` columns of `.var` rather than the names. `process rename`
  (`generate-names`) now only renumbers the sites after filtering.
- **PAS distances** (`qc pas-distance`, `bandwidth score`, and
  `peak_calling_anndata.pas_distance` in Python) are the UTR size minus the end
  of the nearest hexamer, on both strands. Before, distances on the minus strand
  had the wrong sign, and a hexamer 10 to 30 nt upstream was preferred over the
  nearest one.
- `--protocol-version` defaults to 3 instead of 2 in `run`, `call-peaks` and
  `prepare-reference-and-map-reads`. Pass `-v 2` for 10x v2 data.
- Removed commands: `split-train-test` and `subset-experiments` (each a line or
  two of Python with scikit-learn or anndata), and `run-tests` (use `pytest`; see
  the README).
- `-s/--summarization-level` accepts only `utr` and `gene`. `transcript` used to
  give UTR-level output without a warning.
- `prepare-reference-and-map-reads` requires `-A/--annotations-file`; it crashed
  without one.
- `call-peaks` takes the UTR reference as `.parquet` (or `.pkl` from older
  versions); a `.bed` reference used to crash it.
- In Python, `peak_calling_anndata.estimate_best_bandwidth()` takes the genome
  as its second argument.

### Added

- **Commands grouped by purpose.** Next to the pipeline commands, `process`
  holds the commands that filter and convert count matrices (`filter`,
  `select-alternative`, `rename`, `to-fractions`, `mean-sizes`,
  `concatenate`), `qc` the quality control (`pas-distance`), and `bandwidth` the
  choice of KDE bandwidth (`score`, `call-peaks`). The old command names still
  work; the README lists them.
- **Clearer help.** `--help` groups each pipeline command's options by topic.
  Options are spelled with hyphens (`--kde-bandwidth`); the underscore spelling
  still works. The values of `-m` and `-L` are now checked before anything runs.
- The `bulk_pe` protocol, for unstranded paired-end bulk RNA-seq: it counts each
  fragment once, at its polyA-distal base.

### Fixed

Results:

- `--kde-bandwidth` had no effect without `--raw-sizes`: peaks always used a
  bandwidth of 12.
- KDE peaks at UTR size 0, which 5' protocols can produce, lost their molecules
  to the UTR's last peak.
- `peak_call_anndata` in Python swapped counts between peaks when a UTR's sites
  were not sorted by size. The commands always sorted them first.
- Reads without an observation ID were counted as a cell named "nan". They now
  raise an error.

Crashes and silent failures:

- `prepare-reference-and-map-reads` crashed on every run, and
  `concatenate-experiments` on the h5ad files that utrcalling writes.
- `check-distance-to-pas` crashed when no UTR had a PAS hexamer, and
  `estimate-best-bandwidth` did not pass the genome to the hexamer search.
- `--cores 0` crashed when mapping several BAM files.
- `run` and `call-peaks` could exit without error but without writing their
  output, and `-i` crashed on tables that parquet cannot store (they are now
  written as CSV).
- `generate-names` wrote `<out>_1.h5ad` when its output existed, so the next
  step read the old file.
- Integer sample IDs in the annotation file broke sample name resolution, and
  batch columns under pandas 3 and reads without aligned bases crashed the
  pipeline.
- Missing annotations, a missing `sample_id` column and unknown peak-calling
  methods now give clear errors.
- Importing utrcalling hid the FutureWarnings of all other code in the process.

### Installation

- `pyproject.toml` holds all package metadata and dependency ranges; `setup.py`
  and `requirements.txt` are gone. Requires rapidfuzz 3.1 or newer and
  Cyclopts 5.
- `environment.yml` creates a working conda environment (Python 3.12), and
  `envs/utrcalling-frozen.yml` the exact versions the tests passed with.
- Licensed under BSD-3-Clause, with a `LICENSE` file, and citable through
  `CITATION.cff` and Zenodo.
- Tests run with pytest. CI runs them in both environments, with 100% branch
  coverage, ruff and mypy.

## [3.1.10] - 2026-10-27

First release archived on Zenodo (published under the tag `v1.0.0` by mistake;
`v3.1.10` marks the same commit). Earlier versions have no changelog.
