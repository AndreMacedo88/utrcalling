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
  `score-bandwidths` has a row per bandwidth: its score, sites and sites per
  UTR, and takes `-H/--hamming` (1 by default, as before). In Python,
  `peak_calling_anndata.pas_distance_table()` and `score_bandwidths()` return
  these tables.

### Added

- Every count matrix records how it was made, in `.uns["utrcalling"]`: the
  utrcalling version, the commands that made it, what it holds (raw sizes,
  peaks, fractions or mean sizes), the KDE bandwidth, the protocol, the
  summarization level and the end of the UTR its sizes count from. The QC
  commands read it.

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
