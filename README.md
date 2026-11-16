# utrcalling

[![CI](https://github.com/AndreMacedo88/utrcalling/actions/workflows/tests.yml/badge.svg)](https://github.com/AndreMacedo88/utrcalling/actions/workflows/tests.yml)
[![Coverage](https://codecov.io/gh/AndreMacedo88/utrcalling/graph/badge.svg)](https://codecov.io/gh/AndreMacedo88/utrcalling)
![Python >= 3.11](https://img.shields.io/badge/python-%3E%3D3.11-blue)
![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-green)
[![PyPI](https://img.shields.io/pypi/v/utrcalling)](https://pypi.org/project/utrcalling/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23686176.svg)](https://doi.org/10.5281/zenodo.23686176)

A bioinformatics toolkit for calculating molecule UTR (Untranslated Region)
sizes from RNA sequencing reads. **utrcalling** processes BAM alignment files
against GTF gene annotations to produce AnnData (`.h5ad`) count matrices of
UTR-length usage per cell or sample --- enabling the study of alternative
polyadenylation (APA) and alternative transcription initiation (ATI) at
single-cell resolution.

## Features

- **Multi-protocol support** --- works with modified and standard versions of
  10x Chromium and SmartSeq3 sequencing, CapSeq and 5' 10x protocols, and
  stranded or unstranded paired-end bulk RNA-seq.
- **3'UTR and 5'UTR calling** --- analyse either end of the transcript.
- **Flexible summarisation** --- aggregate results at UTR or gene level.
- **KDE peak calling** --- kernel density estimation, with a choice of kernel
  and bandwidth (fixed, set per UTR, or chosen for the data by the genome), to
  identify consensus UTR isoform peaks from per-molecule size distributions.
- **Batch-aware processing** --- union or intersection merge strategies for
  multi-sample experiments.
- **Sparse output** --- count matrices are stored as sparse arrays in AnnData
  objects, ready for downstream single-cell analysis with scanpy.
- **Downstream utilities** --- filter peaks, select APA/ATI events, compute
  mean UTR sizes, convert counts to percentages, concatenate
  experiments, and more.

## Protocols

utrcalling supports several sequencing protocols, each with tailored read
parsing, deduplication, and UTR size estimation logic:

| Platform (`--protocol-platform`) | Read used | UMI | Dedup | Description |
| --- | --- | --- | --- | --- |
| `10x` | Read 1 | Yes | keep first | Modified 10x Chromium (extended Read 1 captures the 3'UTR terminus). |
| `standard_10x` | Read 2 | Yes | keep max | Standard 3' 10x Chromium. Requires Cell Ranger BAM output with CB/UB tags. Keeps the largest UTR size per (barcode, UMI) pair. |
| `5prime10x` | Read 1 | Yes | keep first | 5' 10x Chromium. |
| `smartseq` | Read 1 | Yes | keep first | Modified SmartSeq3 (extended Read 1 captures the 3'UTR terminus). |
| `standard_smartseq` | Any | No | none | Standard SmartSeq3. One BAM per cell, no barcodes/UMIs. Accepts reads from both strands and uses the distal read position (closest to polyA). |
| `smart-seq_cap` | Read 1 | Yes | keep first | CapSeq SmartSeq3. |
| `smart-seq_cap-readname` | Read 1 | Yes | keep first | CapSeq SmartSeq3 with read-name UMI extraction. |
| `truseq` | Any | No | none | Stranded bulk RNA-seq (e.g. TruSeq Stranded mRNA), single- or paired-end. One BAM per sample. Strand-aware via `--strandedness` (default `reverse`, correct for dUTP libraries). Uses the distal read position. |
| `bulk_pe` | Any | No | fragment | Unstranded paired-end bulk RNA-seq (e.g. TruSeq RNA v2). One BAM per sample. Counts each fragment once (mates collapsed by read name) and uses the fragment's distal base, so it reaches further into the UTR than a single mate. Fragments running past the annotated 3' end are placed at the UTR end rather than dropped. |

Each platform is used with a `--protocol-version`, which defaults to `3`. `10x`
and `standard_10x` also support `2`, and `5prime10x` needs `1`. The bulk
protocols (`truseq`, `bulk_pe`) have no versions and accept any value.

## Input

1. **GTF annotation file** --- genome annotations from [Ensembl](http://ftp.ensembl.org/pub/) (plain or gzipped).
2. **BAM alignment file(s)** --- coordinate-sorted and indexed. For
   `standard_10x`, these must be Cell Ranger output BAMs containing `CB`
   and `UB` tags.
3. **Annotations file** --- CSV with experiment metadata (cell barcodes, sample
   IDs, batch information). Required columns depend on the protocol.

## Output

- **Primary output** --- `.h5ad` (AnnData) count matrix with cells/samples as
  observations and UTR peaks as variables.
- **Raw sizes mode** (`--raw-sizes`) --- produces two files: raw UTR sizes
  and KDE-estimated peaks (`*_kde.h5ad`).
- **Site names** --- every output names its sites (raw UTR sizes or peaks)
  `<gene>_<n>-<size>`, e.g. `TSPAN6_2-816`: the gene name (the Ensembl ID for
  genes without one), the number of the site within its gene (1 is the
  shortest site of the gene's first UTR) and its UTR size in nt. A site keeps
  its name through the filtering commands. `.var` also holds `utr_name`,
  `utr_size` and `gene_name`.
- **Mapping summary** --- `<out stem>_mapping_summary.tsv`, also kept in the
  matrix's `.uns["utrcalling"]["mapping_summary"]`: what happened to the
  reads, per BAM file and in total, to answer "why so few molecules?" (see
  [Quality control](#quality-control)).
- **How each matrix was made** --- every count matrix records in
  `.uns["utrcalling"]` the utrcalling version, the commands that made it (as
  typed, the first one making the matrix), what it holds (`matrix`:
  `raw_sizes`, `peaks`, `fractions` or `mean_sizes`), and the settings needed
  to read it: the KDE `bandwidth`, the protocol, the `summarization_level`,
  and `utr_end`, the end of the UTR its sizes count from (`three_prime`
  protocols count from the UTR's start, `five_prime` ones from its end).
- **Intermediate files** (`--store-intermediate`) --- the UTR reference and
  the read mappings, as `.parquet` files in `./intermediate/`, from which
  `count` can redo the last steps.
- **Log files** --- automatically written to the output directory (suppress with
  `--no-log-file`).

## Installation

### Prerequisites

- Python >= 3.11 (tested on 3.12)
- [BEDTools](https://github.com/arq5x/bedtools2) must be installed and
  available on your `PATH`.

### With conda and pip (recommended)

Create a conda environment with Python and BEDTools, then install utrcalling
from PyPI into it:

```bash
conda create -n utrcalling -c conda-forge -c bioconda python=3.12 bedtools
conda activate utrcalling
pip install utrcalling
```

### From the source code

The provided `environment.yml` creates a dedicated conda environment with
Python 3.12, BEDTools, and the latest compatible versions of all Python
dependencies, and installs utrcalling from the cloned repository in editable
mode, for example to use changes not yet released:

```bash
git clone https://github.com/AndreMacedo88/utrcalling.git
cd utrcalling
conda env create -f environment.yml
conda activate utrcalling
```

### Frozen environment (reproducible)

`envs/utrcalling-frozen.yml`, in the cloned repository, pins every package to
the exact versions the test suite was validated with. Use it if the recommended install fails or
behaves differently, for example after breaking releases of dependencies in
the future:

```bash
conda env create -f envs/utrcalling-frozen.yml
conda activate utrcalling
pip install --group test   # pytest, to check the installation
pytest
```

### Into an existing environment

To install utrcalling alongside other tools, install it with `pip` inside your
active environment. Compatible versions are resolved from the ranges declared
in `pyproject.toml`, and BEDTools must be available on your `PATH`:

```bash
pip install utrcalling
```

## Usage

utrcalling has four pipeline commands, and three groups of commands for what
comes after them. Like scanpy's `pp` and `tl`, `process` prepares the count
matrix and `tools` then computes the measures to analyse from it:

| Command | Purpose |
| --- | --- |
| `run` | The whole pipeline, from BAM files to a count matrix of UTR sites |
| `map-reads`, then `count` | The same, in two steps |
| `call-peaks` | The peaks of a raw-sizes count matrix (from `run --raw-sizes`), with one or more KDE bandwidths |
| `process ...` | Prepare the count matrix: filter sites and UTRs, keep multi-site UTRs, number sites again, join experiments |
| `tools ...` | Compute the measures to analyse: site usage fractions, mean UTR sizes, the sequences around sites |
| `qc ...` | Quality control tables, at any step |

`utrcalling --help` lists them, `utrcalling process --help` lists a group's
commands, and `utrcalling run --help` describes a command's options.

### Full pipeline

Run the entire pipeline --- from GTF parsing through read mapping to peak
calling --- in a single command:

```bash
utrcalling run \
    genome_annotations.gtf.gz \
    --alignments sample1.bam sample2.bam \
    --annotations-file experiment_annotations.csv \
    --protocol-platform 10x \
    --protocol-version 3 \
    --out results.h5ad
```

### Standard 3' 10x (Cell Ranger)

For standard 10x Chromium data processed through Cell Ranger:

```bash
utrcalling run \
    genome_annotations.gtf.gz \
    --alignments cellranger_output/possorted_genome_bam.bam \
    --annotations-file annotations.csv \
    --protocol-platform standard_10x \
    --protocol-version 3 \
    --out standard_10x_results.h5ad
```

### Standard SmartSeq3

For standard SmartSeq3 data with one BAM per cell:

```bash
utrcalling run \
    genome_annotations.gtf.gz \
    --alignments cell1.bam cell2.bam cell3.bam \
    --annotations-file annotations.csv \
    --protocol-platform standard_smartseq \
    --protocol-version 3 \
    --out smartseq3_results.h5ad
```

### Bulk RNA-seq

For bulk RNA-seq with one BAM per sample. The annotations file needs a
`sample_id` column whose values appear in the BAM paths (e.g. `S1` for
`aligned/S1.bam`). Use `truseq` for stranded libraries and `bulk_pe` for
unstranded paired-end libraries:

```bash
utrcalling run \
    genome_annotations.gtf.gz \
    --alignments aligned/S1.bam aligned/S2.bam \
    --annotations-file annotations.csv \
    --protocol-platform bulk_pe \
    --protocol-version 1 \
    --out bulk_results.h5ad
```

Random-fragmentation libraries spread reads over a much wider window around
each polyadenylation site than 3'-end assays do, so a wider `--kde-bandwidth`
than the default of 12 is usually needed. Otherwise the spread is split into
many closely spaced pseudo-peaks. `utrcalling qc score-bandwidths` can help
choose a value.

### Step-by-step execution

The pipeline can also be run in two stages for large datasets or debugging:

```bash
# Stage 1 & 2: Build UTR reference and map reads (written to ./intermediate/)
utrcalling map-reads \
    genome_annotations.gtf.gz \
    --alignments sample.bam \
    --annotations-file annotations.csv \
    --protocol-platform 10x \
    --protocol-version 3

# Stage 3: Count the sites from intermediate files (same annotations and --label-batches)
utrcalling count \
    intermediate/utr_reference_regions.parquet \
    --utr-sizes-counts intermediate/UTR_sizes_counts_*.parquet \
    --annotations-file annotations.csv \
    --protocol-platform 10x \
    --protocol-version 3 \
    --out peaks.h5ad
```

### Key options

| Option | Description |
| --- | --- |
| `--feature / -f` | UTR type: `three_prime_utr` (default) or `five_prime_utr`. |
| `--summarization-level / -s` | `utr` (default), or `gene` to join the UTRs of each gene. |
| `--raw-sizes / -r` | Output both raw UTR sizes and KDE peaks. |
| `--kde-bandwidth / -b` | KDE bandwidth for peak calling: a number of nt (default: 12), or a rule that sets one per UTR from the spread of its sizes, `scott` or `silverman`, optionally scaled (`scott:0.5`). |
| `--kde-kernel` | KDE kernel: `gaussian` (default), `linear`, `epanechnikov`, `exponential`, `tophat` or `cosine`. |
| `-b optimize`, `--candidates`, `--genome` | Choose the bandwidth: score the candidates against the genome (see `qc score-bandwidths`) and use the best. |
| `--strandedness` | Library strandedness for `truseq`: `reverse` (default), `forward`, or `unstranded`. |
| `--batch-merge-method / -m` | How to merge batches: `union` or `intersection`. |
| `--store-intermediate / -i` | Save intermediate files for resumable workflows. |
| `--label-batches / -L` | `true` to append the batch to cell barcodes or sample IDs. |
| `--cores` | Maximum number of worker processes (default: all cores but one). |

### Downstream processing

After the pipeline, the `process` commands prepare the count matrix and the
`tools` commands compute the measures to analyse. A typical sequence,
starting from the raw sizes of `run --raw-sizes`:

```bash
# Call the peaks of the raw sizes (writes peaks_bw12.h5ad)
utrcalling call-peaks raw_sizes.h5ad peaks --bandwidths 12

# Remove low-count sites
utrcalling process filter-sites peaks_bw12.h5ad filtered.h5ad \
    --threshold 0.05 --minimum-counts 10

# ... or count per group of cells (a column of the annotations), keeping the
# sites that pass in at least one group, such as those a treatment raises
utrcalling process filter-sites peaks_bw12.h5ad filtered.h5ad \
    --threshold 0.1 --minimum-counts 10 --groupby Treatment

# Remove the UTRs with few molecules over all cells, whose sites are unsure
utrcalling process filter-utrs filtered.h5ad deep.h5ad --minimum-molecules 500

# Keep only the UTRs with two or more sites (APA/ATI events)
utrcalling process keep-multi-site filtered.h5ad apa.h5ad

# Optional: number each gene's remaining sites 1, 2, ... again. The sites
# then have other names than in the files before: compare files by the
# utr_name and utr_size columns.
utrcalling process rename-sites apa.h5ad named.h5ad

# Join the cells of several experiments
utrcalling process concatenate \
    -i experiment1.h5ad experiment2.h5ad \
    -o combined.h5ad

# Each site's share of its UTR's counts, per cell
utrcalling tools to-fractions named.h5ad fractions.h5ad

# Mean UTR size per cell and UTR
utrcalling tools mean-sizes named.h5ad mean_sizes.h5ad

# The genome around each site, for motif analyses (FASTA, or a table for .tsv).
# Positions count from the transcript's end: for 5' data +1 is the start site,
# for 3' data -1 is the cleavage site; --upstream reads before it.
utrcalling tools site-sequences named.h5ad genome.fa sites.fa \
    --upstream 50 --downstream 50
```

### Peak calling for 5' data

The default KDE, Gaussian with a bandwidth of 12 nt, suits 3' data, where the
molecules of a polyadenylation site spread over a few nt. Transcription start
sites are exact to the nucleotide, and promoters often use several within a few
nt of each other; conditions that change which start sites cells use can shift
them by little. In mouse neural stem cells treated with interferon, the 5' UTRs
that shortened lost a median of 36 nt, less than 30 nt in 39% of them (Guerrero
Aruffo, 2026). A 12-nt Gaussian joins such sites and moves its peaks off the
start sites: on that Cap-seq data, its initiator score (see below) is 0.22,
while 42% of the molecules themselves start at a pyrimidine-purine initiator.

When the start sites in your 5' data may move by a few nt, choose the peak
calling for the data. `qc score-bandwidths` scores 5' peaks by the initiator:
the share of molecules at peaks with a pyrimidine (C or T) at -1 and a purine
(A or G) at +1, where RNA polymerase II starts most transcripts. Peaks moved off
the start sites lose it:

```bash
# Score bandwidths, one kernel at a time, on the raw sizes of run -f five_prime_utr -r
utrcalling qc score-bandwidths raw_sizes.h5ad scores_linear.tsv genome.fa --kde-kernel linear
utrcalling qc score-bandwidths raw_sizes.h5ad scores_gaussian.tsv genome.fa

# Or let run, count or call-peaks choose the bandwidth with the best score
utrcalling call-peaks raw_sizes.h5ad peaks -b optimize --kde-kernel linear --genome genome.fa
```

The default stays as it is for other analyses.

#### Interferon-treated neural stem cells (bulk Cap-seq)

For 5' Cap-seq data of adult neural stem cells treated with interferon, Eugenia
Guerrero Aruffo found these settings best in her master's thesis (2026); these
commands give her sites in 99% of the UTRs:

```bash
utrcalling run Mus_musculus.GRCm39.108.gtf.gz -a sample_*.bam -A samples.csv \
    -f five_prime_utr -p <your 5' protocol> -r -o capseq.h5ad
utrcalling call-peaks capseq.h5ad capseq -b scott:0.5 --kde-kernel linear
utrcalling process filter-utrs capseq_bwscott0.5.h5ad deep.h5ad --minimum-molecules 500
utrcalling process filter-sites deep.h5ad sites.h5ad \
    --threshold 0.1 --minimum-counts 10 --groupby Treatment
```

- `--kde-kernel linear`: a triangle, sharp at its centre and zero beyond the
  bandwidth, keeps the peaks on the start sites.
- `-b scott:0.5`: per UTR, half of Scott's bandwidth (SD x n^-1/5), finer where
  the molecules are many: 2 nt in the median UTR of that data. Each site keeps
  its UTR's bandwidth in `.var["bandwidth"]`.
- `filter-utrs`: UTRs with fewer molecules gave unsure sites.
- `filter-sites --groupby Treatment`: thresholds per treatment, so start sites
  that interferon raises stay, though they are a small share of all molecules.

The bandwidths and the filters count the molecules of all the samples in the
matrix: subset it to the samples you study first (she called the sites of the
wild-type Cap-seq libraries).

On her UTRs with 500 molecules or more, `-b optimize --kde-kernel linear`
chooses 2 nt (an initiator score of 0.60, against 0.59 for `scott:0.5`). From
the sites, `tools site-sequences sites.h5ad genome.fa tss.fa` writes the
sequences around the start sites for motif analyses, and `qc priming` flags
those that strand invasion could explain.

### Quality control

The `qc` commands describe count matrices from any step, without changing
them. Each writes a tab-separated table, which pandas, R and MultiQC read:

| Command | One row per | What it tells |
| --- | --- | --- |
| `qc report` | count matrix | What each matrix holds and which steps made it; its cells, sites, UTRs and molecules, per-cell medians, multi-site UTRs, depth correlation and mapping summary |
| `qc cells` | cell (or sample) | Molecules, sites, UTRs measured, multi-site UTRs, and its relative UTR size |
| `qc batches` | group of cells, and pair of groups | Sites only one group has; how alike the groups' site usage is |
| `qc known-sites` | site | Distance to the nearest known site of a database |
| `qc pas-distance` | site | Distance to the nearest polyadenylation signal hexamer |
| `qc priming` | site | Whether the genome around it could explain it as a priming artefact: A's after a 3' site (internal priming), or a sequence like the template-switching oligo before a 5' site (strand invasion) |
| `qc peaks` | peak | Molecules, share of its UTR, and the spread of the raw sizes it collects |
| `qc score-bandwidths` | KDE bandwidth | How well its peaks fall where the genome expects sites: 10-30 nt after a PAS hexamer (3' data), or on a pyrimidine-purine initiator (5' data); and how many it makes |

```bash
# A row per matrix, from mapping to analysis; MultiQC shows *_mqc.tsv files
utrcalling qc report raw_sizes.h5ad peaks_bw12.h5ad filtered.h5ad \
    -o qc_report_mqc.tsv --multiqc

# Per cell. Its log gives the correlation of the cells' relative UTR size
# with their molecules: a strong one means UTR sizes follow sequencing depth.
utrcalling qc cells peaks_bw12.h5ad cells.tsv

# Per batch (or sample, replicate: a column of the cell annotations).
# Writes batches_groups.tsv and batches_pairs.tsv.
utrcalling qc batches peaks_bw12.h5ad batches --groupby Batch

# Distance to known polyadenylation sites (e.g. PolyASite 2.0 clusters) or,
# for 5' data, known transcription start sites (e.g. refTSS)
utrcalling qc known-sites peaks_bw12.h5ad atlas.clusters.2.0.GRCh38.96.bed.gz known_sites.tsv

# Distance of each site to the nearest PAS hexamer
utrcalling qc pas-distance peaks_bw12.h5ad pas_distances.tsv genome.fa

# Sites that the genome could explain as priming artefacts: internal priming
# for 3' data; strand invasion for 5' data, with the template-switching oligo
# of the recorded protocol (or --tso)
utrcalling qc priming peaks_bw12.h5ad genome.fa priming.tsv

# How wide the peaks of a bandwidth are, and how well bandwidths score
utrcalling qc peaks raw_sizes.h5ad peaks_bw12.tsv --bandwidth 12
utrcalling qc score-bandwidths raw_sizes.h5ad bandwidth_scores.tsv genome.fa \
    --bandwidths 4 8 12 16
```

The site tables answer different questions. `known-sites` asks whether a
database has seen the site. `priming` asks whether the genome around it could
explain it as an artefact, for known and new sites alike: internal priming
happens inside the UTR, and its sites look like a shorter UTR. `pas-distance`
asks whether a polyadenylation signal lies where it should. For 3' data, read
together:

| Known site | Priming flag | PAS hexamer 10-30 nt before | Reading |
| --- | --- | --- | --- |
| yes | no | | Solid |
| no | no | yes | Candidate new site |
| no | yes | | Likely internal priming |
| yes | yes | | Check by hand |

```python
import pandas as pd


def read(path):
    return pd.read_csv(path, sep="\t", index_col="site")


sites = read("known_sites.tsv").join(read("priming.tsv")[["internal_priming"]])
sites = sites.join(read("pas_distances.tsv")[["pas_distance"]])
new_sites = sites.query(
    "not near_known_site and internal_priming == False and 10 <= pas_distance <= 30"
)
```

For 5' data, the known sites are transcription start sites and the flag is
strand invasion; the PAS hexamer does not apply. The priming flags need sites
at the molecules' ends: with the standard and bulk protocols, whose reads stop
short of the ends, they say little.

#### Mapping summary

`run` and `count` write `<out>_mapping_summary.tsv`, with a row per BAM file
and a `total` row (`map-reads` writes the BAM rows to
`intermediate/mapping_summary.tsv`). Each matrix keeps the summary of the run
that made it, also after filtering, and `process concatenate` joins those of
the experiments. Each column counts the alignments in the
UTR regions that one step skipped, or the reads and molecules it kept:

| Column | Counts |
| --- | --- |
| `alignments` | Alignments in the UTR regions (one overlapping two UTRs counts twice) |
| `unmapped`, `other_mate`, `not_proper_pair` | Skipped: unmapped, the mate the protocol does not use, or not in a proper pair |
| `duplicate_flag`, `secondary` | Skipped: marked as duplicates in the BAM file, or not the primary alignment |
| `wrong_strand`, `no_aligned_bases`, `outside_utr` | Skipped: on the strand the protocol excludes, without aligned bases, or with their UTR end outside the UTR |
| `no_barcode`, `no_umi` | Skipped: without a cell barcode or a UMI |
| `reads` | Kept, each at a UTR size (`clamped_to_utr_end` of them placed at the UTR's end) |
| `molecules` | Left after removing UMI duplicates (or the second mate of each fragment) |
| `molecules_after_merge` | After merging the samples (`-m intersection` keeps the UTRs all samples have) |
| `molecules_after_size_filter` | After removing UTR sizes below 2 (3' protocols) |
| `molecules_in_cell_list` | In the cells of the annotation file (10x) |
| `molecules_counted` | In the count matrix |

### Command lines from earlier versions

Command lines written before version 4.0.0, for example in existing
workflows, still work. Options may be spelled with underscores
(`--kde_bandwidth`), and the old command names still run their commands,
although `--help` no longer lists them. `call-peaks` with read mappings
(`-c`) runs `count`. The names that version 4.0.0 introduced were replaced in
5.0.0:

| Before 4.0.0 | 4.0.0 | Now |
| --- | --- | --- |
| `prepare-reference-and-map-reads` | `prepare-reference-and-map-reads` | `map-reads` |
| `call-peaks` | `call-peaks` | `count` |
| `call-peaks-different-bandwidths` | `bandwidth call-peaks` | `call-peaks` |
| `remove-peaks-below-threshold` | `process filter` | `process filter-sites` |
| `select-alternative-transcription-events` | `process select-alternative` | `process keep-multi-site` |
| `generate-names` | `process rename` | `process rename-sites` |
| `concatenate-experiments` | `process concatenate` | `process concatenate` |
| `counts-to-percentages` | `process to-fractions` | `tools to-fractions` |
| `compute-mean-utr-sizes` | `process mean-sizes` | `tools mean-sizes` |
| `check-distance-to-pas` | `qc pas-distance` | `qc pas-distance` |
| `estimate-best-bandwidth` | `bandwidth score` | `qc score-bandwidths` |

[CHANGELOG.md](https://github.com/AndreMacedo88/utrcalling/blob/main/CHANGELOG.md) lists what changed in each version.

## Testing

The test suite covers all protocols, edge cases, and the full pipeline
end-to-end.

```bash
# Install the test tools once, inside the utrcalling environment
pip install --group test

# Run all tests
pytest

# Run a single test module, class or test
pytest utrcalling/tests/test_core/test_pipeline.py
pytest utrcalling/tests/test_core/test_pipeline.py::TestCallPeaksSmartseq3LabelMinimal
pytest -k kde_bandwidth   # tests whose names match
```

## Project structure

```
utrcalling/
├── utrcalling/
│   ├── runtime.py              Command line (built with Cyclopts)
│   ├── core/
│   │   ├── core.py             Pipeline orchestrator
│   │   ├── protocol.py         Sequencing protocol definitions
│   │   ├── utr_mapping.py      Read-to-UTR mapping (pysam)
│   │   ├── utr_reference.py    GTF parsing, UTR extraction
│   │   ├── count_matrix.py     Sparse count matrix assembly
│   │   ├── protocols/          KDE peak calling (peak_calling.py) and its settings
│   │   ├── genome.py           Where sites lie, and the genome around them
│   │   ├── barcode_filtering.py
│   │   ├── process_mapped_reads.py
│   │   ├── storage.py
│   │   └── ...
│   ├── commands/               One module per command, a folder per group
│   │                           (options.py: the options they share)
│   └── tests/                  Unit and integration tests
├── notebooks/                  Exploratory analyses
├── envs/
│   └── utrcalling-frozen.yml   Exact versions validated by the test suite
├── pyproject.toml              Package metadata and dependency ranges
├── environment.yml             Recommended conda environment
├── CITATION.cff                Citation metadata
└── README.md
```

## Dependencies

Python dependencies are installed automatically. Their compatible version
ranges are declared in `pyproject.toml`, and the exact validated versions in
`envs/utrcalling-frozen.yml`. The main ones are:

- **pysam** --- BAM file reading
- **pybedtools** --- genomic interval operations (requires BEDTools)
- **anndata** / **scanpy** --- single-cell data structures
- **pandas** / **numpy** / **scipy**
- **joblib** --- parallelism
- **gtfparse** --- GTF file parsing
- **rapidfuzz** --- fuzzy barcode matching
- **scikit-learn** --- used by KDE peak calling
- **pyarrow** --- Parquet I/O

System dependency:

- **BEDTools** (>= 2.30.0) --- install via `conda install -c bioconda bedtools`

## Citation

If you use utrcalling in your research, please cite it. Releases are archived
on Zenodo under the DOI
[10.5281/zenodo.23686176](https://doi.org/10.5281/zenodo.23686176), which
always resolves to the latest version; each version also has its own DOI,
listed on that page. Citation metadata is in `CITATION.cff`, and GitHub's
"Cite this repository" button formats it as APA or BibTeX.

## Acknowledgements

Peak calling for 5' data builds on the master's thesis of Eugenia Guerrero
Aruffo: *Interferon regulates transcription and translation in neural stem
cells by targeting regulatory elements on the 5' UTR* (Heidelberg University,
carried out at the German Cancer Research Center, 2026). She chose the linear
kernel, half of Scott's bandwidth per UTR, the filters per treatment and by
molecules; found and fixed peak calling on flat densities; and proposed the
initiator as a measure of how well peaks find start sites.

## License

BSD-3-Clause. See [LICENSE](https://github.com/AndreMacedo88/utrcalling/blob/main/LICENSE).

## Author

Andre Lopes Martins Macedo ([andre.lopes.macedo@gmail.com](mailto:andre.lopes.macedo@gmail.com))

Source code: https://github.com/AndreMacedo88/utrcalling
