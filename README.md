# utrcalling

[![CI](https://github.com/AndreMacedo88/utrcalling/actions/workflows/tests.yml/badge.svg)](https://github.com/AndreMacedo88/utrcalling/actions/workflows/tests.yml)
![Python >= 3.11](https://img.shields.io/badge/python-%3E%3D3.11-blue)
![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-green)
![Version 3.1.10](https://img.shields.io/badge/version-3.1.10-orange)
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
- **Flexible summarisation** --- aggregate results at UTR, transcript, or gene
  level.
- **KDE peak calling** --- Gaussian kernel density estimation with configurable
  bandwidth to identify consensus UTR isoform peaks from per-molecule size
  distributions.
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

| Platform (`--protocol_platform`) | Read used | UMI | Dedup | Description |
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

Each platform is used with a `--protocol_version`, which defaults to `3`. `10x`
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
- **Raw sizes mode** (`--raw_sizes`) --- produces two files: raw UTR sizes
  and KDE-estimated peaks (`*_kde.h5ad`).
- **Site names** --- every output names its sites (raw UTR sizes or peaks)
  `<gene>_<n>-<size>`, e.g. `TSPAN6_2-816`: the gene name (the Ensembl ID for
  genes without one), the number of the site within its gene (1 is the
  shortest site of the gene's first UTR) and its UTR size in nt. A site keeps
  its name through the filtering commands. `.var` also holds `utr_name`,
  `utr_size` and `gene_name`.
- **Intermediate files** (`--store_intermediate`) --- `.bed` UTR reference
  and `.parquet` read mappings for resumable workflows.
- **Log files** --- automatically written to the output directory (suppress with
  `--no_log_file`).

## Installation

### Prerequisites

- Python >= 3.11 (tested on 3.12)
- [BEDTools](https://github.com/arq5x/bedtools2) must be installed and
  available on your `PATH`.

### With conda (recommended)

The provided `environment.yml` creates a dedicated conda environment with
Python 3.12, BEDTools, and the latest compatible versions of all Python
dependencies, and installs utrcalling in editable mode:

```bash
git clone https://github.com/AndreMacedo88/utrcalling.git
cd utrcalling
conda env create -f environment.yml
conda activate utrcalling
```

### Frozen environment (reproducible)

`envs/utrcalling-frozen.yml` pins every package to the exact versions the
test suite was validated with. Use it if the recommended install fails or
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
git clone https://github.com/AndreMacedo88/utrcalling.git
cd utrcalling
pip install .
```

For development (editable install):

```bash
pip install -e .
```

### From PyPI

Not yet available on PyPI.

## Usage

utrcalling provides a command-line interface with multiple subcommands. Run
`utrcalling --help` for a full list.

### Full pipeline

Run the entire pipeline --- from GTF parsing through read mapping to peak
calling --- in a single command:

```bash
utrcalling run \
    genome_annotations.gtf.gz \
    --alignments sample1.bam sample2.bam \
    --annotations_file experiment_annotations.csv \
    --protocol_platform 10x \
    --protocol_version 3 \
    --out results.h5ad
```

### Standard 3' 10x (Cell Ranger)

For standard 10x Chromium data processed through Cell Ranger:

```bash
utrcalling run \
    genome_annotations.gtf.gz \
    --alignments cellranger_output/possorted_genome_bam.bam \
    --annotations_file annotations.csv \
    --protocol_platform standard_10x \
    --protocol_version 3 \
    --out standard_10x_results.h5ad
```

### Standard SmartSeq3

For standard SmartSeq3 data with one BAM per cell:

```bash
utrcalling run \
    genome_annotations.gtf.gz \
    --alignments cell1.bam cell2.bam cell3.bam \
    --annotations_file annotations.csv \
    --protocol_platform standard_smartseq \
    --protocol_version 3 \
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
    --annotations_file annotations.csv \
    --protocol_platform bulk_pe \
    --protocol_version 1 \
    --out bulk_results.h5ad
```

Random-fragmentation libraries spread reads over a much wider window around
each polyadenylation site than 3'-end assays do, so a wider `--kde_bandwidth`
than the default of 12 is usually needed. Otherwise the spread is split into
many closely spaced pseudo-peaks. `utrcalling estimate-best-bandwidth` can help
choose a value.

### Step-by-step execution

The pipeline can also be run in two stages for large datasets or debugging:

```bash
# Stage 1 & 2: Build UTR reference and map reads (written to ./intermediate/)
utrcalling prepare-reference-and-map-reads \
    genome_annotations.gtf.gz \
    --alignments sample.bam \
    --annotations_file annotations.csv \
    --protocol_platform 10x \
    --protocol_version 3

# Stage 3: Call peaks from intermediate files (same annotations and --label_batches)
utrcalling call-peaks \
    intermediate/utr_reference_regions.parquet \
    --utr_sizes_counts intermediate/UTR_sizes_counts_*.parquet \
    --annotations_file annotations.csv \
    --protocol_platform 10x \
    --protocol_version 3 \
    --out peaks.h5ad
```

### Key options

| Option | Description |
| --- | --- |
| `--feature / -f` | UTR type: `three_prime_utr` (default) or `five_prime_utr`. |
| `--summarization_level / -s` | Aggregate at `UTR`, `transcript`, or `gene` level. |
| `--raw_sizes / -r` | Output both raw UTR sizes and KDE peaks. |
| `--kde_bandwidth` | KDE bandwidth for peak calling (default: 12). |
| `--strandedness` | Library strandedness for `truseq`: `reverse` (default), `forward`, or `unstranded`. |
| `--batch_merge_method / -m` | How to merge batches: `union` or `intersection`. |
| `--store_intermediate / -i` | Save intermediate files for resumable workflows. |
| `--label_batches / -L` | Append batch index to cell barcodes. |

### Downstream processing

After running the main pipeline, several utility commands are available for
further analysis:

```bash
# Remove low-count peaks
utrcalling remove-peaks-below-threshold results.h5ad filtered.h5ad \
    --threshold 0.05 --minimum_counts 10

# Keep only genes with multiple UTR isoforms (APA/ATI events)
utrcalling select-alternative-transcription-events filtered.h5ad apa.h5ad

# Optional: number each gene's remaining sites 1, 2, ... again after filtering
utrcalling generate-names apa.h5ad named.h5ad

# Compute mean UTR sizes per gene
utrcalling compute-mean-utr-sizes named.h5ad mean_sizes.h5ad

# Convert counts to percentage usage per gene
utrcalling counts-to-percentages named.h5ad percentages.h5ad

# Concatenate experiments
utrcalling concatenate-experiments \
    -i experiment1.h5ad experiment2.h5ad \
    -o combined.h5ad

# Estimate optimal KDE bandwidth
utrcalling estimate-best-bandwidth raw_sizes.h5ad bandwidth_scores.joblib genome.fa \
    --bandwidths 1 7 12 15

# Validate peaks against known polyadenylation signals
utrcalling check-distance-to-pas results.h5ad distances.joblib genome.fa
```

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
│   ├── runtime.py              CLI entry point
│   ├── core/
│   │   ├── core.py             Pipeline orchestrator
│   │   ├── protocol.py         Sequencing protocol definitions
│   │   ├── utr_mapping.py      Read-to-UTR mapping (pysam)
│   │   ├── utr_reference.py    GTF parsing, UTR extraction
│   │   ├── count_matrix.py     Sparse count matrix assembly
│   │   ├── barcode_filtering.py
│   │   ├── process_mapped_reads.py
│   │   ├── storage.py
│   │   └── ...
│   ├── tools/                   CLI subcommand implementations
│   ├── tests/                   Unit and integration tests
│   └── notebooks/               Example notebooks
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

## License

BSD-3-Clause. See [LICENSE](LICENSE).

## Author

Andre Lopes Martins Macedo ([andre.lopes.macedo@gmail.com](mailto:andre.lopes.macedo@gmail.com))

Source code: https://github.com/AndreMacedo88/utrcalling
