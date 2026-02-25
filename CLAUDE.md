# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**utrcalling** is a bioinformatics Python package for calculating molecule UTR (Untranslated Region) sizes from RNA sequencing reads. It processes BAM alignment files against GTF gene annotations to produce AnnData (.h5ad) count matrices of UTR-length usage per cell/sample.

## Common Commands

### Conda environment

All project dependencies (including system tools like `bedtools`) live in the `utrcalling` conda environment. **Always use `conda run -n utrcalling` or `conda activate utrcalling`** before running tests, the CLI, or any code that imports the package. Do NOT invoke the environment's Python binary directly — that skips PATH setup and will cause missing-tool errors (e.g. BEDTools).

```bash
# Preferred for non-interactive commands (e.g. from Claude Code):
conda run --no-capture-output -n utrcalling <command>
# Or activate interactively:
conda activate utrcalling
```

### Install (development)

```bash
conda run --no-capture-output -n utrcalling pip install -e .
```

### Run all tests

```bash
conda run --no-capture-output -n utrcalling python -m unittest discover -s utrcalling/tests -p "test_*.py"
# Or with the CLI:
conda run --no-capture-output -n utrcalling utrcalling run-tests --verbosity 2
```

### Run a single test module

```bash
conda run --no-capture-output -n utrcalling python -m unittest utrcalling.tests.test_core
conda run --no-capture-output -n utrcalling python -m unittest utrcalling.tests.test_peak_calling
conda run --no-capture-output -n utrcalling python -m unittest utrcalling.tests.test_peak_calling_anndata
conda run --no-capture-output -n utrcalling python -m unittest utrcalling.tests.test_anndata_processing
conda run --no-capture-output -n utrcalling python -m unittest utrcalling.tests.test_process_mapped_reads
conda run --no-capture-output -n utrcalling python -m unittest utrcalling.tests.test_utr_mapping
```

### Run a single test class or method

```bash
conda run --no-capture-output -n utrcalling python -m unittest utrcalling.tests.test_core.TestCallPeaksSmartseq3LabelMinimal
conda run --no-capture-output -n utrcalling python -m unittest utrcalling.tests.test_core.TestCallPeaksSmartseq3LabelMinimal.test_shape_anndata
```

### CLI usage

```bash
conda run --no-capture-output -n utrcalling utrcalling --version
conda run --no-capture-output -n utrcalling utrcalling run --help          # Main pipeline
conda run --no-capture-output -n utrcalling utrcalling call-peaks --help   # Peak calling only
```

## Design History

The `core-overhaul` branch is a major refactor from `master`. Key changes: modin/ray replaced with pandas+joblib, HTSeq BAM reading replaced with pysam (lightweight `_IntervalData` class replaces `HTSeq.GenomicInterval`), flat intermediate data model (one row per molecule), sparse count matrices throughout, memory-aware barcode filtering, O(1) UMI deduplication, functional decomposition of core.py into thin orchestrator + focused modules (storage.py, barcode_filtering.py), type hints and Google-style docstrings on all public functions. Configuration uses `MergeMethod`/`PeakCallingMethod` enums and `PeakCallingResult` NamedTuple. Protocol base class consolidates `label_batches` defaults via `_default_label_batches` class attribute. Joblib config uses `JOBLIB_DEFAULT_N_JOBS`/`JOBLIB_DEFAULT_BACKEND` constants from `helpers.py`. Gene/transcript ID joins use `|` delimiter (bioinformatics convention). **291 tests pass.** See git history for details.

### Hard Constraints

- Never use `modin` or `ray`
- Never store Python lists inside DataFrame cells
- Never create per-UTR DataFrames inside a loop and concatenate them — accumulate arrays, build once
- Never concatenate DataFrames inside a loop — collect in a list, concatenate once at the end
- Never use `.apply(axis=1)` on large DataFrames — use vectorized operations or explicit loops with array accumulation
- Never materialize the count matrix as dense — use `scipy.sparse` from construction
- Never change behavioral test assertions (expected values/counts) without explicit justification

## Architecture

### Core Pipeline (`utrcalling/core/core.py`)

Thin orchestrator with 6 public functions — each reads like a recipe of named function calls from focused modules:

1. **`prepare_utr_reference_regions()`** — Parses GTF, extracts UTR features, merges overlapping regions via pybedtools, returns a reference BED-like DataFrame.
2. **`map_aligned_reads()`** — Maps reads from BAM files to UTR intervals via pysam (with adapter layer for protocol compatibility). Dispatches to `_map_single_bam()` or `_map_multiple_bams()` (joblib parallelism). Returns a flat DataFrame (or list of DataFrames for multi-BAM).
3. **`merge_samples_utr_sizes()`** — Merges multiple sample DataFrames via `process_mapped_reads.merge_samples_utr_sizes()`.
4. **`merge_utrs_per_feature()`** — Optionally aggregates per-UTR data to gene level via `process_mapped_reads.merge_utrs_per_gene()`.
5. **`filter_mapped_reads()`** — Subsets the UTR reference, filters UTR sizes below threshold, and optionally filters barcodes against a whitelist (with memory-aware chunking via `barcode_filtering.filter_barcodes_memory_aware()`). Returns `(filtered_reads, utr_reference_subset)`.
6. **`call_peaks()`** — Takes pre-filtered reads and subsetted reference (from `filter_mapped_reads`), calls consensus peaks on molecule size distributions using Gaussian KDE, enriches var with reference annotations, and produces the final AnnData with a sparse count matrix.

### Supporting Modules

- **`barcode_filtering.py`** — `filter_barcodes()` (fuzzy match via rapidfuzz) and `filter_barcodes_memory_aware()` (memory-aware chunking wrapper).
- **`storage.py`** — All intermediate file storage utilities (`store_utr_reference`, `store_read_mappings_*`, `store_merged_counts`, `store_peak_calling_results`).
- **`process_mapped_reads.py`** — Sample merging (`merge_samples_utr_sizes`) and gene-level aggregation (`merge_utrs_per_gene`). Helper functions `_join_unique()`, `_split_and_then_join_unique()`, `_compute_gene_offsets()` are module-level for testability.
- **`utr_reference.py`** — GTF parsing, UTR extraction, and `subset_reference_data()`.
- **`peak_calling.py`** — Sparse count matrix assembly (`generate_peak_counts_and_anno`), var annotation enrichment (`add_to_peaks_var`), AnnData construction (`generate_count_matrix_anndata`).
- **`utr_mapping.py`** — BAM reading via pysam, interval processing (`_IntervalData` for genomic intervals), UMI deduplication. `_PySamAdapter` bridges pysam reads to the protocol interface.
- **`helpers.py`** — Shared constants (`JOBLIB_DEFAULT_N_JOBS`, `JOBLIB_DEFAULT_BACKEND`), `ensure_mapped_reads_dtypes()` for canonical DataFrame schema enforcement, utility functions.

### Intermediate Data Model

The flat format used throughout the pipeline after `map_reads_to_utr()`:

```python
# One row per (utr, observation, utr_size) — categorical dtypes for string columns
mapped_reads = pd.DataFrame({
    "utr_name":        pd.Categorical([...]),
    "observation_id":  pd.Categorical([...]),
    "utr_size":        np.array([...], dtype=np.int32),
})
```

### Protocol System (`utrcalling/core/protocol.py`)

Abstract `Protocol` base class using `CombinedMeta(Meta, ABCMeta)` metaclass. Each protocol defines platform-specific read parsing. Factory: `protocol.create(platform, version, **kwargs)`.

| Protocol                          | Key methods                                                        | Identity source                |
| --------------------------------- | ------------------------------------------------------------------ | ------------------------------ |
| `ProtocolTenx3` / `ProtocolTenx2` | `get_reads_identity()` extracts barcode (16bp) + UMI from read seq | barcode + batch index          |
| `ProtocolSmartseq3`               | extracts barcode (6bp) + UMI from read seq                         | barcode + optional batch index |
| `ProtocolFiveTenx`                | reads CB/UB BAM tags                                               | CB tag + batch index           |
| `ProtocolCapseqSmartseq3*`        | regex/readname UMI extraction, uses BAM path as identity           | sample name from BAM path      |

`get_observation_id(barcode, batch_idx)` on `Protocol` base class applies the `-{batch_idx}` suffix when `label_batches=True`. CapSeq protocols override this to use the barcode (BAM-derived sample name) directly.

Peak calling config is mixed in from `utrcalling/core/protocols/peak_calling.py`.

### CLI Layer (`utrcalling/runtime.py` + `utrcalling/tools/`)

Argparse-based. Each module in `utrcalling/tools/` defines `create_parser(subparsers)`. The `main()` function dispatches via `args.func(**args_prepared)`.

### Data Flow

- **Input**: GTF annotation files (gzipped or plain) + BAM alignment files
- **Intermediate**: Parquet files in `./intermediate/` directory (optional, via `store=True`)
- **Output**: AnnData objects (.h5ad) with cells x UTR-peaks sparse count matrices

### dtype Guidance for Large Arrays

| Data                     | dtype                                  |
| ------------------------ | -------------------------------------- |
| UTR sizes (bp)           | `np.int32`                             |
| Read/count values        | `np.int32`                             |
| Genomic positions        | `np.int32`                             |
| Sparse matrix values     | `np.int32`                             |
| Barcodes/observation IDs | `pd.Categorical` or encoded `np.int32` |

### Key Dependencies

- **pybedtools** — genomic interval operations (requires **BEDTools** system-wide)
- **pysam** — BAM file reading in the hot loop (fully replaced HTSeq for read extraction and interval handling)
- **anndata/scanpy** — single-cell data structures
- **joblib** — parallelism across BAM files, across UTR intervals, and across UTRs during peak calling
- **rapidfuzz** — fuzzy barcode matching (Hamming distance ≤1)
- Python 3.9+ (compatible through 3.12+)

### Testing

Tests use Python's `unittest` framework (291 tests). Test data is generated synthetically in `utrcalling/tests/test_data/`. Protocol-specific integration tests live in `test_3'10x/`, `test_5'10x/`, `test_3'smartseq/`, `test_5'smartseq/`. Unit tests for helpers, deduplication, and observation ID logic live in `test_helpers.py`, `test_deduplicate.py`, and `test_observation_id.py`. Post-processing tests in `test_anndata_processing.py` and `test_peak_calling_anndata.py`. Module-level tests in `test_barcode_filtering.py`, `test_filter_mapped_reads.py`, and `test_storage.py`. CLI tests in `test_cli.py`. Error handling and negative tests in `test_error_handling.py`. End-to-end pipeline tests for all protocols (3' 10x v2/v3, 5' 10x, 5' CapSeq), dtype preservation tests, and determinism tests live in `test_core.py`.

Tests `chdir` into `utrcalling/tests/` at setup time, so test data paths are relative to that directory.

### Logging

Custom logger via `utrcalling/core/module_log.py` (coloredlogs + verboselogs). Logger initialization is guarded against duplicate setup from joblib worker processes — preserve this guard.

## Git Conventions

- Current active branch: `core-overhaul`; main branch for PRs: `master`
- Commit messages use gitmoji prefixes (e.g., `:bug:`, `:ambulance:`, `:sparkles:`, `:recycle:`, `:white_check_mark:`)
