# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**utrcalling** is a bioinformatics Python package for calculating molecule UTR (Untranslated Region) sizes from RNA sequencing reads. It processes BAM alignment files against GTF gene annotations to produce AnnData (.h5ad) count matrices of UTR-length usage per cell/sample.

## Common Commands

### Install (development)

```bash
pip install -e . --use-feature=in-tree-build
```

### Run all tests

```bash
utrcalling run-tests --verbosity 2
# Or without the CLI:
python -m unittest discover -s utrcalling/tests -p "test_*.py"
```

### Run a single test module

```bash
python -m unittest utrcalling.tests.test_core
python -m unittest utrcalling.tests.test_peak_calling
```

### CLI usage

```bash
utrcalling --version
utrcalling run --help          # Main pipeline
utrcalling call-peaks --help   # Peak calling only
```

## Architecture

### Three-Step Core Pipeline (`utrcalling/core/core.py`)

1. **`prepare_utr_reference_regions()`** — Parses a GTF file, extracts UTR features, merges overlapping regions using pybedtools, and returns a reference BED-like DataFrame.
2. **`map_aligned_reads()`** — Maps RNA-seq reads from BAM files to UTR intervals using HTSeq. Supports parallelization via joblib across multiple samples.
3. **`call_peaks()`** — Calls consensus peaks on molecule size distributions using Kernel Density Estimation (sklearn's KernelDensity with Gaussian kernel). Finds local extrema via `scipy.signal.argrelextrema`.

### Protocol System (`utrcalling/core/protocol.py`)

Abstract `Protocol` base class (using a combined `ABCMeta` + custom `Meta` metaclass) defines the interface for sequencing platform support. Each protocol specifies how to extract read info, handle sample names, and configure peak calling. Supported protocols:

- 3' 10x Genomics (single-cell, barcode-based)
- 5' 10x Genomics
- 3' SmartSeq (bulk/plate-based)
- 5' SmartSeq

Peak calling method implementations live in `utrcalling/core/protocols/peak_calling.py` and are mixed into protocol classes.

### CLI Layer (`utrcalling/runtime.py` + `utrcalling/tools/`)

Argparse-based CLI with subcommands. Each tool module in `utrcalling/tools/` defines a `create_parser(subparsers)` function that registers its subcommand. The `runtime.main()` function dispatches to `args.func(**args_prepared)`.

### Data Flow

- **Input**: GTF annotation files (gzipped or plain) + BAM alignment files
- **Intermediate**: Parquet/Feather files in `./intermediate/` directory (optional, via `store=True`)
- **Output**: AnnData objects (.h5ad) with cells x UTR-peaks count matrices

### Key Dependencies

- **modin[ray]** for distributed pandas — `import modin.pandas as mpd` is used instead of pandas in core modules
- **pybedtools** for genomic interval operations (requires BEDTools installed on the system)
- **HTSeq** for BAM/SAM read processing
- **anndata/scanpy** for single-cell data structures

### Testing

Tests use Python's built-in `unittest` framework. Test data is generated synthetically by helper modules in `utrcalling/tests/test_data/`. Protocol-specific integration tests live in subdirectories named `test_3'10x/`, `test_5'10x/`, `test_3'smartseq/`, `test_5'smartseq/`.

### Logging

Custom logger via `utrcalling/core/module_log.py` using `coloredlogs` and `verboselogs`. Logger initialization is guarded against duplicate setup from joblib/ray worker processes.

## Git Conventions

- Current active branch: `core-overhaul`; main branch for PRs: `master`
- Commit messages use gitmoji prefixes (e.g., `:bug:`, `:ambulance:`, `:sparkles:`)
