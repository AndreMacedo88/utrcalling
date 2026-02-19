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

## Active Refactor (`core-overhaul` branch)

**`REFACTOR_PLAN.md` is the authoritative guide for all changes on this branch. It takes precedence over stylistic preferences or code brevity.** Read it before making any architectural changes.

### Refactor Goals (in priority order)

1. **Eliminate modin/ray entirely** — Replace `import modin.pandas as mpd` with `import pandas as pd` everywhere. Remove `init_ray()`, `check_if_modin_and_convert_to_pandas()`, and all modin isinstance checks.
2. **Fix count matrix assembly** — Replace the per-UTR DataFrame concat in `peak_calling.py:generate_peak_counts_and_anno()` with direct `scipy.sparse.coo_matrix` construction (accumulate `row`, `col`, `data` arrays across UTRs, build once at the end).
3. **Eliminate lists-in-DataFrame-cells** — `map_reads_to_utr()` currently returns a DataFrame where each row holds Python lists of barcodes/sizes. Replace with a flat format: one row per `(utr_name, observation_id, utr_size)`.
4. **Unify observation identity** — Add `get_observation_id(alignment, bam_name) -> str` to Protocol. For 10x: `{barcode}-{batch_idx}`. For SmartSeq: `{sample_name}`.
5. **Standardize parallelism on joblib** — Always `backend="loky"`, workers return plain data, main process concatenates once.

### Refactor Execution Order

1. Remove modin/ray → run tests
2. Change `map_reads_to_utr()` output to flat format → update all consumers → run tests
3. Rewrite `generate_peak_counts_and_anno()` with sparse assembly → run tests
4. Add `get_observation_id()`, simplify batch handling → run tests

### Hard Constraints (Never Do During Refactor)

- Never use `modin` or `ray`
- Never store Python lists inside DataFrame cells
- Never create per-UTR DataFrames inside a loop and concatenate them — accumulate arrays, build once
- Never concatenate DataFrames inside a loop — collect in a list, concatenate once at the end
- Never use `.apply(axis=1)` on large DataFrames — use vectorized operations or explicit loops with array accumulation
- Never materialize the count matrix as dense — use `scipy.sparse` from construction
- Never change behavioral test assertions (expected values/counts) without explicit justification

### Do Not Touch During Refactor

- `utrcalling/runtime.py` and `utrcalling/tools/` — CLI layer
- `utrcalling/core/module_log.py` — logging is correct
- `utrcalling/core/utr_reference.py` — GTF parsing is correct, not a bottleneck
- `utrcalling/core/anndata_processing.py` — post-processing on small-to-medium data
- `utrcalling/core/peak_calling_anndata.py` — secondary peak calling path
- `utrcalling/core/protocols/peak_calling.py` — KDE logic (sklearn + scipy) is efficient

## Architecture

### Three-Step Core Pipeline (`utrcalling/core/core.py`)

1. **`prepare_utr_reference_regions()`** — Parses GTF, extracts UTR features, merges overlapping regions via pybedtools, returns a reference BED-like DataFrame.
2. **`map_aligned_reads()`** — Maps reads from BAM files to UTR intervals via HTSeq. Parallelizes across samples using `joblib.Parallel(n_jobs=-2)`.
3. **`call_peaks()`** — Calls consensus peaks on molecule size distributions using Gaussian KDE (sklearn) + `scipy.signal.argrelextrema`. Produces the final AnnData.

### Protocol System (`utrcalling/core/protocol.py`)

Abstract `Protocol` base class using `CombinedMeta(Meta, ABCMeta)` metaclass. Each protocol defines platform-specific read parsing. Factory: `protocol.create(platform, version, **kwargs)`.

Supported protocols and their observation identity sources:

| Protocol | Key abstract methods | Identity source |
|---|---|---|
| `ProtocolTenx3` / `ProtocolTenx2` | `get_reads_identity()` extracts barcode (16bp) + UMI | barcode + batch index |
| `ProtocolSmartseq3` | extracts barcode (6bp) + UMI | barcode + sample |
| `ProtocolFiveTenx` | reads CB/UB BAM tags | CB tag + batch index |
| `ProtocolCapseq*` | 5' UTR variants | sample name from BAM path |

Peak calling config is mixed in from `utrcalling/core/protocols/peak_calling.py`.

### CLI Layer (`utrcalling/runtime.py` + `utrcalling/tools/`)

Argparse-based. Each module in `utrcalling/tools/` defines `create_parser(subparsers)`. The `main()` function dispatches via `args.func(**args_prepared)`. Available subcommands include: `run`, `call-peaks`, `prepare-reference-and-map-reads`, `concatenate-experiments`, `subset-experiments`, `select-alternative-transcription-events`, `compute-mean-utr-sizes`, `counts-to-percentages`, `estimate-best-bandwidth`, and others.

### Data Flow

- **Input**: GTF annotation files (gzipped or plain) + BAM alignment files
- **Intermediate**: Parquet/Feather files in `./intermediate/` directory (optional, via `store=True`)
- **Output**: AnnData objects (.h5ad) with cells × UTR-peaks sparse count matrices

### Intermediate Data Model (Post-Refactor Target)

The flat format replacing the legacy lists-in-cells DataFrame:

```python
# One row per (utr, observation, utr_size) — use categorical dtypes for string columns
mapped_reads = pd.DataFrame({
    "utr_name":        np.array([...], dtype="category"),
    "observation_id":  np.array([...], dtype="category"),
    "utr_size":        np.array([...], dtype=np.int32),
})
```

### dtype Guidance for Large Arrays

| Data | dtype |
|---|---|
| UTR sizes (bp) | `np.int32` |
| Read/count values | `np.int32` |
| Genomic positions | `np.int32` |
| Sparse matrix values | `np.int32` |
| Barcodes/observation IDs | `pd.Categorical` or encoded `np.int32` |

### Key Dependencies

- **pybedtools** — genomic interval operations (requires **BEDTools** installed system-wide: `conda install -c bioconda bedtools`)
- **HTSeq** — BAM/SAM read processing
- **anndata/scanpy** — single-cell data structures
- **joblib** — parallelism across BAM files
- **modin[ray]** — *being removed in this refactor*
- Python 3.9+ (tested on 3.9)

### Testing

Tests use Python's `unittest` framework. Test data is generated synthetically in `utrcalling/tests/test_data/`. Protocol-specific integration tests live in `test_3'10x/`, `test_5'10x/`, `test_3'smartseq/`, `test_5'smartseq/`.

When updating tests during the refactor: preserve expected output *values*, update expected DataFrame *shapes/structures* to match the new flat format.

### Logging

Custom logger via `utrcalling/core/module_log.py` (coloredlogs + verboselogs). Logger initialization is guarded against duplicate setup from joblib worker processes — preserve this guard.

## Git Conventions

- Current active branch: `core-overhaul`; main branch for PRs: `master`
- Commit messages use gitmoji prefixes (e.g., `:bug:`, `:ambulance:`, `:sparkles:`)
