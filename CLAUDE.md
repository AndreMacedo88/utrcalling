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

## Completed Improvement Plans

The `core-overhaul` branch completed a major refactor from `master`, followed by a 6-phase improvement plan (see below), a cleanup pass on the two remaining post-processing modules, and a pysam-based performance rewrite of the BAM read loop. **218 tests pass.**

### 6-Phase Improvement Plan (complete)

1. **modin/ray eliminated** — All modin/ray usage replaced with plain pandas + joblib. Dead modin helpers, ray checks, and stale `requirements.txt` references cleaned up (Phase 1).
2. **Flat intermediate data model** — `map_reads_to_utr()` returns a flat DataFrame (one row per molecule: `utr_name`, `observation_id`, `utr_size`).
3. **Sparse count matrix assembly** — `generate_peak_counts_and_anno()` builds `scipy.sparse.csr_matrix` directly from accumulated `(row, col, data)` arrays.
4. **Unified observation identity** — `Protocol.get_observation_id(barcode, batch_idx)` constructs the final cell/sample ID at read time.
5. **joblib parallelism standardized** — BAM-level parallelism in `map_aligned_reads()` (workers return flat DataFrames, main process concatenates once). UTR-level parallelism in `generate_peak_counts_and_anno()` when >50 UTRs (Phase 6).
6. **O(n²) UMI deduplication fixed** — `deduplicate_reads()` now uses `set` for O(1) UMI lookup instead of `list` (Phase 2).
7. **Python 3.12 compatibility** — Deprecated `distutils.strtobool` replaced with inline `_strtobool()` in `protocol.py` (Phase 1).
8. **Memory-aware barcode filtering** — `call_peaks()` compares estimated data size vs available RAM instead of a hard-coded 30% threshold, scales chunk count dynamically (Phase 3).
9. **Unnecessary DataFrame copies removed** — `filter_barcodes()` skips the full `.copy()` when no barcode replacements are needed; re-categorizes both `utr_name` and `observation_id` after filtering (Phase 3).
10. **Type hints and docstrings** — All public functions in the 6 core modules annotated with `from __future__ import annotations`, typed signatures, and Google-style docstrings (Phase 4).
11. **Test coverage expanded** — 182 tests total (up from 136). New test files for `helpers.py` and `Protocol.get_observation_id()`. Edge case tests for peak calling and process_mapped_reads (Phase 5).

### anndata_processing.py & peak_calling_anndata.py Cleanup (complete)

Brought both post-processing modules up to the same standard as the rest of the codebase:

12. **anndata_processing.py cleaned up** — Dead imports removed (`warnings`, `joblib`, `helpers`). Type hints and Google-style docstrings added to all 7 functions. `counts_to_percentages()` made sparse-safe. Dead code removed from `compute_utr_size_means()`. `AttributeError` → `ValueError` in `subset_adata()`. `split_train_test_data()` fixed to use `None` sentinel instead of `False` for missing batch column.
13. **peak_calling_anndata.py refactored** — Dead imports removed (`os`, `HTTPError`, `time`). Legacy code block and commented-out debug code deleted. Module-level logger replaced with per-function calls. Shared per-UTR peak calling logic extracted into `_peak_call_single_utr()`. Nested joblib functions (`peak_call`, `get_distances_per_utr`, `get_distances`) extracted to module-level functions for picklability. `**kwargs` replaced with explicit parameters (`bandwidth`, `min_umis`). `_find_pas()` regex compilation moved outside the while loop; `hex` renamed to `pas_pattern`. `_calculate_umi_distance_to_pas()` vectorised with numpy (replaced `.iterrows()`). `_utr_sizes_vector_from_read_counts()` fixed for sparse input and now returns `pd.Series` preserving var-name index. `_call_peaks_using_kernel_density_estimator()` dtype changed from `uint32` to `np.int32`. `remove_peaks_below_count()` made sparse-safe (avoids full dense materialisation).
14. **Test coverage expanded** — 218 tests total (up from 182). New `test_peak_calling_anndata.py` (23 tests). Expanded `test_anndata_processing.py` with 13 new tests for `select_alternative_transcription_events`, `counts_to_percentages`, `subset_adata`, `concatenate_adata`.

### pysam Performance Rewrite (complete)

Replaced HTSeq's BAM reading with pysam in the hot loop of `map_reads_to_utr()`, plus micro-optimizations and interval-level parallelism:

15. **pysam BAM reading** — `pysam.AlignmentFile.fetch()` replaces `HTSeq.BAM_Reader[interval]`. pysam is a lighter-weight C wrapper and avoids HTSeq's heavyweight alignment object creation. Added as an explicit dependency in `setup.py` (was already a transitive dep of HTSeq).
16. **Adapter layer for protocol compatibility** — `_PySamAdapter` / `_PySamReadAdapter` classes make pysam `AlignedSegment` objects quack like HTSeq `SAM_Alignment` objects. Critically, the adapter reverse-complements `query_sequence` for reverse-strand reads to match HTSeq's convention of returning the original read orientation (pysam returns the stored/aligned orientation). Uses `__slots__` for minimal overhead.
17. **Bitmask flag filtering** — Per-read flag checks (pe_which, proper_pair, pcr_duplicate, not_primary) collapsed from 4 property accesses into 2 integer operations on the fast path. Stats categorization only runs for rejected reads.
18. **Inlined read extraction** — `extract_read_information()` logic inlined into the hot loop, eliminating per-read function call overhead. The function is preserved for backward compatibility but is no longer called from the hot path.
19. **Local variable caching** — Protocol method lookups (`check_strand_matches`, `get_reads_identity`, `get_utr_size_from_read`, `get_observation_id`) cached as local variables. Interval fields (`chrom`, `start`, `end`, `strand`, `start_d`, `end_d`) pre-extracted once per interval.
20. **`to_string()` updated** — `helpers.to_string()` now short-circuits for `str` input (pysam returns `str` natively, HTSeq returned `bytes`).
21. **Interval-level parallelism** — `map_reads_to_utr()` accepts `parallel=True` (default). When >50 intervals, splits work across CPU cores via `joblib.Parallel`. Each worker opens its own `pysam.AlignmentFile` handle. `_IntervalData` snapshots make HTSeq intervals picklable for joblib. Multi-BAM path in `core.py` passes `parallel=False` to prevent nested joblib.
22. **Shared `_process_interval_batch()`** — Both sequential and parallel paths call the same top-level function, avoiding code duplication. Sequential path runs it as a single batch with tqdm progress; parallel path splits into batches.

### Hard Constraints

- Never use `modin` or `ray`
- Never store Python lists inside DataFrame cells
- Never create per-UTR DataFrames inside a loop and concatenate them — accumulate arrays, build once
- Never concatenate DataFrames inside a loop — collect in a list, concatenate once at the end
- Never use `.apply(axis=1)` on large DataFrames — use vectorized operations or explicit loops with array accumulation
- Never materialize the count matrix as dense — use `scipy.sparse` from construction
- Never change behavioral test assertions (expected values/counts) without explicit justification

### Do Not Touch

- `utrcalling/runtime.py` and `utrcalling/tools/` — CLI layer
- `utrcalling/core/module_log.py` — logging is correct
- `utrcalling/core/utr_reference.py` — GTF parsing is correct, not a bottleneck (minor bugfixes like mutable defaults are OK)
- `utrcalling/core/protocols/peak_calling.py` — KDE logic (sklearn + scipy) is efficient

## Architecture

### Three-Step Core Pipeline (`utrcalling/core/core.py`)

1. **`prepare_utr_reference_regions()`** — Parses GTF, extracts UTR features, merges overlapping regions via pybedtools, returns a reference BED-like DataFrame.
2. **`map_aligned_reads()`** — Maps reads from BAM files to UTR intervals via pysam (with HTSeq adapters for protocol compatibility). Parallelizes across samples using `joblib.Parallel(n_jobs=-2)` for multi-BAM, and across intervals within a single BAM when >50 intervals. Returns a flat DataFrame (or list of DataFrames for multi-BAM).
3. **`call_peaks()`** — Filters UTR sizes, optionally filters barcodes (with memory-aware chunking), then calls consensus peaks on molecule size distributions using Gaussian KDE (sklearn) + `scipy.signal.argrelextrema`. Peak calling parallelizes across UTRs with `joblib.Parallel` when >50 UTRs. Produces the final AnnData with a sparse count matrix.

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

| Protocol | Key methods | Identity source |
|---|---|---|
| `ProtocolTenx3` / `ProtocolTenx2` | `get_reads_identity()` extracts barcode (16bp) + UMI from read seq | barcode + batch index |
| `ProtocolSmartseq3` | extracts barcode (6bp) + UMI from read seq | barcode + optional batch index |
| `ProtocolFiveTenx` | reads CB/UB BAM tags | CB tag + batch index |
| `ProtocolCapseqSmartseq3*` | regex/readname UMI extraction, uses BAM path as identity | sample name from BAM path |

`get_observation_id(barcode, batch_idx)` on `Protocol` base class applies the `-{batch_idx}` suffix when `label_batches=True`. CapSeq protocols override this to use the barcode (BAM-derived sample name) directly.

Peak calling config is mixed in from `utrcalling/core/protocols/peak_calling.py`.

### CLI Layer (`utrcalling/runtime.py` + `utrcalling/tools/`)

Argparse-based. Each module in `utrcalling/tools/` defines `create_parser(subparsers)`. The `main()` function dispatches via `args.func(**args_prepared)`.

### Data Flow

- **Input**: GTF annotation files (gzipped or plain) + BAM alignment files
- **Intermediate**: Parquet files in `./intermediate/` directory (optional, via `store=True`)
- **Output**: AnnData objects (.h5ad) with cells x UTR-peaks sparse count matrices

### dtype Guidance for Large Arrays

| Data | dtype |
|---|---|
| UTR sizes (bp) | `np.int32` |
| Read/count values | `np.int32` |
| Genomic positions | `np.int32` |
| Sparse matrix values | `np.int32` |
| Barcodes/observation IDs | `pd.Categorical` or encoded `np.int32` |

### Key Dependencies

- **pybedtools** — genomic interval operations (requires **BEDTools** system-wide)
- **pysam** — BAM file reading in the hot loop (replaced HTSeq for read extraction)
- **HTSeq** — still used for `GenomicInterval` creation in `generate_utr_intervals()`
- **anndata/scanpy** — single-cell data structures
- **joblib** — parallelism across BAM files, across UTR intervals, and across UTRs during peak calling
- **rapidfuzz** — fuzzy barcode matching (Hamming distance ≤1)
- Python 3.9+ (compatible through 3.12+)

### Testing

Tests use Python's `unittest` framework (218 tests). Test data is generated synthetically in `utrcalling/tests/test_data/`. Protocol-specific integration tests live in `test_3'10x/`, `test_5'10x/`, `test_3'smartseq/`, `test_5'smartseq/`. Unit tests for helpers, deduplication, and observation ID logic live in `test_helpers.py`, `test_deduplicate.py`, and `test_observation_id.py`. Post-processing tests in `test_anndata_processing.py` and `test_peak_calling_anndata.py`.

Tests `chdir` into `utrcalling/tests/` at setup time, so test data paths are relative to that directory.

### Logging

Custom logger via `utrcalling/core/module_log.py` (coloredlogs + verboselogs). Logger initialization is guarded against duplicate setup from joblib worker processes — preserve this guard.

## Git Conventions

- Current active branch: `core-overhaul`; main branch for PRs: `master`
- Commit messages use gitmoji prefixes (e.g., `:bug:`, `:ambulance:`, `:sparkles:`, `:recycle:`, `:white_check_mark:`)
