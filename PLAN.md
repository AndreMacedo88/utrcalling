# PLAN.md

# Post-Refactor Improvement Plan for utrcalling

## Context

The `core-overhaul` refactor is complete (modin removed, flat data model, sparse matrix assembly). The codebase needs cleanup of leftover refactor artifacts, performance fixes for remaining bottlenecks, and improvements to code quality and test coverage. This plan addresses those in 6 independent phases, ordered from low-risk/high-impact to higher-risk, each mergeable separately with passing tests.

---

## Phase 1: Cleanup — Dead Code, Deprecations, Packaging Fixes ✅

**Goal**: Remove refactor leftovers, fix a Python 3.12 blocker, and correct packaging bugs.

**Status**: Complete. 136 tests pass.

### Changes

| File | Change |
|------|--------|
| `utrcalling/core/helpers.py` | Remove dead functions: `hamming_distance()` (L136-141), `count_duplicates()` (L144-170), `check_if_modin_and_convert_to_pandas()` (L187-188) — none are called anywhere |
| `utrcalling/core/utr_mapping.py` | Remove dead function `deduplicate_reads_with_barcode_filtering()` (L153-166, never called). Remove commented-out old code block (L222-251, 30 lines of dead code in triple-quoted string) |
| `utrcalling/core/anndata_processing.py` | Remove unused import `from cmath import isnan` (L2) |
| `utrcalling/core/protocol.py` | Replace deprecated `from distutils.util import strtobool` (L16) with an inline `_strtobool()` function. `distutils` was removed in Python 3.12, so this is a blocker for newer Python versions. Used at L83. |
| `utrcalling/core/utr_reference.py` | Fix mutable default arguments in `reduce_regions()` (L57): change `annotations_index=[], operation_list=[]` to `None` with body defaults. |
| `utrcalling/__init__.py` | Remove vestigial `"ray::IDLE"` check (L30) — ray is no longer used. Keep the joblib check. |
| `requirements.txt` | Remove `modin[ray]>=0.19.0` (L3). Fix typo `pymper` → `pympler` (L17). |
| `setup.py` | Remove broken `scripts=` glob (L68, `./tools/` doesn't exist at root — scripts are installed via `entry_points` already). Relax version pins: `numpy>=1.22.3` (drop `<1.24`), `pybedtools>=0.8.2` (drop exact pin), `bidict>=0.22.0` (drop exact pin). |

### Verification
```bash
python -m unittest discover -s utrcalling/tests -p "test_*.py"
pip install -e .
```

---

## Phase 2: Performance — Fix O(n²) `deduplicate_reads()` ✅

**Goal**: Fix the single biggest per-BAM bottleneck. Currently uses Python lists for UMI lookup, making `if umi not in list` O(n) per call — quadratic overall.

**Status**: Complete. 144 tests pass (136 existing + 8 new).

### Changes

| File | Change |
|------|--------|
| `utrcalling/core/utr_mapping.py` | In `deduplicate_reads()` (L140-149): change `umis_unique` values from `list` to `set`, `.append()` to `.add()`. Remove unnecessary `.copy()` on `read_umis`. Use local variable to avoid repeated dict lookup. |
| `utrcalling/core/utr_mapping.py` | Gate the `pympler.asizeof()` debug call (L131-135) behind `logger.isEnabledFor(logging.DEBUG)` — currently traverses the entire DataFrame on every BAM file even when debug logging is off. |
| `utrcalling/tests/test_deduplicate.py` | **New file.** Unit tests for `deduplicate_reads()`: basic dedup, different barcodes with same UMI, empty input, all-duplicates, order preservation. |

### Verification
```bash
python -m unittest utrcalling.tests.test_deduplicate
python -m unittest discover -s utrcalling/tests -p "test_*.py"
```

---

## Phase 3: Performance — Memory Heuristic and Unnecessary Copies ✅

**Goal**: Replace the fragile hard-coded memory splitting (30% threshold, always 10 chunks) with a data-size-aware approach, and remove redundant DataFrame copies.

**Status**: Complete. 144 tests pass.

### Changes

| File | Change |
|------|--------|
| `utrcalling/core/core.py` | In `call_peaks()` (L167-189): Replace the `psutil.virtual_memory().percent > 30` check with a comparison of estimated data size vs available memory. Scale chunk count dynamically instead of hard-coded 10. Re-categorize dtypes after `pd.concat`. |
| `utrcalling/core/process_mapped_reads.py` | In `filter_barcodes()` (L249-257): Remove the first unconditional `.copy()` (L250). Only copy when mutation is needed (`barcodes_replace` is non-empty). Re-categorize `utr_name` after boolean filter (currently only re-categorizes `observation_id`). |

### Verification
```bash
python -m unittest discover -s utrcalling/tests -p "test_*.py"
```

---

## Phase 4: Type Hints and Docstrings ✅

**Goal**: Add type annotations and docstrings to all public functions in the 5 core pipeline modules. Use `from __future__ import annotations` for Python 3.9 compatibility.

**Status**: Complete. 144 tests pass. All 6 modules annotated with `from __future__ import annotations`, typed signatures, and Google-style docstrings.

### Modules (in order)

1. **`utrcalling/core/protocol.py`** — `Protocol.__init__()`, `get_observation_id()`, `get_reads_identity()`, `filter_utr_sizes()`, `check_strand_matches()`, `get_utr_size_from_read()`, `create()`, all concrete protocol `get_reads_identity()` implementations
2. **`utrcalling/core/core.py`** — `prepare_utr_reference_regions()`, `map_aligned_reads()`, `merge_samples_utr_sizes()`, `merge_utrs_per_feature()`, `call_peaks()`
3. **`utrcalling/core/utr_mapping.py`** — `generate_utr_intervals()`, `map_reads_to_utr()`, `deduplicate_reads()`, `extract_read_information()`
4. **`utrcalling/core/process_mapped_reads.py`** — `merge_samples_utr_sizes()`, `merge_utrs_per_gene()`, `subset_reference_data()`, `filter_barcodes()`
5. **`utrcalling/core/peak_calling.py`** — `generate_peak_counts_and_anno()`, `add_to_peaks_var()`, `generate_count_matrix_anndata()`
6. **`utrcalling/core/helpers.py`** — All remaining public functions

### Docstring format
Google-style (Args/Returns), matching the few existing docstrings (e.g., `extract_read_information()`).

### Verification
```bash
python -m unittest discover -s utrcalling/tests -p "test_*.py"
python -c "from utrcalling.core import core, protocol, utr_mapping, process_mapped_reads, peak_calling"
```

---

## Phase 5: Testing Improvements ✅

**Goal**: Fill the most critical testing gaps — unit tests for functions that have zero direct coverage, and edge case tests for data handling.

**Status**: Complete. 182 tests pass (144 existing + 38 new).

### New test files and additions

| File | What to test |
|------|-------------|
| `utrcalling/tests/test_helpers.py` | **New.** `flatten()`, `to_string()`, `check_if_path_and_open()`, `check_file_exists_and_return_unique()` |
| `utrcalling/tests/test_observation_id.py` | **New.** `Protocol.get_observation_id()` for Tenx (with/without batch), SmartSeq, CapSeq |
| `utrcalling/tests/test_peak_calling.py` | **Append.** Edge cases for `generate_peak_counts_and_anno()`: empty DataFrame input, single observation, single UTR |
| `utrcalling/tests/test_process_mapped_reads.py` | **Append.** Edge cases for `merge_samples_utr_sizes()`: single sample, no common UTRs (intersection), invalid method. Edge cases for `subset_reference_data()`: no matching UTRs |

### Verification
```bash
python -m unittest discover -s utrcalling/tests -p "test_*.py"
```

---

## Phase 6: Performance — Parallelize Peak Calling ✅

**Goal**: KDE peak calling in `generate_peak_counts_and_anno()` iterates sequentially over UTRs, but each UTR is independent. Parallelize with joblib (already used in `map_aligned_reads()`).

**Status**: Complete. 182 tests pass.

### Changes

| File | Change |
|------|--------|
| `utrcalling/core/peak_calling.py` | Extract the per-UTR loop body into a `_process_utr_group()` worker function. Use `joblib.Parallel(n_jobs=-2, backend="loky")` when `n_utrs > 50` (avoids overhead on small test datasets). Workers return `(rows, data, peak_names, peak_sizes)` tuples. Main process assembles sparse matrix from all results. |

### Key design decisions
- **Threshold**: Only parallelize when >50 UTRs (test data has ~5-10, real data has thousands)
- **Serialization**: `obs_to_idx` dict (str→int) and `func` (module-level function ref) are both picklable
- **Assembly**: Sparse matrix construction stays in the main process — workers return plain numpy arrays

### Verification
```bash
python -m unittest discover -s utrcalling/tests -p "test_*.py"
python -m utrcalling.performance.perf_call_peaks  # measure speedup
```

---

## Summary

| Phase | Focus | Risk | Key benefit | Status |
|-------|-------|------|-------------|--------|
| 1 | Cleanup & packaging | Very low | Python 3.12 compat, remove dead code, fix packaging | ✅ Done |
| 2 | O(n²) dedup fix | Low | Major speedup on high-depth BAMs | ✅ Done |
| 3 | Memory heuristic & copies | Low-Med | Better RAM usage, smarter splitting | ✅ Done |
| 4 | Type hints & docstrings | None | Maintainability, IDE support | ✅ Done |
| 5 | Test coverage | None | Reliability, catch regressions | ✅ Done |
| 6 | Parallel peak calling | Medium | Multi-core speedup on real datasets | ✅ Done |

**All 6 phases complete. 182 tests pass.**
