# PLAN.md — Functional Decomposition of `core/core.py`

## Goal

Refactor `core/core.py` so that each of its public functions becomes a thin, readable orchestrator that calls focused, single-responsibility functions from appropriately named modules. Extract embedded logic (barcode filtering with memory-aware chunking, intermediate storage, filtering steps buried inside `call_peaks()`) into dedicated modules and new public pipeline functions. Update all callers (`tools/run.py`, `tools/call_peaks.py`, `tools/prepare_reference_and_map_reads.py`, `test_core.py`) to match any API changes.

---

## Current State

`core/core.py` has 5 public functions:

| Function | Lines | Embedded concerns |
|---|---|---|
| `prepare_utr_reference_regions()` | 46–83 | GTF parsing pipeline + intermediate file storage |
| `map_aligned_reads()` | 86–151 | Single/multi-BAM dispatch, joblib parallelism, batch index resolution, intermediate storage |
| `merge_samples_utr_sizes()` | 154–184 | Delegation to `process_mapped_reads` + intermediate storage |
| `merge_utrs_per_feature()` | 187–210 | Conditional delegation to `process_mapped_reads` |
| `call_peaks()` | 213–361 | Reference subsetting, UTR size filtering, memory-aware barcode chunking, peak calling, var/obs assembly, intermediate storage, AnnData construction |

### Problems

1. **`call_peaks()` is ~150 lines and does 6+ distinct things** — subsetting reference data, filtering UTR sizes, memory-aware barcode filtering with chunking, calling peaks, joining annotations, storing intermediates, and building the AnnData. These are conceptually independent steps glued together.
2. **Filtering is hidden inside peak calling** — reference subsetting, UTR size filtering, and barcode filtering are pre-processing steps that have nothing to do with calling peaks. They should be a separate pipeline step, visible at the orchestration level in `tools/run.py`.
3. **Memory-aware barcode filtering** (lines 248–287) is a ~40-line block that estimates memory, decides chunking, splits data, calls `filter_barcodes()` per chunk, and re-concatenates. This is a standalone concern.
4. **Intermediate file storage** is scattered as `if store:` blocks inside every function. This cross-cutting concern obscures the core data flow.
5. **`map_aligned_reads()` handles both single-BAM and multi-BAM dispatch** with batch index resolution logic, which is orchestration-level work that could be more explicit.
6. **Some modules host functions that conceptually belong elsewhere** — e.g., `process_mapped_reads.py` contains `subset_reference_data()` (a reference data operation) and `filter_barcodes()` (a barcode/observation filtering operation) alongside sample merging and gene-level aggregation.

---

## Design Principles

1. **core.py functions should read like a recipe** — each function body should be a sequence of clearly named function calls, with minimal inline logic.
2. **Each helper module should do "one thing"** — functions within a module should share a single domain.
3. **API changes are allowed** — if splitting a function or changing signatures makes the pipeline clearer, do it and update all callers (`tools/*.py`, `test_core.py`).
4. **No behavioral changes** — all 218 tests must pass without changing any assertion values. Test code may change (calling a new function name or passing different args), but expected outputs stay the same.

---

## Phase 0: Update CLAUDE.md — Relax "Do Not Touch" Constraints

### Changes to CLAUDE.md `### Do Not Touch` section

Remove these entries:
- `utrcalling/runtime.py` and `utrcalling/tools/` — these will be updated to reflect core.py API changes
- `utrcalling/core/utr_reference.py` — will receive `subset_reference_data()` from `process_mapped_reads.py`

Keep these entries:
- `utrcalling/core/module_log.py` — logging is correct
- `utrcalling/core/protocols/peak_calling.py` — KDE logic is efficient

Updated section:
```markdown
### Do Not Touch

- `utrcalling/core/module_log.py` — logging is correct
- `utrcalling/core/protocols/peak_calling.py` — KDE logic (sklearn + scipy) is efficient
```

---

## Phase 1: Create `core/storage.py` — Extract Intermediate File Storage

### Rationale
Every function in `core.py` has an `if store:` block that creates directories, saves parquet/npz/csv files. This is a cross-cutting concern that clutters the pipeline logic. Extracting it makes each pipeline function shorter and focused on data transformation.

### New file: `utrcalling/core/storage.py`

```python
"""Intermediate file storage utilities for the UTR calling pipeline."""

def store_utr_reference(utr_bed, utr_reduced, directory="./intermediate"):
    """Save unmerged BED and merged parquet UTR reference files."""

def store_read_mappings_multi(read_mappings_list, directory="./intermediate", filename="UTR_sizes_counts"):
    """Save per-sample read mapping parquet files (multi-BAM)."""

def store_read_mappings_single(read_mappings, directory="./intermediate", filename="UTR_sizes_counts"):
    """Save a single read mapping parquet file (single-BAM)."""

def store_merged_counts(utr_sizes_counts, merge_method, directory="./intermediate"):
    """Save merged UTR sizes counts parquet file."""

def store_peak_calling_results(peaks_count_matrix, peaks_var, peaks_obs, utr_reference_subset, merge_method, directory="./intermediate"):
    """Save peak calling intermediate files (sparse matrix, var, obs, reference subset)."""
```

### Changes to `core/core.py`
Each function's `if store:` block is replaced with a single call:
```python
if store:
    storage.store_utr_reference(utr_bed, utr_reduced)
```

### Impact
- `core.py`: ~60 lines of `if store:` blocks moved out
- No signature changes, no behavioral changes
- Tests unaffected (storage is only invoked when `store=True`, which tests don't use)

---

## Phase 2: Create `core/barcode_filtering.py` — Extract Barcode Filtering

### Rationale
`filter_barcodes()` currently lives in `process_mapped_reads.py` but is conceptually a barcode/observation filtering operation, not a "mapped reads processing" operation. The memory-aware chunking wrapper (currently 40 lines inside `call_peaks()`) is a standalone concern on top of it.

### New file: `utrcalling/core/barcode_filtering.py`

```python
"""Barcode whitelist filtering with memory-aware chunking."""

def filter_barcodes(utr_sizes, whitelist):
    """Filter observations to those matching a barcode whitelist (fuzzy match, Hamming ≤1).

    Exact matches are kept directly. For non-matching barcodes, a fuzzy match
    (Hamming distance <= 1) is attempted via rapidfuzz.
    """
    # Moved from process_mapped_reads.filter_barcodes() — exact same code

def filter_barcodes_memory_aware(filtered_positions, whitelist):
    """Apply barcode filtering with automatic memory-aware chunking.

    Estimates whether barcode filtering will fit in available RAM.
    If not, splits the data into chunks, filters each chunk, and
    re-concatenates. Otherwise, filters in a single pass.
    """
    # Extracted from call_peaks() lines 248-287
```

### Changes to `process_mapped_reads.py`
- Remove `filter_barcodes()` function body
- Add a re-export for backward compatibility:
  ```python
  from utrcalling.core.barcode_filtering import filter_barcodes
  ```
  This preserves the import path for `test_process_mapped_reads.py` and `test_peak_calling.py`.

### New tests: `utrcalling/tests/test_barcode_filtering.py`
- `test_filter_barcodes_memory_aware_small_data` — verify single-pass path
- `test_filter_barcodes_memory_aware_preserves_dtypes` — verify categorical dtypes after filtering
- `test_filter_barcodes_exact_match` — ensures function works from new module location

### Impact
- `process_mapped_reads.py` shrinks by ~70 lines
- Existing tests continue to work via re-export
- 3 new tests

---

## Phase 3: Move `subset_reference_data()` to `core/utr_reference.py`

### Rationale
`process_mapped_reads.subset_reference_data()` takes a read DataFrame and a UTR reference DataFrame, and returns a subset of the reference. It's a reference data operation, not a read processing operation. It belongs in `utr_reference.py`.

### Changes
- Move `subset_reference_data()` from `process_mapped_reads.py` to `utr_reference.py`
- Add re-export in `process_mapped_reads.py` for backward compatibility:
  ```python
  from utrcalling.core.utr_reference import subset_reference_data
  ```
- `core.py` will call `utr_reference.subset_reference_data()` directly

### Impact
- `process_mapped_reads.py` loses 1 function (~22 lines)
- `utr_reference.py` gains 1 function (additive, no existing code changed)
- No test changes needed (re-export preserves import path)

---

## Phase 4: Split `call_peaks()` — Extract `filter_mapped_reads()` as New Public API

### Rationale
`call_peaks()` currently starts with 3 filtering/preparation steps (subset reference, filter UTR sizes, filter barcodes) before doing any actual peak calling. These are conceptually independent from peak calling — they prepare the data. Extracting them into `filter_mapped_reads()` makes the pipeline in `tools/run.py` more granular and readable:

```python
# Before (filtering hidden inside call_peaks):
count_matrix_anndata = core.call_peaks(utr_sizes_counts, utr_reference_regions, protocol_)

# After (filtering is an explicit pipeline step):
filtered_reads, utr_reference_subset = core.filter_mapped_reads(
    utr_sizes_counts, utr_reference_regions, protocol_)
count_matrix_anndata = core.call_peaks(
    filtered_reads, utr_reference_subset, protocol_)
```

### New public function in `core/core.py`:

```python
def filter_mapped_reads(
    utr_sizes_counts: pd.DataFrame,
    utr_reference_regions: pd.DataFrame,
    protocol: object,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter mapped reads and subset the UTR reference before peak calling.

    Applies three preparation steps:
    1. Subset the UTR reference to only UTRs present in the read data.
    2. Filter out UTR sizes below the threshold.
    3. Optionally filter barcodes against a whitelist (memory-aware).

    Args:
        utr_sizes_counts: Flat DataFrame of (utr_name, observation_id, utr_size) rows.
        utr_reference_regions: Full UTR reference DataFrame.
        protocol: A Protocol instance with filtering config.

    Returns:
        A (filtered_reads, utr_reference_subset) tuple.
    """
    utr_reference_subset = utr_reference.subset_reference_data(
        utr_sizes_counts, utr_reference_regions
    )
    filtered = protocol.filter_utr_sizes(utr_sizes_counts, threshold=1)
    if protocol.barcode_whitelist is not None:
        filtered = barcode_filtering.filter_barcodes_memory_aware(
            filtered, protocol.barcode_whitelist
        )
    return filtered, utr_reference_subset
```

### Slimmed `call_peaks()`:

```python
def call_peaks(
    filtered_reads: pd.DataFrame,
    utr_reference_subset: pd.DataFrame,
    protocol: object,
    store: bool = False,
) -> anndata.AnnData:
    """Call consensus UTR peaks and build a sparse AnnData count matrix.

    Expects pre-filtered reads and a subsetted UTR reference (from
    filter_mapped_reads). Runs peak calling on each UTR's size distribution,
    enriches var with reference annotations, and returns an AnnData object.

    Args:
        filtered_reads: Filtered flat DataFrame (from filter_mapped_reads).
        utr_reference_subset: Subsetted UTR reference (from filter_mapped_reads).
        protocol: A Protocol instance with peak calling config.
        store: If True, save intermediate files to ./intermediate/.

    Returns:
        An AnnData object with cells/samples as obs and UTR peaks as var.
    """
    peaks_count_matrix, *peaks_var_and_obs = peak_calling.generate_peak_counts_and_anno(
        filtered_reads, protocol=protocol
    )
    peaks_var = peaks_var_and_obs[0]
    peaks_obs = peaks_var_and_obs[1] if len(peaks_var_and_obs) == 2 else None

    peaks_var = peak_calling.add_to_peaks_var(peaks_var, utr_reference_subset)

    if store:
        storage.store_peak_calling_results(
            peaks_count_matrix, peaks_var, peaks_obs,
            utr_reference_subset, protocol.merge_method,
        )

    return peak_calling.generate_count_matrix_anndata(
        count_matrix=peaks_count_matrix, var=peaks_var, obs=peaks_obs
    )
```

### Changes to `tools/run.py`:

```python
# Before:
count_matrix_anndata = core.call_peaks(
    utr_sizes_counts=utr_sizes_counts,
    utr_reference_regions=utr_reference_regions,
    protocol=protocol_,
    store=store_intermediate,
)

# After:
logger.info("Filtering mapped reads")
filtered_reads, utr_reference_subset = core.filter_mapped_reads(
    utr_sizes_counts=utr_sizes_counts,
    utr_reference_regions=utr_reference_regions,
    protocol=protocol_,
)

logger.info("Calling peaks from filtered data")
count_matrix_anndata = core.call_peaks(
    filtered_reads=filtered_reads,
    utr_reference_subset=utr_reference_subset,
    protocol=protocol_,
    store=store_intermediate,
)
```

### Changes to `tools/call_peaks.py`:
Same pattern — insert `core.filter_mapped_reads()` call before `core.call_peaks()`.

### Changes to `test_core.py`:
Tests that call `core.call_peaks(utr_sizes_counts=..., utr_reference_regions=..., protocol=...)` will be updated to:
```python
filtered, ref_subset = core.filter_mapped_reads(
    utr_sizes_counts=cls.mapped,
    utr_reference_regions=cls.utr_regions,
    protocol=cls.protocol_,
)
cls.results = core.call_peaks(
    filtered_reads=filtered,
    utr_reference_subset=ref_subset,
    protocol=cls.protocol_,
)
```
No assertion values change — the outputs are identical, just the call path is split.

### Impact
- `call_peaks()` goes from ~150 lines to ~20 lines — pure peak calling
- New `filter_mapped_reads()` is ~15 lines — pure filtering
- `tools/run.py`: 3 lines become 6 lines (clearer pipeline)
- `tools/call_peaks.py`: same change
- `test_core.py`: ~4 call sites updated (mechanical)

---

## Phase 5: Refactor `map_aligned_reads()` — Separate Single vs Multi-BAM

### Rationale
`map_aligned_reads()` has two distinct code paths: single-BAM (direct call) and multi-BAM (joblib parallelism with batch index resolution). These can be extracted into private helper functions to make the orchestrator cleaner.

### New private functions in `core/core.py`:

```python
def _resolve_batch_idx(alignment, protocol):
    """Resolve the batch index for a single BAM file."""
    if protocol.sample_idx is not None:
        return protocol.sample_idx.get(alignment)
    return None

def _map_single_bam(alignment, intervals, protocol, batch_idx=None):
    """Map reads from a single BAM file to UTR intervals."""
    return utr_mapping.map_reads_to_utr(
        alignment, intervals, read="first", protocol=protocol, batch_idx=batch_idx
    )

def _map_multiple_bams(alignments, intervals, protocol):
    """Map reads from multiple BAM files in parallel via joblib."""
    sample_names = protocol.sample_names
    if sample_names is not None:
        batch_indices = [protocol.sample_idx[a] for a in alignments]
    else:
        batch_indices = [str(i + 1) for i in range(len(alignments))]

    return joblib.Parallel(n_jobs=-2)(
        joblib.delayed(utr_mapping.map_reads_to_utr)(
            alignment, intervals, read="first", protocol=protocol,
            batch_idx=bidx, parallel=False,
        )
        for alignment, bidx in tqdm(
            zip(alignments, batch_indices), colour="#ffb700", ascii=" ■"
        )
    )
```

### Refactored `map_aligned_reads()`:

```python
def map_aligned_reads(alignments, utr_regions, protocol, store=False):
    intervals = utr_mapping.generate_utr_intervals(utr_regions)

    if isinstance(alignments, (list, tuple)) and len(alignments) > 1:
        read_mappings = _map_multiple_bams(alignments, intervals, protocol)
        if store:
            storage.store_read_mappings_multi(read_mappings)
    else:
        if isinstance(alignments, (list, tuple)):
            alignments = alignments[0]
        batch_idx = _resolve_batch_idx(alignments, protocol)
        read_mappings = _map_single_bam(alignments, intervals, protocol, batch_idx)
        if store:
            storage.store_read_mappings_single(read_mappings)

    return read_mappings
```

### Impact
- `map_aligned_reads()` becomes a clean dispatcher (~15 lines)
- No signature changes
- Tests unaffected

---

## Phase 6: Clean Up `process_mapped_reads.py`

### After Phases 2–3
`process_mapped_reads.py` will contain:
1. `merge_samples_utr_sizes()` — sample merging (stays)
2. `merge_utrs_per_gene()` — gene-level aggregation (stays)
3. Re-export: `filter_barcodes` (from `barcode_filtering`)
4. Re-export: `subset_reference_data` (from `utr_reference`)

This is now a focused module for "merging and aggregation" operations on mapped reads. The re-exports provide backward compatibility for tests and any external code.

### Keep filename as `process_mapped_reads.py`
The module could be renamed to better reflect its reduced scope, but this would break import paths in tests and add churn. Keeping the filename avoids this.

---

## Phase 7: Add New Tests

### New test file: `utrcalling/tests/test_barcode_filtering.py`

```python
class TestFilterBarcodesMemoryAware(unittest.TestCase):
    """Test the memory-aware barcode filtering wrapper."""

    def test_small_data_single_pass(self):
        """Verify that small datasets are filtered in a single pass."""

    def test_preserves_categorical_dtypes(self):
        """Verify categorical dtypes are preserved after filtering."""

    def test_exact_match_from_new_module(self):
        """Verify filter_barcodes works when imported from new location."""
```

### New test file: `utrcalling/tests/test_storage.py`

```python
class TestStoreUtrReference(unittest.TestCase):
    """Test UTR reference storage."""

    def test_creates_directory_and_files(self):
        """Verify directory creation and file output."""

class TestStoreReadMappings(unittest.TestCase):
    """Test read mapping storage."""

    def test_single_bam_storage(self):
        """Verify single parquet file creation."""

    def test_multi_bam_storage(self):
        """Verify per-sample parquet file creation."""

class TestStorePeakCallingResults(unittest.TestCase):
    """Test peak calling results storage."""

    def test_stores_sparse_matrix_and_dataframes(self):
        """Verify npz + parquet/csv file creation."""
```

### New test file: `utrcalling/tests/test_filter_mapped_reads.py`

```python
class TestFilterMappedReads(unittest.TestCase):
    """Test the new core.filter_mapped_reads() public function."""

    def test_returns_filtered_reads_and_reference_subset(self):
        """Verify return type is (DataFrame, DataFrame) tuple."""

    def test_utr_reference_subset_matches_read_data(self):
        """Verify the reference subset only contains UTRs present in reads."""

    def test_utr_sizes_below_threshold_removed(self):
        """Verify UTR sizes below threshold are filtered out."""

    def test_with_barcode_whitelist(self):
        """Verify barcode filtering is applied when whitelist is set."""

    def test_without_barcode_whitelist(self):
        """Verify no barcode filtering when whitelist is None."""
```

---

## Final File Layout

```
utrcalling/core/
├── __init__.py
├── core.py                    # Thin orchestrator — 6 public functions, each ≤25 lines
├── barcode_filtering.py       # NEW: filter_barcodes() + filter_barcodes_memory_aware()
├── storage.py                 # NEW: all intermediate file storage utilities
├── peak_calling.py            # Unchanged
├── utr_mapping.py             # Unchanged
├── utr_reference.py           # + subset_reference_data() moved from process_mapped_reads
├── process_mapped_reads.py    # Reduced: merge_samples_utr_sizes, merge_utrs_per_gene + re-exports
├── protocol.py                # Unchanged
├── helpers.py                 # Unchanged
├── module_log.py              # Unchanged (Do Not Touch)
├── anndata_processing.py      # Unchanged
├── peak_calling_anndata.py    # Unchanged
└── protocols/
    └── peak_calling.py        # Unchanged (Do Not Touch)
```

```
utrcalling/tools/
├── run.py                     # Updated: calls filter_mapped_reads() then call_peaks()
├── call_peaks.py              # Updated: calls filter_mapped_reads() then call_peaks()
├── prepare_reference_and_map_reads.py  # Unchanged (doesn't call call_peaks)
└── (other tools unchanged)
```

```
utrcalling/tests/
├── test_core.py               # Updated: call sites split into filter + peaks
├── test_barcode_filtering.py  # NEW: 3 tests
├── test_filter_mapped_reads.py # NEW: 5 tests
├── test_storage.py            # NEW: 5 tests
└── (all other test files — unchanged, re-exports preserve imports)
```

---

## Execution Order

| Phase | Description | Files Modified | Files Created | Tests to Run |
|---|---|---|---|---|
| 0 | Update CLAUDE.md constraints | `CLAUDE.md` | — | — |
| 1 | Extract `storage.py` | `core.py` | `storage.py` | All 218 |
| 2 | Extract `barcode_filtering.py` | `core.py`, `process_mapped_reads.py` | `barcode_filtering.py`, `test_barcode_filtering.py` | All 218 + new |
| 3 | Move `subset_reference_data` | `process_mapped_reads.py`, `utr_reference.py`, `core.py` | — | All |
| 4 | Split `call_peaks()`, add `filter_mapped_reads()` | `core.py`, `tools/run.py`, `tools/call_peaks.py`, `test_core.py` | `test_filter_mapped_reads.py` | All + new |
| 5 | Refactor `map_aligned_reads()` | `core.py` | — | All |
| 6 | Clean up `process_mapped_reads.py` | `process_mapped_reads.py` | — | All |
| 7 | Add remaining new tests | — | `test_storage.py` | All + new |

**Run the full test suite after each phase.** Each phase is independently committable.

---

## Risk Assessment

- **Low risk**: Phases 0, 1, 5, 6, 7 — internal restructuring or additive changes
- **Medium risk**: Phases 2, 3 — moving functions between modules (mitigated by re-exports)
- **Medium risk**: Phase 4 — API change to `call_peaks()` (mitigated by updating all 3 callers + tests in the same phase, and keeping identical behavioral output)

---

## Summary of core.py After Refactoring

```python
# 6 public functions, each a thin orchestrator:

def prepare_utr_reference_regions(gtf, feature, store=False) -> pd.DataFrame:
    """Parse GTF → extract UTRs → merge overlaps → return reference DataFrame."""
    # ~12 lines: sequence of utr_reference.* calls + optional storage

def map_aligned_reads(alignments, utr_regions, protocol, store=False) -> pd.DataFrame | list:
    """Map BAM reads to UTR intervals (single or multi-BAM dispatch)."""
    # ~15 lines: generate intervals → dispatch to _map_single/_map_multiple → optional storage

def merge_samples_utr_sizes(utr_sizes_counts, protocol, store=False) -> pd.DataFrame:
    """Merge multiple sample DataFrames into one."""
    # ~5 lines: delegate to process_mapped_reads → optional storage

def merge_utrs_per_feature(utr_regions, read_mappings, level="UTR") -> tuple:
    """Optionally merge per-UTR data up to gene level."""
    # ~4 lines: conditional delegate to process_mapped_reads

def filter_mapped_reads(utr_sizes_counts, utr_reference_regions, protocol) -> tuple:
    """Subset reference + filter UTR sizes + optionally filter barcodes."""
    # ~10 lines: subset → filter sizes → conditionally filter barcodes

def call_peaks(filtered_reads, utr_reference_subset, protocol, store=False) -> AnnData:
    """Call consensus peaks → enrich var → build AnnData."""
    # ~20 lines: generate peaks → add var annotations → optional storage → build AnnData
```

---

## What This Plan Does NOT Do

- Does not change behavioral test assertions (expected values/counts)
- Does not touch `module_log.py` or `protocols/peak_calling.py`
- Does not introduce new dependencies
- Does not change the intermediate data model
- Does not change parallelism strategies or performance characteristics
