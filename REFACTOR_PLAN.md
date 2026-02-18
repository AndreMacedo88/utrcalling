# REFACTOR_PLAN.md

# utrcalling — Core Overhaul: Performance, Memory & Modularity Refactor

> **Purpose of this document**: Anchor every architectural decision during the refactor.
> When in doubt between two approaches, choose the one that uses less RAM and runs faster,
> even if it requires writing more code.
> This document takes precedence over stylistic preferences or code brevity.
> This document will be used as a set of consistent instructions when entering plan mode
> to implement the changes.

---

## 1. Goals & Non-Goals

### Goals

1. **Eliminate modin/ray entirely.** The codebase is full of workarounds for modin limitations (forced conversions to pandas, "unfortunately for now we must revert back to pandas" comments, feather roundtrips). Ray adds startup latency and memory overhead. Replace all `modin.pandas` usage with plain `pandas` for metadata or with numpy/scipy for large data.
2. **Fix the count matrix assembly bottleneck.** The current `generate_peak_counts_and_anno()` in `peak_calling.py` creates individual DataFrames per UTR and tries to concatenate tens of thousands of them horizontally — the code is littered with commented-out "patchy patch" attempts that all fail on large data. Replace with direct sparse matrix construction.
3. **Eliminate the lists-in-DataFrame-cells pattern.** The intermediate data stores Python lists of barcodes and UTR sizes inside DataFrame cells, forcing repeated `explode()` → process → `groupby().agg(list)` cycles that are slow and memory-wasteful.
4. **Build a clean, consistent sample/cell identity model** that works uniformly for single-cell (barcoded) and bulk (sample-named) data, replacing the current inconsistent handling of `sample_names`, `barcode_whitelist`, `label_batches`, and `batch`.
5. **Use joblib for parallelism** (already partially in use — standardize it) and remove all ray-based parallelism.
6. **Keep peak RAM usage minimal**, especially during the count matrix assembly step.

### Non-Goals

- Rewriting the CLI layer (`runtime.py`, `utrcalling/tools/`) — leave as-is unless it directly feeds bad data structures into core.
- Replacing AnnData/h5ad as the output format.
- Replacing pybedtools for GTF/BED interval merging — it is correct and not a bottleneck.
- Replacing HTSeq for BAM reading — it is correct and not a bottleneck.
- Replacing pandas for **small metadata DataFrames** (annotations, obs, var) — these are acceptable.
- Replacing scikit-learn's `KernelDensity` or scipy's `argrelextrema` — these are already compiled C/Cython and are not bottlenecks.
- Touching `utrcalling/core/module_log.py` — logging is correct.

---

## 2. Bottleneck Analysis (Priority Order)

This section documents the actual bottlenecks found by reading every core module. The priorities are ordered by impact on real-world performance.

### Priority 1: Count Matrix Assembly (`peak_calling.py:generate_peak_counts_and_anno`)

**The problem**: This is the single biggest bottleneck. The function:
1. Converts the input to modin via a feather temp file roundtrip (lines 33-39).
2. Calls `utr_sizes.apply(call_peaks, axis=1)` which runs peak calling **sequentially per UTR** through modin's `.apply()`, which has massive per-call overhead.
3. The per-UTR `call_peaks()` function (line 323) creates a DataFrame from each row, calls `.explode()`, does groupby, and returns two DataFrames (count_matrix, peaks_var).
4. Tries to horizontally concatenate potentially tens of thousands of per-UTR DataFrames. The code contains **three separate commented-out "patchy patch" areas** where the developer tried chunked concat, disk-based approaches, sparse DataFrames, and manual numpy stacking — evidence that this step fails catastrophically on real-sized data.

**The fix**: Build the final sparse matrix directly from accumulated (cell_index, peak_index, count) arrays. Each UTR's peak calling should return raw arrays, not DataFrames. See Section 3.2.

### Priority 2: Lists-in-Cells Pattern (`utr_mapping.py` + `process_mapped_reads.py`)

**The problem**: `map_reads_to_utr()` stores per-UTR results as a DataFrame where each cell contains a Python list:
```
utr_name | read_barcodes           | utr_sizes_from_reads
UTR_A    | [barcode1, barcode2]    | [153, 148]
UTR_B    | [barcode3]              | [375]
```
This forces every downstream consumer to call `.explode()` to flatten the lists, process them, then `.groupby().agg(list)` to re-nest. This happens in:
- `ProtocolThreePrime.filter_utr_sizes()` — explode, filter, re-group
- `process_mapped_reads.filter_barcodes()` — explode, replace, filter, re-group
- `process_mapped_reads.merge_samples_utr_sizes()` — concat with complex agg lambdas
- `peak_calling.call_peaks()` — explode per UTR row

Each explode/regroup cycle materializes the full flat data, which for 10x data can be millions of reads.

**The fix**: Store the intermediate mapping data in a flat representation — one row per (utr, barcode, utr_size) — or use a dict of per-UTR arrays. See Section 3.1.

### Priority 3: modin/ray Removal

**The problem**: modin is imported in `core.py`, `process_mapped_reads.py`, `peak_calling.py`, and `helpers.py`. Ray is initialized in `helpers.init_ray()` which is called from multiple entry points. The code has numerous `check_if_modin_and_convert_to_pandas()` calls and `isinstance(x, mpd.DataFrame)` checks. Many operations fall back to pandas anyway because modin doesn't support them (groupby.apply with multiple columns, joins not on index).

**The fix**: Replace `import modin.pandas as mpd` with `import pandas as pd` everywhere. Remove `init_ray()`, `check_if_modin_and_convert_to_pandas()`, and all modin isinstance checks. Remove `modin[ray]` and `ray` from `setup.py` dependencies.

### Priority 4: Sample/Cell Identity Model (`protocol.py`)

**The problem**: The current protocol system conflates multiple identity concepts:
- **Cell barcodes** (10x: extracted from read sequence, 16bp) vs **sample names** (SmartSeq: derived from BAM file path)
- **Batch identity** (from annotations file column) vs sample identity
- `protocol.barcode_whitelist` is set for some protocols but not others
- `protocol.sample_names` is set for some protocols but not others
- The `label_batches` flag appends "-N" to barcodes but the logic for when to do this vs not is scattered across `Protocol.__init__`, `ProtocolTenx.__init__`, `ProtocolSmartseq.__init__`
- In `map_aligned_reads()`, the `batch` parameter is passed differently depending on whether there's one or multiple alignments
- `filter_barcodes()` does fuzzy matching (Hamming distance ≤1) which is 10x-specific but applied generically

**The fix**: Introduce a unified "observation ID" concept. Every read gets an observation ID that uniquely identifies the cell/sample. For 10x, this is `{barcode}-{batch}`. For SmartSeq, this is `{sample_name}` or `{barcode}-{sample_name}`. The protocol should define how to construct this ID, and all downstream code should work with observation IDs uniformly. See Section 4.

### Priority 5: `explode()` and `pd.concat()` in Remaining Code

While these operations exist throughout the codebase, many instances are on **small data** (metadata, annotations, var DataFrames) where they are perfectly fine. Only the following are on **large data** and need replacement:

- `process_mapped_reads.filter_barcodes()` — explodes the full mapping data (see Priority 2)
- `ProtocolThreePrime.filter_utr_sizes()` — explodes the full mapping data (see Priority 2)
- `process_mapped_reads.merge_samples_utr_sizes()` with `mpd.concat` — concat of multiple sample mappings
- `peak_calling.generate_peak_counts_and_anno()` — the massive per-UTR concat (see Priority 1)

Small-data uses of concat/explode in `anndata_processing.py`, `peak_calling_anndata.py`, and var/obs construction are acceptable and should be left alone.

---

## 3. Core Data Model Changes

### 3.1 Intermediate Read Mapping Format

**Current**: DataFrame with lists-in-cells (per UTR row).

**New**: A flat structure. Two options, choose based on implementation simplicity:

**Option A — Flat DataFrame** (simpler, still uses pandas but efficiently):
```python
# One row per (utr, barcode, utr_size) observation
mapped_reads = pd.DataFrame({
    "utr_name": np.array([...], dtype="category"),
    "observation_id": np.array([...], dtype="category"),
    "utr_size": np.array([...], dtype=np.int32),
})
```

**Option B — Dict of per-UTR arrays** (lower memory):
```python
# Each UTR maps to its observation IDs and sizes
mapped_reads = {
    "UTR_A": {
        "observation_ids": np.array([...], dtype=np.int32),  # encoded
        "utr_sizes": np.array([...], dtype=np.int32),
    },
    "UTR_B": { ... },
}
```

Option A is recommended because it integrates naturally with pandas groupby for operations that still need it, while avoiding the explode/regroup cycle. Use categorical dtypes for string columns to save memory.

**Rules**:
- Workers (joblib) should return flat DataFrames or dicts-of-arrays, never DataFrames with list cells.
- Main process concatenates worker results once after all workers finish.

### 3.2 Count Matrix Assembly

**Current**: Per-UTR DataFrames concatenated horizontally.

**New**: Build directly as a `scipy.sparse.coo_matrix` (then convert to CSR):

```python
from scipy.sparse import coo_matrix, csr_matrix

# Accumulate across all UTRs
all_rows = []      # cell indices
all_cols = []      # peak indices
all_data = []      # count values
peak_names = []    # for var DataFrame
peak_sizes = []    # for var DataFrame
col_offset = 0

# Build vocabulary for observation IDs
obs_vocab, obs_indices = np.unique(all_observation_ids, return_inverse=True)

for utr_name, utr_data in mapped_reads.groupby("utr_name"):
    # Run KDE peak calling on this UTR's sizes
    peaks = kde_call_peaks(utr_data["utr_size"].values)

    # Count per (observation, peak) combination
    for peak_idx, peak_val in enumerate(unique_peaks):
        mask = peaks == peak_val
        obs_for_peak = utr_data["observation_id"].values[mask]
        obs_encoded, counts = np.unique(obs_for_peak, return_counts=True)

        all_rows.append(obs_encoded)
        all_cols.append(np.full(len(obs_encoded), col_offset + peak_idx, dtype=np.int32))
        all_data.append(counts.astype(np.int32))

        peak_names.append(f"{utr_name}-{peak_val}")
        peak_sizes.append(peak_val)

    col_offset += len(unique_peaks)

# Build sparse matrix in one shot
rows = np.concatenate(all_rows)
cols = np.concatenate(all_cols)
data = np.concatenate(all_data)
X = csr_matrix(coo_matrix((data, (rows, cols)), shape=(len(obs_vocab), col_offset)))

# Build AnnData
adata = anndata.AnnData(X=X, obs=obs_df, var=var_df)
```

This completely eliminates:
- The per-UTR DataFrame creation
- The horizontal concat of thousands of DataFrames
- The fillna/downcast/sparsify cycles
- The modin `.apply()` overhead
- All three "patchy patch" workaround areas

### 3.3 String Encoding for Large Columns

For barcodes/observation IDs that appear millions of times, encode to integers early:
```python
vocab, encoded = np.unique(raw_barcodes, return_inverse=True)
# Carry `encoded` (int32) through computation
# Decode via `vocab[encoded]` only at final output
```

Use `pd.Categorical` when staying in pandas, or `np.unique` with `return_inverse` when in numpy.

---

## 4. Sample/Cell Identity Model

### 4.1 Unified Observation ID

Every protocol must define a method `get_observation_id(alignment, bam_name) -> str` that returns a unique string identifying the cell/sample for that read. This replaces the current split between `get_reads_identity` returning `(umi, barcode)` and the later batch-labeling logic.

| Protocol | Observation ID format | Components |
|----------|----------------------|------------|
| 10x v2/v3 | `{barcode}-{batch_idx}` | barcode from read seq (16bp), batch from annotations |
| SmartSeq3 | `{barcode}-{batch_idx}` | barcode from read seq (6bp), batch from annotations |
| CapSeq SmartSeq3 | `{sample_name}` | sample from BAM path |
| 5' 10x | `{CB_tag}-{batch_idx}` | CB tag from BAM, batch from annotations |
| Bulk (future) | `{sample_name}` | sample from BAM path |

### 4.2 Protocol Cleanup

- Move batch-labeling logic (the "-N" suffix) into `get_observation_id()` so it happens at read time, not as a post-processing step in `merge_samples_utr_sizes`.
- Remove `_resolve_barcode_reuse_by_batch()` from `Protocol.__init__` — this is a runtime concern, not an init concern.
- Separate barcode whitelist filtering from fuzzy matching. Fuzzy matching (Hamming distance correction) should be an explicit opt-in step, not hardcoded into `filter_barcodes()`.
- The `sample_names` bidict in SmartSeq protocols is only used to pass the sample name as `batch` to `map_reads_to_utr`. This should be simplified.

---

## 5. Parallelism Rules (joblib)

joblib is already used in `map_aligned_reads()`. Standardize all parallel code to this pattern:

```python
from joblib import Parallel, delayed

results = Parallel(n_jobs=-2, backend="loky")(
    delayed(worker_fn)(sample, *args) for sample in samples
)
```

**Rules:**
- **Always use `backend="loky"`** (process-based). Never use `threading` backend for CPU-bound work.
- **Workers must return plain data** — DataFrames (flat, not lists-in-cells), dicts of numpy arrays, or tuples of arrays. Not modin DataFrames, not AnnData objects, not objects with file handles.
- **Never share mutable state between workers.**
- **Main process concatenates all worker results once** after `Parallel` returns.
- **Preserve the existing logging guard** in `module_log.py` that prevents duplicate logger setup from worker processes.
- **For the peak calling step**: Consider parallelizing across UTRs using joblib instead of modin `.apply()`. Each worker processes a batch of UTRs and returns accumulated (row, col, data) arrays for sparse matrix construction.

---

## 6. Memory Management Rules

- **Delete large intermediates explicitly** with `del` after they are no longer needed (the codebase already does this in places — keep this practice).
- **Avoid boolean fancy indexing on large arrays** when a slice or view suffices.
- **Never accumulate results by concatenating inside a loop** — collect in a list, concatenate once at the end.
- **Use categorical dtypes** for string columns in pandas DataFrames to reduce memory.
- **Use `scipy.sparse`** for the count matrix from the moment of construction — never materialize it as dense.
- **Do not call `gc.collect()`** unless profiling proves it helps at a specific location.

---

## 7. File-by-File Refactor Scope

### Priority 1 — Remove modin/ray (do first, unblocks everything else)

- `utrcalling/core/helpers.py` — Remove `init_ray()`, `check_if_modin_and_convert_to_pandas()`. Change `mpd.read_csv` / `mpd.read_parquet` to `pd.read_csv` / `pd.read_parquet`. Remove `import ray`, `import modin.pandas`.
- `utrcalling/core/core.py` — Change `import modin.pandas as mpd` to `import pandas as pd`. Remove `helpers.init_ray()` calls. Remove `mpd.concat` usage.
- `utrcalling/core/process_mapped_reads.py` — Change `import modin.pandas as mpd` to `import pandas as pd`. Remove `from modin.utils import to_pandas`. Remove modin isinstance checks.
- `utrcalling/core/peak_calling.py` — Same modin removal. Remove the feather roundtrip in `generate_peak_counts_and_anno()`.
- `setup.py` / `requirements.txt` / `environment.yml` — Remove `modin[ray]` and `ray` from dependencies.

### Priority 2 — Refactor the intermediate data model

- `utrcalling/core/utr_mapping.py` — Change `map_reads_to_utr()` to return a flat DataFrame (one row per read/molecule) with columns `[utr_name, observation_id, utr_size]` instead of a DataFrame with list cells. Update `deduplicate_reads()` accordingly.
- `utrcalling/core/process_mapped_reads.py` — Rewrite `merge_samples_utr_sizes()` to work with the flat format (simple `pd.concat` of flat DataFrames). Rewrite `filter_barcodes()` to filter rows directly instead of explode/regroup. Rewrite `merge_utrs_per_gene()`.
- `utrcalling/core/protocol.py` — Rewrite `filter_utr_sizes()` in `ProtocolThreePrime` to work with flat data (simple boolean mask instead of explode/regroup).

### Priority 3 — Rewrite count matrix assembly

- `utrcalling/core/peak_calling.py` — Replace `generate_peak_counts_and_anno()` entirely. The new version should:
  1. Group the flat read mappings by UTR name.
  2. For each UTR, run KDE peak calling on UTR sizes to get peak assignments.
  3. Accumulate (row, col, data) arrays for direct sparse matrix construction.
  4. Build `scipy.sparse.csr_matrix` once.
  5. Build var and obs DataFrames from accumulated metadata.
  6. Return the sparse matrix, var, and obs.
- Remove the per-row `call_peaks()` function (line 323-356) — its logic folds into the new assembly loop.
- Remove all "patchy patch" code blocks.

### Priority 4 — Clean up Protocol identity model

- `utrcalling/core/protocol.py` — Add `get_observation_id()` method. Consolidate batch-labeling. Simplify `sample_names` handling.
- `utrcalling/core/utr_mapping.py` — Use `protocol.get_observation_id()` in `extract_read_information()`.

### Priority 5 — Update tests

- **Behavioral tests** (testing correctness of output values) — preserve the expected values.
- **Implementation tests** (testing DataFrame shapes/structures) — update to match new intermediate formats.
- `utrcalling/tests/test_peak_calling.py` — Update `TestMatrix`, `TestVar`, `TestObs` to expect sparse matrix output. Value-based assertions should remain the same.
- `utrcalling/tests/test_process_mapped_reads.py` — Update to work with flat DataFrames instead of list-cell DataFrames. The expected barcodes and UTR sizes should remain the same.
- Synthetic test data generators in `utrcalling/tests/test_data/` — update to produce flat-format data.

### Do Not Touch

- `utrcalling/runtime.py` and `utrcalling/tools/` — CLI layer (unless needed to pass new arguments).
- `utrcalling/core/module_log.py` — logging is correct.
- `utrcalling/core/utr_reference.py` — GTF parsing is correct and not a bottleneck. It uses pandas appropriately for small reference data.
- `utrcalling/core/anndata_processing.py` — Post-processing utilities that operate on already-built AnnData. These use pandas/numpy appropriately on small-to-medium data.
- `utrcalling/core/peak_calling_anndata.py` — Secondary peak calling path that operates on pre-built AnnData. Can be updated later if needed, but is not the primary bottleneck.
- `utrcalling/core/protocols/peak_calling.py` — KDE logic is correct and efficient (sklearn + scipy). The inner peak assignment loop is small per-UTR and not worth optimizing further.

---

## 8. Refactor Sequence

Execute in this order. Do not move to the next step until the current step's tests pass.

1. **Remove modin/ray** (Priority 1 above). Replace all imports, remove init_ray, remove conversion helpers. Run tests — they should pass with pandas in place of modin, possibly with minor fixes.
2. **Refactor the intermediate data model** (Priority 2). Change `map_reads_to_utr()` output format. Update all consumers. Run tests — update expected structures where needed, verify expected values are preserved.
3. **Rewrite count matrix assembly** (Priority 3). Replace `generate_peak_counts_and_anno()` with sparse-matrix-based approach. Run tests — peak calling output values should match.
4. **Clean up Protocol identity model** (Priority 4). Add `get_observation_id()`, simplify batch handling. Run tests.
5. **Final integration test**: measure peak RAM (use `tracemalloc` or `/usr/bin/time -v`) and wall time before/after on the test data.

---

## 9. What Claude Must Never Do During This Refactor

- **Never use `modin` or `ray`** — they are being removed entirely.
- **Never store Python lists inside DataFrame cells** — use flat rows or separate arrays.
- **Never create per-row/per-UTR DataFrames inside a loop and concatenate them** — accumulate arrays and build the result once.
- **Never concatenate DataFrames inside a loop** — collect in a list, concatenate once.
- **Never use `.apply()` with `axis=1` on large DataFrames** — use vectorized operations or explicit loops with array accumulation.
- **Never materialize the full count matrix as a dense array** — use `scipy.sparse` from the start.
- **Never change behavioral test assertions** (expected values, expected counts) without explicit justification.

---

## 10. On Numba

The original plan included extensive numba usage. After reading the actual code, **numba is not recommended for this refactor** for the following reasons:

1. The KDE scoring uses sklearn's `KernelDensity` which is already compiled C. Wrapping it in numba would not help.
2. The `argrelextrema` call is scipy C code.
3. The inner peak assignment loop in `protocols/peak_calling.py` (lines 68-86) iterates over `n_minima × n_reads_per_utr`. For typical UTRs this is small (tens of minima × hundreds of reads). The overhead is dominated by DataFrame creation around it, not the numeric loop.
4. Adding numba as a dependency increases install complexity, adds JIT compilation latency, and limits supported Python versions and platforms.

If profiling after the main refactor reveals specific numeric inner loops that are bottlenecks, numba can be considered at that point as a targeted optimization. But it should not be part of the initial refactor.

---

## 11. dtype Guidance

Use explicit dtypes for large arrays:

| Data | dtype | Rationale |
|------|-------|-----------|
| UTR sizes (bp) | `np.int32` | Signed for potential sentinel values |
| Read counts | `np.int32` or `np.uint32` | Non-negative counts |
| Genomic positions | `np.int32` | Chromosomes < 2^31 bp |
| Sparse matrix values | `np.int32` | Count data in CSR/COO matrix |
| Boolean masks | `np.bool_` | Standard numpy boolean |
| Barcodes/observation IDs | `pd.Categorical` or encoded `np.int32` | Save memory on repeated strings |

For small arrays (per-UTR KDE evaluation, metadata), default numpy dtypes are fine — the memory difference is negligible.
