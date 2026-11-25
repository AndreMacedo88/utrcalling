# Output files

## The count matrix

`run` and `count` write a count matrix as an AnnData file (`.h5ad`): cells or
samples are its observations (`.obs`, with the columns of your annotation
table) and UTR sites its variables (`.var`). `.X` is sparse and holds each
cell's molecules at each site. scanpy, anndata and R (with `anndata` or
`zellkonverter`) read it.

```python
import anndata as ad

sites = ad.read_h5ad("counts.h5ad")
sites.var[["gene_name", "utr_name", "utr_size"]].head()
```

With `--raw-sizes` (`-r`), `run` writes two files: the raw UTR sizes, a
column per size that molecules have, to `--out`; and their peaks (sites) to
`<out>_kde.h5ad`. `call-peaks` calls the peaks of raw sizes again, with other
settings (see [](peak-calling.md)).

### Sites

Each site is named `<gene>_<n>-<size>`, such as `TSPAN6_2-816`: the gene's name
(its Ensembl ID if it has none), the site's number within the gene (1 is the
shortest site of the gene's first UTR) and its UTR size in nt. A site keeps
its name through the filtering commands; `process rename-sites` numbers the
remaining ones again. To find sites in code, use the columns of `.var`
rather than the names:

| Column | Holds |
| --- | --- |
| `utr_name` | The UTR: the gene ID, with `_2`, `_3`... for the gene's other UTRs |
| `utr_size` | The site's UTR size in nt: from the UTR's start for 3' data, from its end for 5' data |
| `gene_name`, `name` | The gene's name and ID |
| `chrom`, `start`, `end`, `strand` | The UTR in the genome (1-based, inclusive) |
| `transcript_id` | The transcripts whose UTRs make it |
| `bandwidth` | The UTR's bandwidth, when a rule (`-b scott`) set it |

A UTR is a stretch of the genome where the annotation's UTRs of one strand
overlap. A UTR spliced into several exons is several UTRs here (`gene`,
`gene_2`, ...).

### How each matrix was made

Every count matrix records in `.uns["utrcalling"]` how it was made: the
utrcalling version, the commands that made it (as typed, the first one making
the matrix), what it holds (`matrix`: `raw_sizes`, `peaks`, `fractions`,
`mean_sizes` or `length_index`), and the settings needed to read it: the
protocol, the `summarization_level`, `utr_end` (where its sizes count from:
`three_prime` or `five_prime`), the peak calling (`peak_caller`, `kernel`,
`bandwidth`, and the `bandwidth_scores` of `-b optimize`), how cells were
summed into units (`pseudobulk_by`, `pseudobulk_bins`) and the mapping
summary. The `qc` commands read it.

```python
sites.uns["utrcalling"]["commands"]
```

## The mapping summary

`run` and `count` write `<out>_mapping_summary.tsv`, with a row per BAM file
and a `total` row (`map-reads` writes the BAM rows to
`intermediate/mapping_summary.tsv`). It answers "why so few molecules?". Each
matrix keeps the summary of the run that made it, also after filtering, and
`process concatenate` joins those of the experiments. Each column counts the
alignments in the UTR regions that one step skipped, or the reads and
molecules it kept:

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

## Other files

- **Logs**: `run`, `map-reads` and `count` write a log next to `--out`, with
  every message down to debug (`--no-log-file` to skip it). The terminal shows
  messages from `--log-level` up.
- **Intermediate files** (`--store-intermediate`, `-i`): the UTR reference and
  the read mappings, as Parquet files in `./intermediate/`, from which `count`
  can redo the last steps quickly (see [](workflows.md)).
- **Bandwidth scores**: with `-b optimize`, `<out>_bandwidth_scores.tsv`.
- **Differential site usage**: `stats site-usage` writes a table of UTRs,
  `<out>_sites.tsv` and `<out>_settings.json` (see [](site-usage.md)).
