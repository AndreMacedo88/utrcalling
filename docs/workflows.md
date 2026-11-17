# In workflows

## In two steps

`run` is `map-reads` followed by `count`. Mapping reads takes most of the
time; counting is quick. Run them apart to count again with other settings
(another `--summarization-level`, or other peak calling) without mapping
again:

```bash
# Build the UTR reference and map the reads; both go to ./intermediate/
utrcalling map-reads Mus_musculus.GRCm39.110.gtf.gz \
    --alignments sample1.bam sample2.bam \
    --annotations-file cells.csv \
    --protocol-platform 10x --protocol-version 3

# Count the sites from them (the same annotation table and --label-batches)
utrcalling count intermediate/utr_reference_regions.parquet \
    --utr-sizes-counts intermediate/UTR_sizes_counts_*.parquet \
    --annotations-file cells.csv \
    --protocol-platform 10x --protocol-version 3 \
    --out counts.h5ad
```

`run --store-intermediate` (`-i`) keeps the same files.

## With Snakemake or Nextflow

Each command is one step that reads files and writes files, so it fits a
workflow rule as it is. Give it the rule's threads with `--cores` (by default
utrcalling uses all cores but one), and its log with `--out`: the log file
goes next to it.

```python
rule utrcalling:
    input:
        gtf="annotation/genes.gtf.gz",
        bams=expand("aligned/{sample}.bam", sample=SAMPLES),
        cells="cells.csv",
    output:
        raw="utrcalling/raw_sizes.h5ad",
        sites="utrcalling/raw_sizes_kde.h5ad",
    threads: 16
    shell:
        "utrcalling run {input.gtf} -a {input.bams} -A {input.cells} "
        "-p 10x -v 3 -r -o {output.raw} --cores {threads}"
```

Outside a terminal, utrcalling writes plain log lines, which the workflow's
log captures as they are.

## Command lines from earlier versions

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

The [changelog](changelog.md) lists what changed in each version.
