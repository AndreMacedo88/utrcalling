# Quickstart

From aligned reads to the UTR sites that each cell uses, in four commands.
You need the BAM files of your cells or samples (sorted and indexed), the GTF
annotation of the genome they were aligned to, and a table of the cells or
samples (see [](protocols.md)).

## The commands

| Command | Purpose |
| --- | --- |
| `run` | The whole pipeline, from BAM files to a count matrix of UTR sites |
| `map-reads`, then `count` | The same, in two steps |
| `call-peaks` | The peaks of a raw-sizes count matrix (from `run --raw-sizes`), with one or more settings |
| `process ...` | Prepare the count matrix: filter sites and UTRs, keep multi-site UTRs, number sites again, join experiments |
| `tools ...` | Compute the measures to analyse: site usage fractions, mean UTR sizes, the sequences around sites |
| `qc ...` | Quality control tables, at any step |

`utrcalling --help` lists them, `utrcalling process --help` lists a group's
commands, and `utrcalling run --help` describes a command's options.

## 1. Count the molecules at each site

```bash
utrcalling run Mus_musculus.GRCm39.110.gtf.gz \
    --alignments sample1.bam sample2.bam \
    --annotations-file cells.csv \
    --protocol-platform 10x --protocol-version 3 \
    --raw-sizes \
    --out raw_sizes.h5ad
```

`run` maps the reads to the 3' UTRs of the annotation, measures each
molecule's UTR size, and collapses each UTR's sizes into sites. It writes the
raw sizes (`raw_sizes.h5ad`), the sites (`raw_sizes_kde.h5ad`), what happened
to the reads (`raw_sizes_mapping_summary.tsv`), and a log. For 5' UTRs, add
`--feature five_prime_utr` and read [](peak-calling.md).

## 2. Check the result

```bash
utrcalling qc report raw_sizes.h5ad raw_sizes_kde.h5ad -o report.tsv
utrcalling qc cells raw_sizes_kde.h5ad cells.tsv
```

The report gives, per matrix, its cells, sites, UTRs and molecules, and the
mapping summary: where the reads that did not count were lost.

## 3. Keep the sites to study

```bash
# Sites with at least 5% of their UTR's molecules, and 10 molecules
utrcalling process filter-sites raw_sizes_kde.h5ad sites.h5ad --threshold 0.05 --minimum-counts 10

# UTRs with two or more sites: alternative polyadenylation
utrcalling process keep-multi-site sites.h5ad apa.h5ad
```

## 4. Compute what to analyse

```bash
# Each site's share of its UTR's molecules, per cell
utrcalling tools to-fractions apa.h5ad fractions.h5ad
```

Then analyse `fractions.h5ad` like any AnnData, for example with scanpy:

```python
import anndata as ad

fractions = ad.read_h5ad("fractions.h5ad")
fractions.obs.groupby("genotype").size()
```

## Next

- [](protocols.md): the protocols, and what your annotation table needs.
- [](peak-calling.md): how sites are called, and how to choose the settings for
  your data.
- [](analysis.md) and [](quality-control.md): every `process`, `tools` and `qc`
  command.
- [](cli.rst): all the options of every command.
