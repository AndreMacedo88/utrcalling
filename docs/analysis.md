# From sites to analysis

After the pipeline, the `process` commands prepare the count matrix and the
`tools` commands compute the measures to analyse from it, like scanpy's `pp`
and `tl`. Each reads a count matrix and writes another, recording the step
in it (see [](outputs.md)).

## Preparing the count matrix: `process`

```bash
# Call the peaks of the raw sizes (writes peaks_bw12.h5ad)
utrcalling call-peaks raw_sizes.h5ad peaks --bandwidths 12

# Remove low-count sites: keep those whose molecules, in all cells together,
# reach 5% of their UTR's and 10
utrcalling process filter-sites peaks_bw12.h5ad filtered.h5ad \
    --threshold 0.05 --minimum-counts 10

# ... or count per group of cells (a column of the annotations), keeping the
# sites that pass in at least one group, such as those a treatment raises
utrcalling process filter-sites peaks_bw12.h5ad filtered.h5ad \
    --threshold 0.1 --minimum-counts 10 --groupby Treatment

# Remove the UTRs with few molecules over all cells, whose sites are unsure
utrcalling process filter-utrs filtered.h5ad deep.h5ad --minimum-molecules 500

# Keep only the UTRs with two or more sites: alternative polyadenylation
# (3' data) or alternative transcription initiation (5' data)
utrcalling process keep-multi-site filtered.h5ad apa.h5ad

# Optional: number each gene's remaining sites 1, 2, ... again. The sites
# then have other names than in the files before: compare files by the
# utr_name and utr_size columns.
utrcalling process rename-sites apa.h5ad named.h5ad

# Join the cells of several experiments
utrcalling process concatenate -i experiment1.h5ad experiment2.h5ad -o combined.h5ad
```

## Computing what to analyse: `tools`

```bash
# Each site's share of its UTR's molecules, per cell
utrcalling tools to-fractions named.h5ad fractions.h5ad

# Each cell's mean UTR size, per UTR
utrcalling tools mean-sizes named.h5ad mean_sizes.h5ad
```

### The sequences around sites

`tools site-sequences` writes the genome around each site, as FASTA for motif
tools such as the MEME suite or HOMER (for a `.fa`, `.fasta` or `.fna`
output), or as a table. Positions count from the transcript's end: for 5'
data +1 is the start site, for 3' data -1 is the cleavage site; `--upstream`
reads before it and `--downstream` after it.

```bash
# 50 nt on each side
utrcalling tools site-sequences named.h5ad genome.fa sites.fa --upstream 50 --downstream 50

# The UTR each site makes, to the end of its UTR (utr instead of a number):
# for 5' sites from the start site to the start codon, for 3' sites from the
# stop codon to the cleavage site
utrcalling tools site-sequences named.h5ad genome.fa utrs.fa --upstream 0 --downstream utr
utrcalling tools site-sequences named.h5ad genome.fa utrs.fa --upstream utr --downstream 0
```

Each command's options: [](cli.rst).
