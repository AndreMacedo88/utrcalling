# Quality control

The `qc` commands describe count matrices from any step, without changing
them. Each writes a tab-separated table, which pandas, R and MultiQC read:

| Command | One row per | What it tells |
| --- | --- | --- |
| `qc report` | count matrix | What each matrix holds and which steps made it; its cells, sites, UTRs and molecules, per-cell medians, multi-site UTRs, depth correlation and mapping summary |
| `qc cells` | cell (or sample) | Molecules, sites, UTRs measured, multi-site UTRs, and its relative UTR size |
| `qc batches` | group of cells, and pair of groups | Sites only one group has; how alike the groups' site usage is |
| `qc known-sites` | site | Distance to the nearest known site of a database |
| `qc pas-distance` | site | Distance to the nearest polyadenylation signal hexamer |
| `qc priming` | site | Whether the genome around it could explain it as a priming artefact: A's after a 3' site (internal priming), or a sequence like the template-switching oligo before a 5' site (strand invasion) |
| `qc peaks` | peak | Molecules, share of its UTR, the spread of the raw sizes it collects, and with `--bootstrap`, how often it comes back when the UTR's molecules are drawn again |
| `qc score-bandwidths` | KDE kernel and bandwidth | How well its peaks fall where true sites are: 10-30 nt after a PAS hexamer (3' data), on a pyrimidine-purine initiator (5' data), or near the known sites of a database; and how many it makes |

```bash
# A row per matrix, from mapping to analysis; MultiQC shows *_mqc.tsv files
utrcalling qc report raw_sizes.h5ad peaks_bw12.h5ad filtered.h5ad \
    -o qc_report_mqc.tsv --multiqc

# Per cell. Its log gives the correlation of the cells' relative UTR size
# with their molecules: a strong one means UTR sizes follow sequencing depth.
utrcalling qc cells peaks_bw12.h5ad cells.tsv

# Per batch (or sample, replicate: a column of the cell annotations).
# Writes batches_groups.tsv and batches_pairs.tsv.
utrcalling qc batches peaks_bw12.h5ad batches --groupby Batch

# Distance to known polyadenylation sites (e.g. PolyASite 2.0 clusters) or,
# for 5' data, known transcription start sites (e.g. refTSS)
utrcalling qc known-sites peaks_bw12.h5ad atlas.clusters.2.0.GRCh38.96.bed.gz known_sites.tsv

# Distance of each site to the nearest PAS hexamer
utrcalling qc pas-distance peaks_bw12.h5ad pas_distances.tsv genome.fa

# Sites that the genome could explain as priming artefacts: internal priming
# for 3' data; strand invasion for 5' data, with the template-switching oligo
# of the recorded protocol (or --tso)
utrcalling qc priming peaks_bw12.h5ad genome.fa priming.tsv

# How wide the peaks of a bandwidth are, and how stable: the share of 100
# draws of each UTR's molecules in which the peak comes back
utrcalling qc peaks raw_sizes.h5ad peaks_bw12.tsv --bandwidth 12 --bootstrap 100

# How well bandwidths (and kernels) score
utrcalling qc score-bandwidths raw_sizes.h5ad bandwidth_scores.tsv genome.fa \
    --bandwidths 4 8 12 16
```

Peak stability compares the peaks of one setting: peaks of few molecules, or
on the flank of a larger one, come back less often. It does not compare
settings, since wider bandwidths make every peak more stable.

The site tables answer different questions. `known-sites` asks whether a
database has seen the site. `priming` asks whether the genome around it could
explain it as an artefact, for known and new sites alike: internal priming
happens inside the UTR, and its sites look like a shorter UTR. `pas-distance`
asks whether a polyadenylation signal lies where it should. For 3' data, read
together:

| Known site | Priming flag | PAS hexamer 10-30 nt before | Reading |
| --- | --- | --- | --- |
| yes | no | | Solid |
| no | no | yes | Candidate new site |
| no | yes | | Likely internal priming |
| yes | yes | | Check by hand |

```python
import pandas as pd


def read(path):
    return pd.read_csv(path, sep="\t", index_col="site")


sites = read("known_sites.tsv").join(read("priming.tsv")[["internal_priming"]])
sites = sites.join(read("pas_distances.tsv")[["pas_distance"]])
new_sites = sites.query(
    "not near_known_site and internal_priming == False and 10 <= pas_distance <= 30"
)
```

For 5' data, the known sites are transcription start sites and the flag is
strand invasion; the PAS hexamer does not apply. The priming flags need sites
at the molecules' ends: with the standard and bulk protocols, whose reads stop
short of the ends, they say little.


## Why so few molecules?

The mapping summary of each run (see [](outputs.md)) counts, per BAM file,
the alignments each step skipped: on the wrong strand, without a barcode,
outside the UTR, duplicates, and so on. `qc report` shows it per matrix.
