# utrcalling

utrcalling counts the molecules at each site of each UTR, per cell or sample,
from RNA sequencing reads: the polyadenylation sites of 3' UTRs and the
transcription start sites of 5' UTRs. From BAM files and a GTF annotation it
writes AnnData count matrices of cells (or samples) by UTR sites, ready to
study alternative polyadenylation (APA) and alternative transcription
initiation (ATI) at single-cell resolution, with scanpy or any AnnData tool.

```bash
pip install utrcalling
utrcalling run genes.gtf.gz -a sample1.bam sample2.bam -A cells.csv -p 10x -o counts.h5ad
```

- **Single-cell and bulk protocols**: modified and standard 10x Chromium and
  Smart-seq3, 5' 10x and Cap-seq, and stranded or unstranded bulk RNA-seq
  ([](protocols.md)).
- **3' and 5' UTRs**, at UTR or gene level.
- **Peak calling chosen for your data**: a kernel density estimate with a
  choice of kernel and bandwidth, scored by the genome around its sites or by
  a database of known sites ([](peak-calling.md)).
- **Differential site usage**: which UTRs change the use of their sites,
  along trajectories or between conditions, on cells or samples, with effect
  sizes and permutation p-values ([](site-usage.md)).
- **Quality control at every step**, as tables that pandas, R and MultiQC read
  ([](quality-control.md)).
- **Extensible**: add your own scores, peak callers, bandwidth rules and
  protocols ([](extending.md)).

```{toctree}
:caption: Getting started
:maxdepth: 2

installation
quickstart
protocols
outputs
```

```{toctree}
:caption: Guides
:maxdepth: 2

peak-calling
analysis
site-usage
quality-control
workflows
extending
```

```{toctree}
:caption: Reference
:maxdepth: 2

cli
api
changelog
contributing
citation
```
