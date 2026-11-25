# utrcalling

[![CI](https://github.com/AndreMacedo88/utrcalling/actions/workflows/tests.yml/badge.svg)](https://github.com/AndreMacedo88/utrcalling/actions/workflows/tests.yml)
[![Coverage](https://codecov.io/gh/AndreMacedo88/utrcalling/graph/badge.svg)](https://codecov.io/gh/AndreMacedo88/utrcalling)
[![Documentation](https://readthedocs.org/projects/utrcalling/badge/?version=latest)](https://utrcalling.readthedocs.io)
![Python >= 3.11](https://img.shields.io/badge/python-%3E%3D3.11-blue)
![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-green)
[![PyPI](https://img.shields.io/pypi/v/utrcalling)](https://pypi.org/project/utrcalling/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23686176.svg)](https://doi.org/10.5281/zenodo.23686176)

utrcalling counts the molecules at each site of each UTR, per cell or sample,
from RNA sequencing reads: the polyadenylation sites of 3' UTRs and the
transcription start sites of 5' UTRs. From BAM files and a GTF annotation it
writes AnnData (`.h5ad`) count matrices of cells (or samples) by UTR sites,
ready to study alternative polyadenylation (APA) and alternative
transcription initiation (ATI) at single-cell resolution.

**Documentation: [utrcalling.readthedocs.io](https://utrcalling.readthedocs.io)**

## Features

- **Single-cell and bulk protocols** --- modified and standard 10x Chromium and
  Smart-seq3, 5' 10x and Cap-seq, and stranded or unstranded bulk RNA-seq.
- **3' and 5' UTRs**, at UTR or gene level.
- **Peak calling chosen for your data** --- a kernel density estimate with a
  choice of kernel and bandwidth (fixed, set per UTR, or chosen by scoring its
  sites against the genome or a database of known sites), and the stability
  of each peak.
- **Quality control at every step** --- tables of matrices, cells, batches and
  sites that pandas, R and MultiQC read, and a record in each matrix of how it
  was made.
- **From sites to analysis** --- filter sites and UTRs, keep the UTRs with
  several sites, site usage, mean UTR sizes and length indices per cell or
  sample, pseudobulk units, and the sequences around sites for motif analyses.
- **Differential site usage** --- which UTRs change the use of their sites,
  along a trajectory or between conditions, with R-style model formulas: a
  Dirichlet-multinomial test on cells or samples, permutations that respect
  the replicates, DRIMSeq as an optional engine, and effect sizes.
- **Extensible** --- add your own scores, peak callers, bandwidth rules and
  protocols, as plugins or in a notebook.
- **Nothing outside Python** --- `pip install` is all it needs (R only for
  the optional DRIMSeq engine).

## Installation

utrcalling needs Python 3.11 or later, on Linux or macOS. In a conda
environment:

```bash
conda create -n utrcalling -c conda-forge python=3.12
conda activate utrcalling
pip install utrcalling
```

Other ways, and a frozen environment for reproducibility:
[Installation](https://utrcalling.readthedocs.io/en/latest/installation.html).

## Quick start

```bash
# Count the molecules at each UTR site of each cell
utrcalling run Mus_musculus.GRCm39.110.gtf.gz \
    --alignments sample1.bam sample2.bam --annotations-file cells.csv \
    --protocol-platform 10x --protocol-version 3 --raw-sizes --out raw_sizes.h5ad

# How the run went, per matrix
utrcalling qc report raw_sizes.h5ad raw_sizes_kde.h5ad -o report.tsv

# Keep the frequent sites, and the UTRs with two or more of them
utrcalling process filter-sites raw_sizes_kde.h5ad sites.h5ad --threshold 0.05 --minimum-counts 10
utrcalling process keep-multi-site sites.h5ad apa.h5ad

# Each site's share of its UTR's molecules, per cell
utrcalling tools to-fractions apa.h5ad fractions.h5ad
```

`utrcalling --help` lists the commands, and `utrcalling COMMAND --help`
describes one. The documentation explains each step:

- [Quickstart](https://utrcalling.readthedocs.io/en/latest/quickstart.html)
- [Protocols and input files](https://utrcalling.readthedocs.io/en/latest/protocols.html):
  what your annotation table needs
- [Output files](https://utrcalling.readthedocs.io/en/latest/outputs.html)
- [Peak calling](https://utrcalling.readthedocs.io/en/latest/peak-calling.html):
  what the bandwidth means, choosing it for your data, and 5' data
- [From sites to analysis](https://utrcalling.readthedocs.io/en/latest/analysis.html)
  and [quality control](https://utrcalling.readthedocs.io/en/latest/quality-control.html)
- [Differential site usage](https://utrcalling.readthedocs.io/en/latest/site-usage.html):
  which test answers which question, and how to prepare the data
- [In workflows](https://utrcalling.readthedocs.io/en/latest/workflows.html),
  and the command lines of earlier versions, which still work
- [Extending utrcalling](https://utrcalling.readthedocs.io/en/latest/extending.html)
- [Command line](https://utrcalling.readthedocs.io/en/latest/cli.html) and
  [Python](https://utrcalling.readthedocs.io/en/latest/api.html) references

## Citation

If you use utrcalling in your research, please cite it. Releases are archived
on Zenodo under the DOI
[10.5281/zenodo.23686176](https://doi.org/10.5281/zenodo.23686176), which
always resolves to the latest version; each version also has its own DOI,
listed on that page. Citation metadata is in `CITATION.cff`, and GitHub's
"Cite this repository" button formats it as APA or BibTeX.

## Acknowledgements

Peak calling for 5' data builds on the master's thesis of Eugenia Guerrero
Aruffo: *Interferon regulates transcription and translation in neural stem
cells by targeting regulatory elements on the 5' UTR* (Heidelberg University,
carried out at the German Cancer Research Center, 2026). She chose the linear
kernel, half of Scott's bandwidth per UTR, the filters per treatment and by
molecules; found and fixed peak calling on flat densities; and proposed the
initiator as a measure of how well peaks find start sites, and bootstrap
resampling as a measure of how stable peaks are.

## Contributing

Bug reports, data that utrcalling does not handle well, and code are welcome:
see [CONTRIBUTING.md](https://github.com/AndreMacedo88/utrcalling/blob/main/CONTRIBUTING.md).

## License

BSD-3-Clause. See [LICENSE](https://github.com/AndreMacedo88/utrcalling/blob/main/LICENSE).
Author: André Lopes Martins Macedo.
