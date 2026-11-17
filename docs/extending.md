# Extending utrcalling

utrcalling's methods that data most often need an alternative to are
*extension points*. You can add your own method to any of them without
changing utrcalling: it then works by name in every command that takes one,
like utrcalling's own.

| Extension point | What it does | Chosen with | utrcalling's own |
| --- | --- | --- | --- |
| [Scores](#a-score) | Judge how well the sites of a peak-calling setting fall where true sites are, so `qc score-bandwidths` can compare settings and `-b optimize` choose one | `--score` | `pas`, `initiator`, `known-sites` |
| [Peak callers](#a-peak-caller) | Collapse each UTR's molecule sizes into sites | `--peak-caller` | `kde` |
| [Bandwidth rules](#a-bandwidth-rule) | Set each UTR's bandwidth from its molecules' sizes | `-b NAME` or `-b NAME:FACTOR` | `scott`, `silverman` |
| [Protocols](#a-protocol) | Read a sequencing protocol's BAM files: which reads count, where molecules end, which cell they come from | `--protocol-platform` | `10x`, `smartseq`, `5prime10x`, `truseq`, ... |

For example, when 5' data come from degraded RNA, start sites move and the
`initiator` score no longer finds the best setting: a score that rewards what
such data still have can take its place, as `--score my-score`.

## How utrcalling finds methods

Each extension point has a *registry*, which maps names to methods. Names are
case-insensitive, and utrcalling's own names cannot be taken over. A method
comes from one of three places:

1. **utrcalling itself.** Its modules hold one method each (see
   [Where the code is](#where-the-code-is)).
2. **A plugin**: another installed package, which declares the method as an
   entry point (see [Sharing a method as a plugin](#sharing-a-method-as-a-plugin)).
   The command line finds it by its name.
3. **Your own code**, running in the same Python process as utrcalling, such
   as a notebook: register the method, then call utrcalling's Python
   functions with its name.

```python
from utrcalling.core.peak_calling.settings import BANDWIDTH_RULES, PEAK_CALLERS, SCORES
from utrcalling.core.protocol import PROTOCOLS

print(SCORES.names())  # ('pas', 'initiator', 'known-sites', ...and the plugins')
```

A registry's `register(name, method)` adds a method (it also works as a
decorator, `@SCORES.register(name)`), and `unregister(name)` removes it.

## A score

A score is a subclass of `SiteScore`. utrcalling creates it once per raw-sizes
matrix, then asks it for each setting's UTR scores: a pandas Series with a
score per UTR, higher for better sites, where NaN leaves a UTR out. A
setting's score is the mean over UTRs.

As an example, a score for 3' sites by their downstream element: cleavage
sites have a U-rich stretch in the 30 nt after them, which a setting that
moves the sites off them loses.

```python
import numpy as np
import pandas as pd

from utrcalling.core import genome
from utrcalling.core.peak_calling.scores import (
    SiteScore,
    share_at_frequent_sites,
    site_molecules,
)
from utrcalling.core.peak_calling.settings import SCORES


class DownstreamUScore(SiteScore):
    """Per UTR, the share of molecules at sites with 6 or more T's in the 30 nt after them."""

    utr_ends = ("three_prime",)  # the sites it can score
    needs = ("genome",)  # checked before any work: --genome must be given

    def __init__(self, raw_sizes, context):
        super().__init__(raw_sizes, context)
        # Work that does not depend on the setting goes here, done once.
        self.genome = context.genome

    def utr_scores(self, sites):
        var = sites.var
        table = pd.DataFrame(
            {
                "chrom": var["chrom"].astype(str).to_numpy(),
                "position": genome.site_positions(var, "three_prime"),
                "strand": var["strand"].astype(str).to_numpy(),
            }
        )
        # Offset 0 is a 3' site's last base: 1 to 30 are the 30 nt after it.
        after, found = genome.read_sequences(self.genome, table, 1, 31)
        u_rich = np.array([sequence.count("T") >= 6 for sequence in after])
        return share_at_frequent_sites(
            var["utr_name"].to_numpy()[found], site_molecules(sites)[found], u_rich[found]
        )


SCORES.register("downstream-u", DownstreamUScore)
```

`context` holds what the command line gives a score (`ScoreContext`): the
`utr_end` of the sizes, the `--genome`, the `--known-sites`, the `--hamming`
and `--window` of the built-in scores, and the number of worker processes.
`needs` names the files it must have, so the commands refuse to start
without them. `share_at_frequent_sites` counts only the sites with more than
1% of their UTR's molecules, as utrcalling's scores do; it is not required.

With `raw_sizes` a raw-sizes count matrix (from `run -r`) and its genome:

```python
from utrcalling.core import peak_calling_anndata

scores = peak_calling_anndata.score_bandwidths(
    raw_sizes, genome_fasta, bandwidths=[4, 12], scores=["downstream-u", "pas"], n_jobs=1
)
print(scores)  # a row per kernel and bandwidth, a column per score
print(peak_calling_anndata.best_setting(scores))  # ('gaussian', 4) or ('gaussian', 12)
```

As a plugin, `--score downstream-u` works in `qc score-bandwidths`, and with
`-b optimize` in `run`, `count` and `call-peaks`.

## A peak caller

A peak caller takes the UTR sizes of one UTR's molecules, one per molecule,
and returns the size of each molecule's site, in the same order. It is
registered as a *factory*: a function that takes the bandwidth and the kernel
of the command line and returns the peak caller. A method without a kernel
ignores it; a method can read the bandwidth as its own scale, in nt, so that
`-b` sets it and `-b optimize` chooses it.

As an example, distance clustering, as CAGEr's `distclu` clusters
transcription start sites: sizes at most `-b` nt apart join one site, at its
most used size.

```python
from functools import partial

import numpy as np

from utrcalling.core.peak_calling.settings import PEAK_CALLERS


def distance_clusters(sizes, gap):
    """Each molecule's site: the most used size of its cluster of sizes at most gap nt apart."""
    sizes = np.asarray(sizes).ravel()
    order = np.argsort(sizes, kind="stable")
    ordered = sizes[order]
    sites = np.empty(len(sizes), dtype=np.int32)
    # A new cluster starts after each gap of more than `gap` nt.
    for cluster in np.split(np.arange(len(sizes)), np.flatnonzero(np.diff(ordered) > gap) + 1):
        values, counts = np.unique(ordered[cluster], return_counts=True)
        sites[order[cluster]] = values[counts.argmax()]
    return sites


def cluster_caller(bandwidth, kernel):
    if isinstance(bandwidth, str):
        raise ValueError(f"bandwidth {bandwidth!r}: the cluster caller takes a number of nt")
    return partial(distance_clusters, gap=bandwidth)


PEAK_CALLERS.register("cluster", cluster_caller)
```

Peaks are called in worker processes, so the peak caller must pickle: a
module-level function, or a `functools.partial` of one, as here; not a lambda
or a function defined inside another.

```python
from utrcalling.core import peak_calling_anndata

sites = peak_calling_anndata.peak_call_anndata_parallel(
    raw_sizes, method="cluster", bandwidth=20, n_jobs=1
)
```

As a plugin, `--peak-caller cluster -b 20` works in `run`, `count`,
`call-peaks`, `qc peaks` and `qc score-bandwidths`; `call-peaks` names its
files `<out>_cluster_bw20.h5ad`.

## A bandwidth rule

A rule takes the UTR sizes of one UTR's molecules (four or more, one per
molecule) and returns its bandwidth in nt. `-b NAME` uses it, and
`-b NAME:0.5` half of it; with a rule, each site of the peaks keeps its UTR's
bandwidth in `.var["bandwidth"]`.

As an example, a rule that measures the spread by the median absolute
deviation, which a second site in the UTR sways even less than Silverman's
interquartile range:

```python
import numpy as np

from utrcalling.core.peak_calling.settings import BANDWIDTH_RULES


@BANDWIDTH_RULES.register("mad")
def mad(sizes):
    """1.4826 x MAD x n^-1/5: Scott's rule with a robust spread."""
    spread = 1.4826 * np.median(np.abs(sizes - np.median(sizes)))
    return spread * len(sizes) ** -0.2
```

```python
from utrcalling.core.peak_calling.callers import get_caller

caller = get_caller("kde", bandwidth="mad:0.5", kernel="linear")
```

## A protocol

A protocol is a subclass of `Protocol` (`utrcalling/core/protocol.py`), which
tells the read mapping which reads count, where each molecule ends, and which
cell or sample it comes from. Start from the protocol nearest to yours, and
change what differs. Register it under a platform name, either as a class,
for any `--protocol-version`, or as a mapping from version to class:

```python
from utrcalling.core import protocol


class ProtocolSenseBulk(protocol.ProtocolTruseq):
    """Stranded bulk RNA-seq whose reads lie on the transcript's strand."""

    def check_strand_matches(self, read_strand, utr_strand):
        return read_strand == utr_strand


protocol.PROTOCOLS.register("sense-bulk", ProtocolSenseBulk)
```

Then `--protocol-platform sense-bulk` (as a plugin) reads with it. A protocol for
5' data subclasses `ProtocolFivePrime`, whose sizes count from the UTR's end
(`utr_end = "five_prime"`): the 5' scores and the commands that place sites in
the genome read it.

## Sharing a method as a plugin

A plugin is a Python package that declares its methods as entry points, in the
group of each registry:

| Registry | Entry point group |
| --- | --- |
| Scores | `utrcalling.scores` |
| Peak callers | `utrcalling.peak_callers` |
| Bandwidth rules | `utrcalling.bandwidth_rules` |
| Protocols | `utrcalling.protocols` |

For a package `utrcalling_degraded` with the score above in
`utrcalling_degraded/scores.py`, its `pyproject.toml`:

```toml
[project]
name = "utrcalling-degraded"
version = "0.1.0"
dependencies = ["utrcalling>=5"]

[project.entry-points."utrcalling.scores"]
downstream-u = "utrcalling_degraded.scores:DownstreamUScore"
```

The module defines the class without registering it: the entry point does.
Once the package is installed in the same environment (`pip install -e .`
while you develop it), `utrcalling qc score-bandwidths ... --score
downstream-u` finds it. utrcalling looks for plugins only when a name is not
its own, so they do not slow the command line.

## Testing a method

- Test it like utrcalling tests its own: `utrcalling/tests/test_peak_calling/`
  has tests of each score, peak caller and rule on small matrices whose
  answers are known, and `test_cli_workflows.py` runs whole command lines.
- Check it on data you know: does `qc score-bandwidths` rank the settings as
  you expect, and do the sites of the best one (`qc peaks`, `tools
  site-sequences`) look right?
- A score should change with what it measures, and little with what it does
  not: compare it with utrcalling's scores on the same settings.

## Adding a method to utrcalling

To add a method to utrcalling itself, rather than as a plugin, put it in its
own module beside the others, and add its name and "module:attribute" to the
registry's built-in methods. Built-in methods are imported only when used, so
the command line starts fast. Then test it, and describe it in the command's
help and in this documentation (`docs/`).

## Where the code is

```
utrcalling/core/
├── registry.py               Registry: names to methods, and plugin discovery
├── peak_calling/
│   ├── settings.py           The registries of peak calling, kernels and bandwidths
│   ├── callers.py            Peak callers: the interface and get_caller
│   ├── kde.py                The kde peak caller
│   ├── bandwidth_rules.py    scott, silverman, and each UTR's bandwidth
│   └── scores/
│       ├── __init__.py       SiteScore, ScoreContext, and helpers
│       ├── pas.py            pas
│       ├── initiator.py      initiator
│       └── known_sites.py    known-sites
├── peak_calling_anndata.py   Peak calling and scoring of count matrices
└── protocol.py               Protocols, and their registry PROTOCOLS
```
