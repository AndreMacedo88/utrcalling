# Peak calling

The reads of a UTR end (3' data) or start (5' data) at a spread of sizes
around each of its sites. Peak calling collapses each UTR's molecule sizes
into sites: the columns of the count matrix. `run` calls the peaks itself;
with `--raw-sizes` (`-r`) it also writes the raw sizes, whose peaks
`call-peaks` calls again with other settings.

## The kernel density estimate

utrcalling's peak caller is a kernel density estimate (KDE). It spreads each
molecule into a small curve, the kernel, and adds up the curves: each peak of
the sum is a site, and each molecule goes to the peak on its side of the
lowest points between peaks.

### What the bandwidth means

The bandwidth (`--kde-bandwidth`, `-b`) sets how wide each molecule's curve
is:

- With the Gaussian kernel (the default), the bandwidth is the standard
  deviation of the bell curve, in nt. With 12, a molecule weighs on sizes up to
  about 24 nt away (95% of its weight). Two sites with as many molecules stay
  two peaks only from 25 nt apart (more than twice the bandwidth); when one
  site has twice or ten times the molecules of the other, from 32 or 41 nt.
  Closer sites, and the scatter of a polyadenylation site's molecules over a
  few nt, become one peak. That suits 3' data: PolyA_DB, too, joins cleavage
  sites within 24 nt into one polyadenylation site (Tian et al. 2005).
- With the kernels that end, such as the linear one (a triangle), the
  bandwidth is the kernel's half-width: beyond it a molecule weighs nothing.
  With the linear kernel, two sites stay apart from the bandwidth plus 1 nt
  on, whatever their molecules: 3 nt for a bandwidth of 2. That suits
  transcription start sites, which lie a few nt apart.

So the same number means different things for different kernels: a linear
kernel smooths as much as a Gaussian of 0.4 times its bandwidth. The peak lies
where the curves add up most, which is not always on a size that molecules
have: the wider the bandwidth, the more it can move off the most used size.

### Kernels

`--kde-kernel` takes the kernels of scikit-learn's KernelDensity:
`gaussian` (the default), `linear`, `epanechnikov`, `exponential`, `tophat`
and `cosine`. The Gaussian smooths the most and suits 3' data; the linear
kernel keeps the peaks on the sizes where the molecules are, as transcription
start sites need. The exponential kernel never joins two sites of equal
molecules, whatever its bandwidth.

### Bandwidth rules

Instead of a number, a rule sets each UTR's bandwidth from the spread of its
molecules' sizes, narrower as they grow in number:

- `scott`: SD x n^-1/5, Scott's rule (the default of SciPy's `gaussian_kde`).
- `silverman`: 0.9 x min(SD, IQR / 1.34) x n^-1/5, Silverman's rule of thumb,
  as R's `bw.nrd0`. A UTR's other sites inflate its SD more than its IQR.

A factor scales them: `-b scott:0.5` is half of Scott's bandwidth. Each site
keeps its UTR's bandwidth in `.var["bandwidth"]`.

## Choosing the peak calling for your data

`qc score-bandwidths` calls the peaks of a raw-sizes matrix with each kernel
and bandwidth you give, and scores the sites of each by what true sites
have. Only sites with more than 1% of their UTR's molecules count; a
setting's score averages, over UTRs, the share of molecules at sites that
have it. Higher is better:

| Score (`--score`) | For | Sites that count |
| --- | --- | --- |
| `pas` | 3' data (the default) | 10-30 nt after a polyadenylation signal, AATAAA (up to `--hamming` mismatches), in the genome |
| `initiator` | 5' data (the default) | With C or T at -1 and A or G at +1, the start site, where RNA polymerase II starts most transcripts |
| `known-sites` | Either | Within `--window` nt of the `--known-sites` of a database, such as PolyASite (3') or refTSS (5') |

```bash
# Gaussian and linear kernels, by the polyadenylation signal (3' data)
utrcalling qc score-bandwidths raw_sizes.h5ad scores.tsv genome.fa \
    --kde-kernel gaussian linear --bandwidths 4 8 12 16

# 5' data, by the initiator and by refTSS
utrcalling qc score-bandwidths capseq.h5ad scores.tsv GRCm39.fa --kde-kernel linear \
    --score initiator known-sites --known-sites refTSS_v4.1_mouse_coordinate.mm39.bed.gz
```

With `-b optimize`, `run`, `count` and `call-peaks` score the `--candidates`
(by default those of the data's UTR end) with each `--kde-kernel` given, call
the peaks with the best, and write the scores to `<out>_bandwidth_scores.tsv`:

```bash
utrcalling call-peaks raw_sizes.h5ad peaks -b optimize --kde-kernel gaussian linear \
    --genome genome.fa
```

For 3' data, the default of 12 scores close to the best: the polyadenylation
signal score barely changes between 6 and 20 nt.

A database of known sites knows clusters of sites, not each nucleotide:
refTSS's clusters are 14 nt wide in the median. Smoothing that joins the sites
of a cluster raises the known-sites score, also when it moves the peak off
the start site. On the Cap-seq data below, it rose from 0.77 to 0.83 between
linear bandwidths of 1 and 12 nt, while the initiator score fell from 0.54
to 0.47. It checks that sites lie where others have seen sites, and compares
settings at the scale of the clusters; for 5' data, let the initiator choose.

## How stable each peak is

`qc peaks` describes the peaks of one setting: their molecules, share of
their UTR, and the spread of the raw sizes they collect. With `--bootstrap N`
it draws each UTR's molecules again, with replacement, N times, calls their
peaks, and gives each peak the share of draws in which it comes back within
its UTR's bandwidth:

```bash
utrcalling qc peaks raw_sizes.h5ad peaks.tsv --bandwidth 12 --bootstrap 100
```

Peaks of few molecules, or on the flank of a larger one, come back less often:
filter them out, or check them. Stability compares the peaks of one setting,
not settings: wider bandwidths make every peak more stable (one peak per UTR
always comes back).

## Peak calling for 5' data

The default KDE, Gaussian with a bandwidth of 12 nt, suits 3' data, where the
molecules of a polyadenylation site spread over a few nt. Transcription start
sites are exact to the nucleotide, and promoters often use several within a few
nt of each other; conditions that change which start sites cells use can shift
them by little. In mouse neural stem cells treated with interferon, the 5' UTRs
that shortened lost a median of 36 nt, less than 30 nt in 39% of them (Guerrero
Aruffo, 2026). A 12-nt Gaussian joins such sites and moves its peaks off the
start sites: on that Cap-seq data, its initiator score is 0.22, while 42% of
the molecules themselves start at a pyrimidine-purine initiator.

When the start sites in your 5' data may move by a few nt, choose the peak
calling for the data, by the initiator (above). The default stays as it is for
other analyses.

### Interferon-treated neural stem cells (bulk Cap-seq)

For 5' Cap-seq data of adult neural stem cells treated with interferon, Eugenia
Guerrero Aruffo found these settings best in her master's thesis (2026); these
commands give her sites in 99% of the UTRs:

```bash
# -p: your 5' protocol, such as smart-seq_cap
utrcalling run Mus_musculus.GRCm39.108.gtf.gz -a sample_*.bam -A samples.csv \
    -f five_prime_utr -p smart-seq_cap -r -o capseq.h5ad
utrcalling call-peaks capseq.h5ad capseq -b scott:0.5 --kde-kernel linear
utrcalling process filter-utrs capseq_linear_bwscott0.5.h5ad deep.h5ad --minimum-molecules 500
utrcalling process filter-sites deep.h5ad sites.h5ad \
    --threshold 0.1 --minimum-counts 10 --groupby Treatment
```

- `--kde-kernel linear`: a triangle, sharp at its centre and zero beyond the
  bandwidth, keeps the peaks on the start sites.
- `-b scott:0.5`: per UTR, half of Scott's bandwidth (SD x n^-1/5), finer where
  the molecules are many: 2 nt in the median UTR of that data.
- `filter-utrs`: UTRs with fewer molecules gave unsure sites.
- `filter-sites --groupby Treatment`: thresholds per treatment, so start sites
  that interferon raises stay, though they are a small share of all molecules.

The bandwidths and the filters count the molecules of all the samples in the
matrix: subset it to the samples you study first (she called the sites of the
wild-type Cap-seq libraries).

On her UTRs with 500 molecules or more, `-b optimize --kde-kernel linear`
chooses 2 nt (an initiator score of 0.60, against 0.59 for `scott:0.5`). From
the sites, `tools site-sequences sites.h5ad genome.fa tss.fa` writes the
sequences around the start sites for motif analyses, and `qc priming` flags
those that strand invasion could explain.

## Other peak callers

A method other than the KDE can be added as a plugin and chosen with
`--peak-caller`; see [](extending.md).
