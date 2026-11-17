# Protocols and input files

utrcalling measures each molecule's UTR size from the reads of a BAM file:
where its read ends in the UTR (3' data) or starts (5' data). How to find that
position, and which cell or sample a read comes from, depends on the
sequencing protocol, given with `--protocol-platform` (`-p`) and
`--protocol-version` (`-v`).

## Protocols

| Platform (`-p`) | Data | Read used | UMI | Duplicates | Notes |
| --- | --- | --- | --- | --- | --- |
| `10x` | 3' single-cell, modified 10x Chromium | Read 1 | Yes | keep first | An extended Read 1 reaches the 3' end of the UTR. Versions 2 and 3. |
| `standard_10x` | 3' single-cell, standard 10x Chromium | Read 2 | Yes | keep max | Cell Ranger BAM files, with `CB` and `UB` tags. Keeps each molecule's longest UTR size. Versions 2 and 3. |
| `5prime10x` | 5' single-cell, 10x Chromium | Read 1 | Yes | keep first | `CB` and `UB` tags. Version 1. |
| `smartseq` | 3' single-cell, modified Smart-seq3 | Read 1 | Yes | keep first | An extended Read 1 reaches the 3' end of the UTR. |
| `standard_smartseq` | 3' single-cell, standard Smart-seq3 | Any | No | none | One BAM per cell. Reads from both strands; the read's base furthest toward the polyA counts. |
| `smart-seq_cap` | 5' Cap-seq with Smart-seq3 | Read 1 | Yes | keep first | One BAM per sample. |
| `smart-seq_cap-readname` | 5' Cap-seq with Smart-seq3 | Read 1 | Yes | keep first | UMIs in the read names. |
| `truseq` | Stranded bulk RNA-seq (TruSeq Stranded mRNA) | Any | No | none | One BAM per sample, single- or paired-end. `--strandedness` `reverse` (default, right for dUTP libraries), `forward` or `unstranded`. |
| `bulk_pe` | Unstranded paired-end bulk RNA-seq (TruSeq RNA v2) | Any | No | fragment | One BAM per sample. Counts each fragment once, at its base furthest toward the polyA, which reaches further into the UTR than one mate. |

`--protocol-version` is 3 by default; `5prime10x` needs 1, and the bulk
protocols take any.

The standard protocols (`standard_10x`, `standard_smartseq`, and the bulk
ones) do not read the UTR's end directly: their reads stop somewhere in the
UTR, and the read's furthest base estimates how far the molecule reaches.
Their sites spread over many more nt than those of 3'-end protocols: give
them a wider bandwidth than the default of 12 (see [](peak-calling.md)).

## Input files

1. **A GTF annotation** of the genome, plain or gzipped, such as Ensembl's or
   Cell Ranger's. utrcalling takes its UTRs (`--feature three_prime_utr`, the
   default, or `five_prime_utr`) and merges the overlapping ones of each
   strand into the UTRs it maps reads to.
2. **BAM files** of the reads aligned to the same genome, sorted by
   coordinate and indexed (a `.bai` next to each). For `standard_10x`, Cell
   Ranger's `possorted_genome_bam.bam`.
3. **An annotation table** (CSV) of the cells or samples, `--annotations-file`
   (`-A`). Its other columns go to the count matrix's `.obs`, for example a
   treatment to filter sites by later.

### The annotation table

utrcalling finds its columns by their names, in any case:

- **A column whose name contains `barcode`**, for the protocols that tell
  cells apart by barcode (`10x`, `standard_10x`, `5prime10x`, `smartseq`):
  the cells to keep. Reads of other barcodes are left out.
- **A column whose name contains `sample_id`**, for the protocols with one BAM
  file per cell or sample (`smartseq`, `standard_smartseq`, `smart-seq_cap`,
  `smart-seq_cap-readname`, `truseq`, `bulk_pe`): an ID that appears in the
  path of its BAM file, such as `S1` for `aligned/S1.bam` or
  `S1/Aligned.sortedByCoord.out.bam`. Each must match one BAM file.
- **A column whose name contains `batch`**, optional: the batch (or sample) of
  each row, when several BAM files make one experiment. With
  `--label-batches true`, each barcode or sample ID gets its batch appended
  (`AAACCTGAGAAACCAT-1`), so barcodes that recur across batches stay apart.

For example, for 10x data of two samples, one BAM file each:

```text
barcode,batch,genotype
AAACCTGAGAAACCAT,1,wild-type
AAACCTGAGAAACCGC,1,wild-type
AAACCTGAGAAACCAT,2,knock-out
```

and for bulk RNA-seq:

```text
sample_id,treatment
S1,control
S2,interferon
```

## Running the pipeline

`utrcalling run` takes the annotation, the BAM files and the table, and writes
a count matrix of UTR sites (see [](outputs.md)):

```bash
# 3' 10x, modified protocol
utrcalling run Mus_musculus.GRCm39.110.gtf.gz \
    --alignments sample1.bam sample2.bam \
    --annotations-file cells.csv \
    --protocol-platform 10x --protocol-version 3 \
    --out counts.h5ad

# Standard 3' 10x, from Cell Ranger
utrcalling run genes.gtf.gz -a outs/possorted_genome_bam.bam -A cells.csv \
    -p standard_10x -v 3 -o counts.h5ad

# Standard Smart-seq3, one BAM per cell
utrcalling run genes.gtf.gz -a cell1.bam cell2.bam cell3.bam -A cells.csv \
    -p standard_smartseq -o counts.h5ad

# Unstranded paired-end bulk RNA-seq, with a wider bandwidth
utrcalling run genes.gtf.gz -a aligned/S1.bam aligned/S2.bam -A samples.csv \
    -p bulk_pe -b 30 -o counts.h5ad

# 5' Cap-seq: the 5' UTRs
utrcalling run genes.gtf.gz -a S1.bam S2.bam -A samples.csv \
    -p smart-seq_cap -f five_prime_utr -r -o capseq.h5ad
```

The options that matter most:

| Option | Description |
| --- | --- |
| `--feature / -f` | UTR type: `three_prime_utr` (default) or `five_prime_utr`. |
| `--summarization-level / -s` | `utr` (default), or `gene` to join the UTRs of each gene. |
| `--raw-sizes / -r` | Output both raw UTR sizes and KDE peaks. |
| `--kde-bandwidth / -b` | KDE bandwidth for peak calling: a number of nt (default: 12), or a rule that sets one per UTR from the spread of its sizes, `scott` or `silverman`, optionally scaled (`scott:0.5`). |
| `--kde-kernel` | KDE kernel: `gaussian` (default), `linear`, `epanechnikov`, `exponential`, `tophat` or `cosine`. With `-b optimize`, several to choose among. |
| `-b optimize`, `--candidates`, `--score`, `--genome`, `--known-sites` | Choose the bandwidth (and the kernel): score each candidate setting by its sites (see `qc score-bandwidths`) and use the best. |
| `--peak-caller` | The method that calls peaks: `kde` (default), or one that a plugin adds. |
| `--strandedness` | Library strandedness for `truseq`: `reverse` (default), `forward`, or `unstranded`. |
| `--batch-merge-method / -m` | How to merge batches: `union` or `intersection`. |
| `--store-intermediate / -i` | Save intermediate files for resumable workflows. |
| `--label-batches / -L` | `true` to append the batch to cell barcodes or sample IDs. |
| `--cores` | Maximum number of worker processes (default: all cores but one). |

A protocol for data that none of these read can be added without changing
utrcalling: see [](extending.md).
