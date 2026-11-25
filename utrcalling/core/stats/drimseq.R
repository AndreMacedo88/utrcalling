# utrcalling's bridge to DRIMSeq (Nowicka and Robinson 2016, F1000Research
# 5:1356), a Bioconductor package that users install themselves; it is not
# part of utrcalling. utrcalling runs this script with Rscript: it reads the
# files that utrcalling wrote in a directory, fits DRIMSeq's
# Dirichlet-multinomial models with the design matrices utrcalling built from
# the model formulas, and writes the results there for utrcalling to read.
#
#   Rscript drimseq.R DIRECTORY
#
# Reads:  counts.tsv (gene_id, feature_id and a column per sample),
#         samples.tsv (sample_id), full.tsv and reduced.tsv (design matrices,
#         a row per sample), settings.tsv (key, value: moderation, cores, seed).
# Writes: genes.tsv, features.tsv, precision.tsv, versions.tsv.

arguments <- commandArgs(trailingOnly = TRUE)
directory <- arguments[1]
read_table <- function(name) {
  read.delim(file.path(directory, name), check.names = FALSE, stringsAsFactors = FALSE)
}
write_table <- function(table, name) {
  write.table(table, file.path(directory, name), sep = "\t", quote = FALSE, row.names = FALSE)
}

settings <- read_table("settings.tsv")
setting <- function(key) settings$value[settings$key == key]

suppressPackageStartupMessages(library(DRIMSeq))

counts <- read_table("counts.tsv")
samples <- read_table("samples.tsv")
samples$group <- "all"
full <- as.matrix(read_table("full.tsv"))
reduced <- as.matrix(read_table("reduced.tsv"))

set.seed(as.integer(setting("seed")))
cores <- as.integer(setting("cores"))
workers <- if (cores > 1) {
  BiocParallel::MulticoreParam(workers = cores)
} else {
  BiocParallel::SerialParam()
}

data <- dmDSdata(counts = counts, samples = samples)
data <- dmPrecision(
  data,
  design = full,
  prec_moderation = setting("moderation"),
  prec_subset = 1,
  BPPARAM = workers
)
# Gene-wise precisions that could not be estimated take the common precision.
precision <- genewise_precision(data)
unestimated <- !is.finite(precision$genewise_precision)
if (any(unestimated)) {
  precision$genewise_precision[unestimated] <- common_precision(data)
  genewise_precision(data) <- precision
}
data <- dmFit(data, design = full, bb_model = TRUE, BPPARAM = workers)
data <- dmTest(data, design = reduced, bb_model = TRUE, BPPARAM = workers)

write_table(results(data), "genes.tsv")
write_table(results(data, level = "feature"), "features.tsv")
write_table(precision, "precision.tsv")
write_table(
  data.frame(
    name = c("R", "DRIMSeq"),
    version = c(
      paste(R.version$major, R.version$minor, sep = "."),
      as.character(packageVersion("DRIMSeq"))
    )
  ),
  "versions.tsv"
)
