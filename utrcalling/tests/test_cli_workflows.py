"""The command chain of the paper workflows, run end to end from the command line.

Each test runs `utrcalling <command> <flags>` in this process, with the flags the
Snakemake workflows of Macedo et al. 2026 use (here on the 10x v2 mouse test data,
at UTR level). These command lines must keep working when the CLI changes, so they
keep the command names of before version 4.0.0, as the workflows do.
"""

import os
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse

from utrcalling import runtime
from utrcalling.core import provenance
from utrcalling.core.peak_calling.bandwidth_rules import utr_bandwidth
from utrcalling.core.utr_mapping import MAPPING_STEPS

TEST_DATA = Path(__file__).resolve().parent / "test_data"
GTF = TEST_DATA / "test_mus_musculus_GRCm38_102.gtf.gz"
# The old cells are in the first BAM and the young cells in the second.
BAMS = [
    TEST_DATA / "test_UTRoldNSCgenome_bwt2_102_sort.bam",
    TEST_DATA / "test_UTRyoungNSCgenome_bwt2_102_sort.bam",
]
ANNOTATIONS = TEST_DATA / "vivo_anno.csv"
# Flags that `run` and `call-peaks` share in the paper's 10x workflow.
PIPELINE_FLAGS = ["-s", "utr", "-p", "10x", "-v", "2", "-m", "union"]
PIPELINE_FLAGS += ["-A", ANNOTATIONS, "-L", "True"]

# Output of the paper's `run` command, shared by the tests below (setUpModule).
WORKDIR = Path()


def utrcalling(*args, cwd):
    """Run `utrcalling <args>` from directory *cwd*, in this process."""
    previous = Path.cwd()
    os.chdir(cwd)
    try:
        with mock.patch.object(sys, "argv", ["utrcalling", *map(str, args)]):
            runtime.main()
    finally:
        os.chdir(previous)


def _dense(matrix):
    return matrix.toarray() if scipy.sparse.issparse(matrix) else np.asarray(matrix)


def assert_same_anndata(actual, expected):
    """Same cells, sites, annotations and counts, in the same order."""
    pd.testing.assert_index_equal(actual.obs_names, expected.obs_names)
    pd.testing.assert_index_equal(actual.var_names, expected.var_names)
    pd.testing.assert_frame_equal(actual.obs, expected.obs)
    pd.testing.assert_frame_equal(actual.var, expected.var)
    np.testing.assert_array_equal(_dense(actual.X), _dense(expected.X))


def expected_site_names(var):
    """<gene>_<n>-<size>, worked out here independently of utrcalling's code.

    Genes without a name use the Ensembl ID at the start of their UTR name; n
    numbers each gene's sites by UTR name and then size.
    """
    sites = []
    utr_names = var.utr_name.astype(str)
    sizes = var.utr_size.astype(int)
    genes = var.gene_name.astype(object)
    for row, (utr, gene, size) in enumerate(zip(utr_names, genes, sizes, strict=True)):
        if pd.isna(gene) or str(gene) == "None":
            gene = re.split(r"[-_]", utr)[0]
        sites.append((str(gene), utr, size, row))
    names = [""] * len(sites)
    numbers = {}
    for gene, _utr, size, row in sorted(sites):
        numbers[gene] = numbers.get(gene, 0) + 1
        names[row] = f"{gene}_{numbers[gene]}-{size}"
    return names


def place_utrs(adata):
    """*adata* with its UTRs moved onto chromosome 1 of a small genome, and its size.

    The real genome is too big for a test. Each UTR keeps its length and strand.
    """
    placed = adata.copy()
    var = placed.var
    utr, _ = pd.factorize(var.utr_name.astype(str))
    length = var.end - var.start
    spacing = int(length.max()) + 100
    var["chrom"] = "chr1"
    var["start"] = utr * spacing + 51
    var["end"] = var.start + length
    return placed, (utr.max() + 1) * spacing


def random_genome(size, seed):
    """A random sequence of *size* nt."""
    return "".join(np.random.default_rng(seed).choice(list("ACGT"), size=size))


def read_summary(path):
    """A mapping summary table, as utrcalling writes it."""
    return pd.read_csv(path, sep="\t", index_col="input").astype("Int64")


def commands_run(adata):
    """The commands that made *adata*, by name: ["run", "call-peaks-...", ...]."""
    return [command.split()[1] for command in provenance.recorded(adata)["commands"]]


def setUpModule():
    global WORKDIR
    WORKDIR = Path(tempfile.mkdtemp())
    utrcalling(
        "run",
        GTF,
        "-a",
        *BAMS,
        "-f",
        "three_prime_utr",
        *PIPELINE_FLAGS,
        "-r",
        "-i",
        "-o",
        "raw.h5ad",
        cwd=WORKDIR,
    )


def tearDownModule():
    shutil.rmtree(WORKDIR)


class _InTempDir(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.addClassCleanup(shutil.rmtree, cls.tmp)


class TestRunCommand(unittest.TestCase):
    """`utrcalling run -r -i` with the flags of the paper's 10x workflow."""

    @classmethod
    def setUpClass(cls):
        cls.raw = ad.read_h5ad(WORKDIR / "raw.h5ad")
        cls.kde = ad.read_h5ad(WORKDIR / "raw_kde.h5ad")
        cls.annotations = pd.read_csv(ANNOTATIONS, index_col="CellBarcode")

    def test_sites_are_named_gene_number_size(self):
        for adata in (self.raw, self.kde):
            names = list(adata.var_names)
            self.assertListEqual(names, expected_site_names(adata.var))
            self.assertEqual(len(set(names)), len(names))

    def test_outputs_record_how_they_were_made(self):
        for adata, matrix in ((self.raw, "raw_sizes"), (self.kde, "peaks")):
            info = provenance.recorded(adata)
            with self.subTest(matrix=matrix):
                self.assertEqual(info["matrix"], matrix)
                self.assertEqual(info["utr_end"], "three_prime")
                self.assertEqual(info["summarization_level"], "utr")
                self.assertEqual(info["protocol_platform"], "10x")
                self.assertEqual(info["protocol_version"], 2)
                self.assertListEqual(commands_run(adata), ["run"])
                self.assertIn("-L True -r -i -o raw.h5ad", info["commands"][0])
        self.assertEqual(provenance.recorded(self.kde)["bandwidth"], 12)
        self.assertNotIn("bandwidth", provenance.recorded(self.raw))

    def test_the_mapping_summary_follows_every_read(self):
        summary = read_summary(WORKDIR / "raw_mapping_summary.tsv")
        bams = [str(bam) for bam in BAMS]
        self.assertListEqual(list(summary.index), [*bams, "total"])
        # Every alignment is skipped at one check, or kept as a read.
        skipped = list(
            MAPPING_STEPS[MAPPING_STEPS.index("unmapped") : MAPPING_STEPS.index("reads")]
        )
        per_bam = summary.loc[bams]
        pd.testing.assert_series_equal(
            per_bam.alignments, per_bam[skipped].sum(axis=1) + per_bam.reads, check_names=False
        )
        total = summary.loc["total"]
        self.assertEqual(total.alignments, per_bam.alignments.sum())
        self.assertGreater(total.reads, total.molecules)  # UMI duplicates
        self.assertEqual(total.molecules_after_merge, total.molecules)  # -m union
        self.assertLess(total.molecules_in_cell_list, total.molecules_after_size_filter)
        for adata in (self.raw, self.kde):
            self.assertEqual(total.molecules_counted, adata.X.sum())
            recorded = provenance.recorded(adata)["mapping_summary"]
            pd.testing.assert_frame_equal(recorded, summary, check_index_type=False)

    def test_cells_and_their_columns_come_from_the_annotation_file(self):
        self.assertGreater(self.raw.n_obs, 0)
        expected = self.annotations.loc[self.raw.obs_names]
        for column in expected.columns:
            with self.subTest(column=column):
                self.assertListEqual(
                    self.raw.obs[column].astype(str).tolist(), expected[column].astype(str).tolist()
                )

    def test_cells_are_labelled_with_the_batch_of_their_bam(self):
        # -L True: each barcode gets the 1-based position of its BAM as suffix.
        suffix = self.raw.obs_names.str.rsplit("-", n=1).str[1]
        expected = self.raw.obs.CellAge.astype(str).map({"old": "1", "young": "2"})
        self.assertListEqual(suffix.tolist(), expected.tolist())

    def test_kde_peaks_keep_every_molecule_of_every_cell(self):
        np.testing.assert_array_equal(
            _dense(self.kde.X).sum(axis=1), _dense(self.raw.X).sum(axis=1)
        )
        pd.testing.assert_index_equal(self.kde.obs_names, self.raw.obs_names)

    def test_kde_peaks_lie_within_the_sizes_of_their_utr(self):
        ranges = self.raw.var.groupby("utr_name", observed=True).utr_size.agg(["min", "max"])
        peaks = self.kde.var.join(ranges, on="utr_name")
        self.assertTrue(peaks.utr_size.between(peaks["min"], peaks["max"]).all())
        self.assertLess(self.kde.n_vars, self.raw.n_vars)

    def test_intermediate_files_are_stored(self):
        self.assertTrue(
            {
                "utr_reference_regions.parquet",
                "UTR_sizes_counts_1.parquet",
                "UTR_sizes_counts_2.parquet",
                "UTR_sizes_counts_merged_union.parquet",
            }
            <= {path.name for path in (WORKDIR / "intermediate").iterdir()}
        )


class TestRunCommandWithoutRawSizes(_InTempDir):
    """Without -r, `run` writes only the peaks: the same as the _kde file of -r."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        utrcalling(
            "run",
            GTF,
            "-a",
            *BAMS,
            "-f",
            "three_prime_utr",
            *PIPELINE_FLAGS,
            "-o",
            "peaks.h5ad",
            cwd=cls.tmp,
        )

    def test_peaks_match_the_kde_output_of_raw_sizes(self):
        assert_same_anndata(
            ad.read_h5ad(self.tmp / "peaks.h5ad"), ad.read_h5ad(WORKDIR / "raw_kde.h5ad")
        )

    def test_writes_neither_raw_sizes_nor_intermediate_files(self):
        written = {path.name for path in self.tmp.iterdir()}
        self.assertIn("peaks.h5ad", written)
        self.assertNotIn("peaks_kde.h5ad", written)
        self.assertNotIn("intermediate", written)


class TestCallPeaksCommand(_InTempDir):
    """`call-peaks` on the intermediate files of `run` (the paper's fallback rule)."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        intermediate = WORKDIR / "intermediate"
        utrcalling(
            "call-peaks",
            intermediate / "utr_reference_regions.parquet",
            "-c",
            intermediate / "UTR_sizes_counts_1.parquet",
            intermediate / "UTR_sizes_counts_2.parquet",
            *PIPELINE_FLAGS,
            "-r",
            "-i",
            "-o",
            "peaks.h5ad",
            cwd=cls.tmp,
        )
        # The merged counts file that -i stores is a single input on its own.
        utrcalling(
            "call-peaks",
            intermediate / "utr_reference_regions.parquet",
            "-c",
            intermediate / "UTR_sizes_counts_merged_union.parquet",
            *PIPELINE_FLAGS,
            "-r",
            "-o",
            "merged.h5ad",
            cwd=cls.tmp,
        )
        # Without -r: only the peaks.
        utrcalling(
            "call-peaks",
            intermediate / "utr_reference_regions.parquet",
            "-c",
            intermediate / "UTR_sizes_counts_1.parquet",
            intermediate / "UTR_sizes_counts_2.parquet",
            *PIPELINE_FLAGS,
            "-o",
            "peaks_only.h5ad",
            cwd=cls.tmp,
        )

    def test_raw_sizes_match_run(self):
        assert_same_anndata(
            ad.read_h5ad(self.tmp / "peaks.h5ad"), ad.read_h5ad(WORKDIR / "raw.h5ad")
        )

    def test_the_merged_counts_file_alone_gives_the_same_result(self):
        assert_same_anndata(
            ad.read_h5ad(self.tmp / "merged.h5ad"), ad.read_h5ad(WORKDIR / "raw.h5ad")
        )

    def test_without_raw_sizes_writes_only_the_peaks(self):
        assert_same_anndata(
            ad.read_h5ad(self.tmp / "peaks_only.h5ad"), ad.read_h5ad(WORKDIR / "raw_kde.h5ad")
        )
        self.assertFalse((self.tmp / "peaks_only_kde.h5ad").exists())

    def test_peaks_match_run(self):
        assert_same_anndata(
            ad.read_h5ad(self.tmp / "peaks_kde.h5ad"), ad.read_h5ad(WORKDIR / "raw_kde.h5ad")
        )


class TestRunCommandAtGeneLevel(_InTempDir):
    """`run -s gene` (the paper's mouse and blood workflows) joins the UTRs of each gene."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        gene_flags = [flag if flag != "utr" else "gene" for flag in PIPELINE_FLAGS]
        utrcalling("run", GTF, "-a", *BAMS, *gene_flags, "-r", "-i", "-o", "gene.h5ad", cwd=cls.tmp)
        intermediate = cls.tmp / "intermediate"
        utrcalling(
            "call-peaks",
            intermediate / "utr_reference_regions.parquet",
            "-c",
            intermediate / "UTR_sizes_counts_1.parquet",
            intermediate / "UTR_sizes_counts_2.parquet",
            *gene_flags,
            "-r",
            "-o",
            "call_peaks.h5ad",
            cwd=cls.tmp,
        )
        cls.gene = ad.read_h5ad(cls.tmp / "gene.h5ad")
        cls.utr = ad.read_h5ad(WORKDIR / "raw.h5ad")

    def test_a_genes_utrs_become_one(self):
        # At UTR level, a gene's second, third... UTR are named <gene ID>_2, _3...
        second_utrs = r"_\d+$"
        self.assertTrue(self.utr.var.utr_name.astype(str).str.contains(second_utrs).any())
        self.assertFalse(self.gene.var.utr_name.astype(str).str.contains(second_utrs).any())

    def test_every_molecule_of_every_cell_is_kept(self):
        np.testing.assert_array_equal(
            _dense(self.gene[self.utr.obs_names].X).sum(axis=1), _dense(self.utr.X).sum(axis=1)
        )

    def test_call_peaks_gives_the_same_result(self):
        assert_same_anndata(ad.read_h5ad(self.tmp / "call_peaks.h5ad"), self.gene)


class TestRunCommandWithOneBam(_InTempDir):
    """A sample's cells get the same counts whether it is run alone or with others."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        utrcalling(
            "run", GTF, "-a", BAMS[0], *PIPELINE_FLAGS, "-r", "-i", "-o", "old.h5ad", cwd=cls.tmp
        )
        cls.alone = ad.read_h5ad(cls.tmp / "old.h5ad")
        cls.together = ad.read_h5ad(WORKDIR / "raw.h5ad")[cls.alone.obs_names]

    def test_stores_the_mappings_of_its_one_bam(self):
        # The same mappings as the first BAM's file of the run with both samples.
        stored = pd.read_parquet(self.tmp / "intermediate" / "UTR_sizes_counts.parquet")
        first = pd.read_parquet(WORKDIR / "intermediate" / "UTR_sizes_counts_1.parquet")
        columns = ["utr_name", "observation_id", "utr_size"]

        def rows(df):
            return df[columns].astype(str).sort_values(columns).reset_index(drop=True)

        pd.testing.assert_frame_equal(rows(stored), rows(first))

    def test_has_only_the_cells_of_that_sample(self):
        self.assertSetEqual(set(self.alone.obs.CellAge.astype(str)), {"old"})

    def test_counts_match_the_run_with_both_samples(self):
        together_sites = {key: i for i, key in enumerate(_site_keys(self.together))}
        columns = [together_sites[key] for key in _site_keys(self.alone)]
        np.testing.assert_array_equal(_dense(self.alone.X), _dense(self.together.X)[:, columns])
        # and no site with molecules of these cells is missing
        seen = _dense(self.together.X).sum(axis=0) > 0
        self.assertEqual(int(seen.sum()), len(columns))


class TestPrepareReferenceThenCallPeaks(_InTempDir):
    """Running the pipeline in two halves gives the same result as `run`.

    With the names of the paper's workflows: `map-reads` and `count` since 5.0.0.
    prepare-reference-and-map-reads used to crash on every call: the protocol
    needs the annotation file, which the command did not accept.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        utrcalling(
            "prepare-reference-and-map-reads",
            GTF,
            "-a",
            *BAMS,
            "-A",
            ANNOTATIONS,
            "-L",
            "True",
            "-f",
            "three_prime_utr",
            "-p",
            "10x",
            "-v",
            "2",
            cwd=cls.tmp,
        )
        intermediate = cls.tmp / "intermediate"
        utrcalling(
            "call-peaks",
            intermediate / "utr_reference_regions.parquet",
            "-c",
            intermediate / "UTR_sizes_counts_1.parquet",
            intermediate / "UTR_sizes_counts_2.parquet",
            *PIPELINE_FLAGS,
            "-r",
            "-o",
            "halves.h5ad",
            cwd=cls.tmp,
        )

    def test_raw_sizes_match_run(self):
        assert_same_anndata(
            ad.read_h5ad(self.tmp / "halves.h5ad"), ad.read_h5ad(WORKDIR / "raw.h5ad")
        )

    def test_map_reads_and_count_summarise_the_mapping_as_run_does(self):
        run_summary = read_summary(WORKDIR / "raw_mapping_summary.tsv")
        mapped = read_summary(self.tmp / "intermediate" / "mapping_summary.tsv")
        self.assertListEqual(list(mapped.columns), list(MAPPING_STEPS))
        pd.testing.assert_frame_equal(mapped, run_summary[list(MAPPING_STEPS)])
        # count reads the mapping steps stored with the read mappings.
        counted = read_summary(self.tmp / "halves_mapping_summary.tsv")
        intermediate = self.tmp / "intermediate"
        mappings = [str(intermediate / f"UTR_sizes_counts_{n}.parquet") for n in (1, 2)]
        self.assertListEqual(list(counted.index), [*mappings, "total"])
        pd.testing.assert_frame_equal(
            counted.reset_index(drop=True), run_summary.reset_index(drop=True)
        )

    def test_outputs_record_the_command_as_typed(self):
        raw = provenance.recorded(ad.read_h5ad(self.tmp / "halves.h5ad"))
        kde = provenance.recorded(ad.read_h5ad(self.tmp / "halves_kde.h5ad"))
        self.assertEqual((raw["matrix"], kde["matrix"]), ("raw_sizes", "peaks"))
        for info in (raw, kde):
            self.assertEqual(len(info["commands"]), 1)
            self.assertTrue(info["commands"][0].startswith("utrcalling call-peaks "))

    def test_peaks_match_run(self):
        assert_same_anndata(
            ad.read_h5ad(self.tmp / "halves_kde.h5ad"), ad.read_h5ad(WORKDIR / "raw_kde.h5ad")
        )


def _site_keys(adata):
    """(utr_name, utr_size) of each site, which identifies it whatever its name."""
    return list(zip(adata.var.utr_name.astype(str), adata.var.utr_size.astype(int), strict=True))


class TestCallPeaksDifferentBandwidthsCommand(_InTempDir):
    """`call-peaks-different-bandwidths` on the raw sizes of `run`."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        utrcalling(
            "call-peaks-different-bandwidths",
            WORKDIR / "raw.h5ad",
            "peaks",
            "-b",
            "12",
            "60",
            "--cores",
            "2",
            cwd=cls.tmp,
        )
        cls.bw12 = ad.read_h5ad(cls.tmp / "peaks_bw12.h5ad")
        cls.bw60 = ad.read_h5ad(cls.tmp / "peaks_bw60.h5ad")

    def test_bandwidth_12_finds_the_peaks_of_run(self):
        # run's own KDE uses bandwidth 12 too; only the site names differ.
        kde = ad.read_h5ad(WORKDIR / "raw_kde.h5ad")
        self.assertCountEqual(_site_keys(self.bw12), _site_keys(kde))
        order = {key: i for i, key in enumerate(_site_keys(kde))}
        columns = [order[key] for key in _site_keys(self.bw12)]
        bw12 = self.bw12[kde.obs_names]
        np.testing.assert_array_equal(_dense(bw12.X), _dense(kde.X)[:, columns])

    def test_peak_calling_the_raw_sizes_reproduces_the_peaks_of_run(self):
        # Same sites, counts, annotations and names; only the cells come sorted.
        kde = ad.read_h5ad(WORKDIR / "raw_kde.h5ad")
        assert_same_anndata(self.bw12[kde.obs_names], kde)

    def test_sites_are_named_gene_number_size(self):
        for adata in (self.bw12, self.bw60):
            self.assertListEqual(list(adata.var_names), expected_site_names(adata.var))

    def test_every_bandwidth_keeps_every_molecule(self):
        raw_totals = _dense(ad.read_h5ad(WORKDIR / "raw.h5ad").X).sum(axis=1)
        for adata in (self.bw12, self.bw60):
            np.testing.assert_array_equal(_dense(adata.X).sum(axis=1), raw_totals)

    def test_a_wider_bandwidth_merges_peaks(self):
        self.assertLess(self.bw60.n_vars, self.bw12.n_vars)

    def test_each_file_records_its_bandwidth(self):
        for adata, bandwidth in ((self.bw12, 12), (self.bw60, 60)):
            info = provenance.recorded(adata)
            self.assertEqual(info["matrix"], "peaks")
            self.assertEqual(info["bandwidth"], bandwidth)
            self.assertEqual(info["kernel"], "gaussian")
            self.assertListEqual(commands_run(adata), ["run", "call-peaks-different-bandwidths"])

    def test_bandwidth_0_keeps_the_raw_sizes(self):
        utrcalling("call-peaks", WORKDIR / "raw.h5ad", "same", "-b", "0", cwd=self.tmp)
        same = ad.read_h5ad(self.tmp / "same_bw0.h5ad")
        raw = ad.read_h5ad(WORKDIR / "raw.h5ad")
        assert_same_anndata(same, raw[:, same.var_names].copy())
        self.assertEqual(provenance.recorded(same)["matrix"], "raw_sizes")
        self.assertEqual(provenance.recorded(same)["bandwidth"], 0)
        self.assertNotIn("kernel", provenance.recorded(same))
        self.assertListEqual(commands_run(same), ["run", "call-peaks"])


class TestKdeKernel(_InTempDir):
    """`--kde-kernel` calls the peaks of `count` and of `call-peaks` alike."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        intermediate = WORKDIR / "intermediate"
        utrcalling(
            "count",
            intermediate / "utr_reference_regions.parquet",
            "-c",
            intermediate / "UTR_sizes_counts_1.parquet",
            intermediate / "UTR_sizes_counts_2.parquet",
            *PIPELINE_FLAGS,
            "--kde-bandwidth",
            "3",
            "--kde-kernel",
            "linear",
            "-o",
            "linear.h5ad",
            cwd=cls.tmp,
        )
        utrcalling(
            "call-peaks",
            WORKDIR / "raw.h5ad",
            "peaks",
            "-b",
            "3",
            "--kde-kernel",
            "linear",
            cwd=cls.tmp,
        )
        cls.count = ad.read_h5ad(cls.tmp / "linear.h5ad")
        # Files name the kernel, unless it is the Gaussian.
        cls.call_peaks = ad.read_h5ad(cls.tmp / "peaks_linear_bw3.h5ad")
        utrcalling(
            "call-peaks",
            WORKDIR / "raw.h5ad",
            "both",
            "-b",
            "3",
            "0",
            "--kde-kernel",
            "linear",
            "gaussian",
            cwd=cls.tmp,
        )

    def test_count_and_call_peaks_give_the_same_sites(self):
        assert_same_anndata(self.call_peaks[self.count.obs_names], self.count)

    def test_the_linear_kernel_gives_other_sites_than_the_gaussian(self):
        gaussian = pd.Index(_site_keys(ad.read_h5ad(WORKDIR / "raw_kde.h5ad")))
        self.assertFalse(pd.Index(_site_keys(self.count)).equals(gaussian))

    def test_every_molecule_is_kept_at_a_size_of_its_utr(self):
        raw = ad.read_h5ad(WORKDIR / "raw.h5ad")
        np.testing.assert_array_equal(
            _dense(self.count.X).sum(axis=1), _dense(raw[self.count.obs_names].X).sum(axis=1)
        )
        sizes = raw.var.groupby("utr_name", observed=True).utr_size.agg(["min", "max"])
        site = self.count.var.join(sizes, on="utr_name")
        self.assertTrue(((site.utr_size >= site["min"]) & (site.utr_size <= site["max"])).all())

    def test_several_kernels_give_a_file_each(self):
        written = sorted(path.name for path in self.tmp.glob("both_*.h5ad"))
        # Bandwidth 0 keeps the raw sizes, whatever the kernel: one file.
        self.assertListEqual(written, ["both_bw0.h5ad", "both_bw3.h5ad", "both_linear_bw3.h5ad"])
        assert_same_anndata(ad.read_h5ad(self.tmp / "both_linear_bw3.h5ad"), self.call_peaks)
        gaussian = ad.read_h5ad(self.tmp / "both_bw3.h5ad")
        self.assertEqual(provenance.recorded(gaussian)["kernel"], "gaussian")
        raw = ad.read_h5ad(self.tmp / "both_bw0.h5ad")
        self.assertEqual(raw.n_vars, ad.read_h5ad(WORKDIR / "raw.h5ad").n_vars)

    def test_both_record_the_kernel(self):
        for adata in (self.count, self.call_peaks):
            info = provenance.recorded(adata)
            self.assertEqual((info["bandwidth"], info["kernel"]), (3, "linear"))


class TestBandwidthRule(_InTempDir):
    """A rule sets each UTR's bandwidth, in `count` and `call-peaks` alike.

    With the settings Eugenia Guerrero Aruffo chose for 5' Cap-seq data: half
    of Scott's bandwidth, with the linear kernel.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        intermediate = WORKDIR / "intermediate"
        rule = ["-b", "scott:0.5", "--kde-kernel", "linear"]
        utrcalling(
            "count",
            intermediate / "utr_reference_regions.parquet",
            "-c",
            intermediate / "UTR_sizes_counts_1.parquet",
            intermediate / "UTR_sizes_counts_2.parquet",
            *PIPELINE_FLAGS,
            *rule,
            "-r",
            "-o",
            "count.h5ad",
            cwd=cls.tmp,
        )
        utrcalling("call-peaks", WORKDIR / "raw.h5ad", "peaks", *rule, cwd=cls.tmp)
        utrcalling("qc", "report", "count_kde.h5ad", "-o", "report.tsv", cwd=cls.tmp)
        cls.count = ad.read_h5ad(cls.tmp / "count_kde.h5ad")
        cls.call_peaks = ad.read_h5ad(cls.tmp / "peaks_linear_bwscott0.5.h5ad")

    def test_count_and_call_peaks_give_the_same_sites_and_bandwidths(self):
        assert_same_anndata(self.call_peaks[self.count.obs_names], self.count)

    def test_each_site_has_the_bandwidth_of_its_utr(self):
        raw = ad.read_h5ad(WORKDIR / "raw.h5ad")
        molecules = np.asarray(raw.X.sum(axis=0)).ravel().astype(int)
        for utr, columns in raw.var.groupby("utr_name", observed=True).indices.items():
            sizes = np.sort(np.repeat(raw.var.utr_size.to_numpy()[columns], molecules[columns]))
            expected = utr_bandwidth(sizes, "scott:0.5")
            recorded = self.count.var.bandwidth[self.count.var.utr_name == utr].unique()
            with self.subTest(utr=utr):
                np.testing.assert_array_equal(recorded, [expected])

    def test_the_record_and_the_report_name_the_rule(self):
        for adata in (self.count, self.call_peaks):
            info = provenance.recorded(adata)
            self.assertEqual((info["bandwidth"], info["kernel"]), ("scott:0.5", "linear"))
        report = pd.read_csv(self.tmp / "report.tsv", sep="\t", index_col="file")
        self.assertEqual(report.loc["count_kde.h5ad", "bandwidth"], "scott:0.5")


class TestOptimize(_InTempDir):
    """`-b optimize` scores the candidate bandwidths and calls the peaks with the best."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # call-peaks, scored for real: run's raw sizes on a small random genome.
        placed, genome_size = place_utrs(ad.read_h5ad(WORKDIR / "raw.h5ad"))
        placed.write_h5ad(cls.tmp / "placed.h5ad")
        (cls.tmp / "genome.fa").write_text(f">1\n{random_genome(genome_size, seed=1)}\n")
        candidates = ["--candidates", "4", "12", "60"]
        utrcalling(
            "call-peaks",
            "placed.h5ad",
            "opt",
            "-b",
            "optimize",
            *candidates,
            "--genome",
            "genome.fa",
            "--cores",
            "2",
            cwd=cls.tmp,
        )
        cls.scores = pd.read_csv(cls.tmp / "opt_bandwidth_scores.tsv", sep="\t", index_col=[0, 1])
        cls.chosen = int(cls.scores.pas.idxmax()[1])
        utrcalling("call-peaks", "placed.h5ad", "direct", "-b", cls.chosen, cwd=cls.tmp)
        # Each kernel with each candidate, chosen among all.
        utrcalling(
            "call-peaks",
            "placed.h5ad",
            "kernels",
            "-b",
            "optimize",
            *candidates,
            "--kde-kernel",
            "gaussian",
            "linear",
            "--genome",
            "genome.fa",
            cwd=cls.tmp,
        )
        cls.kernel_scores = pd.read_csv(
            cls.tmp / "kernels_bandwidth_scores.tsv", sep="\t", index_col=[0, 1]
        )

        # count, with the scores given: the real genome is too big for a test.
        cls.mocked_scores = pd.DataFrame(
            {"pas": [0.1, 0.3], "sites": [9, 4], "sites_per_utr": [2.0, 1.0]},
            index=pd.MultiIndex.from_tuples(
                [("gaussian", 12), ("gaussian", 60)], names=["kernel", "bandwidth"]
            ),
        )
        intermediate = WORKDIR / "intermediate"
        count = [
            "count",
            intermediate / "utr_reference_regions.parquet",
            "-c",
            intermediate / "UTR_sizes_counts_1.parquet",
            intermediate / "UTR_sizes_counts_2.parquet",
            *PIPELINE_FLAGS,
            "-b",
            "optimize",
            "--candidates",
            "12",
            "60",
            "--genome",
            cls.tmp / "genome.fa",
        ]
        with mock.patch(
            "utrcalling.core.peak_calling_anndata.score_bandwidths",
            side_effect=lambda *args, **kwargs: cls.mocked_scores.copy(),
        ) as scorer:
            utrcalling(*count, "-r", "-o", "count.h5ad", cwd=cls.tmp)
            utrcalling(*count, "-o", "peaks.h5ad", cwd=cls.tmp)
        cls.scorer_calls = scorer.call_args_list
        utrcalling("call-peaks", WORKDIR / "raw.h5ad", "fixed", "-b", "60", cwd=cls.tmp)

    def test_call_peaks_chooses_the_best_score(self):
        self.assertListEqual(
            self.scores.index.tolist(), [("gaussian", 4), ("gaussian", 12), ("gaussian", 60)]
        )
        self.assertEqual(self.scores.sites.is_monotonic_decreasing, True)
        optimized = ad.read_h5ad(self.tmp / f"opt_bw{self.chosen}.h5ad")
        assert_same_anndata(optimized, ad.read_h5ad(self.tmp / f"direct_bw{self.chosen}.h5ad"))
        info = provenance.recorded(optimized)
        self.assertEqual(info["bandwidth"], self.chosen)
        np.testing.assert_allclose(info["bandwidth_scores"].pas, self.scores.pas)

    def test_call_peaks_chooses_among_kernels(self):
        expected = [(kernel, bw) for kernel in ("gaussian", "linear") for bw in (4, 12, 60)]
        self.assertListEqual(self.kernel_scores.index.tolist(), expected)
        # The Gaussian rows are those scored alone.
        pd.testing.assert_frame_equal(self.kernel_scores.loc[["gaussian"]], self.scores)
        kernel, bandwidth = self.kernel_scores.pas.idxmax()
        label = f"bw{bandwidth}" if kernel == "gaussian" else f"{kernel}_bw{bandwidth}"
        chosen = ad.read_h5ad(self.tmp / f"kernels_{label}.h5ad")
        info = provenance.recorded(chosen)
        self.assertEqual((info["kernel"], info["bandwidth"]), (kernel, bandwidth))
        self.assertListEqual(info["bandwidth_scores"].kernel.tolist(), [k for k, _ in expected])

    def test_count_calls_the_peaks_of_the_best_bandwidth(self):
        fixed = ad.read_h5ad(self.tmp / "fixed_bw60.h5ad")
        for name in ("count_kde.h5ad", "peaks.h5ad"):
            optimized = ad.read_h5ad(self.tmp / name)
            with self.subTest(output=name):
                assert_same_anndata(fixed[optimized.obs_names], optimized)
                info = provenance.recorded(optimized)
                self.assertEqual((info["bandwidth"], info["kernel"]), (60, "gaussian"))
                self.assertEqual(info["peak_caller"], "kde")
                self.assertListEqual(info["bandwidth_scores"].bandwidth.tolist(), ["12", "60"])
        # With -r, the raw sizes too; both runs summarise the mapping.
        assert_same_anndata(
            ad.read_h5ad(self.tmp / "count.h5ad"), ad.read_h5ad(WORKDIR / "raw.h5ad")
        )
        for stem in ("count", "peaks"):
            summary = read_summary(self.tmp / f"{stem}_mapping_summary.tsv").loc["total"]
            self.assertEqual(summary.molecules_counted, fixed.X.sum())

    def test_count_scores_the_raw_sizes_of_its_utr_end(self):
        for call in self.scorer_calls:
            raw_sizes, genome, candidates = call.args
            self.assertEqual(raw_sizes.X.sum(), ad.read_h5ad(WORKDIR / "raw.h5ad").X.sum())
            self.assertEqual(genome, str(self.tmp / "genome.fa"))
            self.assertEqual(candidates, (12, 60))
            self.assertEqual(
                (call.kwargs["kernel"], call.kwargs["utr_end"]), (("gaussian",), "three_prime")
            )
            self.assertIsNone(call.kwargs["scores"])  # the UTR end's
        written = pd.read_csv(self.tmp / "count_bandwidth_scores.tsv", sep="\t", index_col=[0, 1])
        pd.testing.assert_frame_equal(written, self.mocked_scores)


class TestSiteProcessingCommands(_InTempDir):
    """The paper's steps after peak calling, each on the output of the previous one."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        utrcalling("call-peaks-different-bandwidths", WORKDIR / "raw.h5ad", "peaks", cwd=cls.tmp)
        # The paper's thresholds leave no site with alternative polyadenylation
        # in the small test data, so the chain continues from -m 2.
        utrcalling(
            "remove-peaks-below-threshold",
            "peaks_bw12.h5ad",
            "paper_thresholds.h5ad",
            "-t",
            "0.05",
            "-m",
            "10",
            cwd=cls.tmp,
        )
        commands = [
            [
                "remove-peaks-below-threshold",
                "peaks_bw12.h5ad",
                "kept.h5ad",
                "-t",
                "0.05",
                "-m",
                "2",
            ],
            ["select-alternative-transcription-events", "kept.h5ad", "apa.h5ad"],
            ["generate-names", "apa.h5ad", "named.h5ad", "-u", "utr_name", "-g", "gene_name"],
            ["counts-to-percentages", "named.h5ad", "fractions.h5ad", "-u", "utr_name"],
        ]
        for command in commands:
            utrcalling(*command, cwd=cls.tmp)
        # The UTRs with more molecules than the median: before and after peak calling.
        raw = ad.read_h5ad(WORKDIR / "raw.h5ad")
        cls.utr_molecules = (
            pd.Series(_dense(raw.X).sum(axis=0))
            .groupby(raw.var.utr_name.astype(str).to_numpy())
            .sum()
        )
        cls.minimum = int(cls.utr_molecules.median())
        deep = ["process", "filter-utrs", "-m", cls.minimum]
        utrcalling(*deep, WORKDIR / "raw.h5ad", "deep_raw.h5ad", cwd=cls.tmp)
        utrcalling("call-peaks", "deep_raw.h5ad", "deep", "-b", "12", cwd=cls.tmp)
        utrcalling(*deep, "peaks_bw12.h5ad", "deep_peaks.h5ad", cwd=cls.tmp)
        utrcalling(
            "process",
            "filter-sites",
            "peaks_bw12.h5ad",
            "by_batch.h5ad",
            "-t",
            "0.3",
            "-m",
            "3",
            "-g",
            "Batch",
            cwd=cls.tmp,
        )
        cls.peaks = ad.read_h5ad(cls.tmp / "peaks_bw12.h5ad")
        cls.kept = ad.read_h5ad(cls.tmp / "kept.h5ad")
        cls.apa = ad.read_h5ad(cls.tmp / "apa.h5ad")
        cls.named = ad.read_h5ad(cls.tmp / "named.h5ad")
        cls.fractions = ad.read_h5ad(cls.tmp / "fractions.h5ad")

    def _kept_by_threshold(self, threshold, minimum_counts):
        site_counts = pd.Series(_dense(self.peaks.X).sum(axis=0), index=self.peaks.var_names)
        utr_counts = site_counts.groupby(self.peaks.var.utr_name.astype(str).values).transform(
            "sum"
        )
        return site_counts >= np.maximum(utr_counts * threshold, minimum_counts)

    def test_threshold_keeps_the_sites_with_enough_of_their_utr_counts(self):
        # A site stays if it has at least max(5% of its UTR's molecules, 10).
        paper = ad.read_h5ad(self.tmp / "paper_thresholds.h5ad")
        keep = self._kept_by_threshold(0.05, 10)
        self.assertListEqual(list(paper.var_names), list(keep[keep].index))
        self.assertListEqual(
            list(self.kept.var_names), list(self._kept_by_threshold(0.05, 2)[lambda s: s].index)
        )
        np.testing.assert_array_equal(_dense(paper.X), _dense(self.peaks[:, paper.var_names].X))

    def test_filter_utrs_keeps_the_same_utrs_before_and_after_peak_calling(self):
        deep = set(self.utr_molecules[self.utr_molecules >= self.minimum].index)
        before = ad.read_h5ad(self.tmp / "deep_bw12.h5ad")
        after = ad.read_h5ad(self.tmp / "deep_peaks.h5ad")
        self.assertSetEqual(set(after.var.utr_name.astype(str)), deep)
        self.assertLess(len(deep), len(self.utr_molecules))
        # The same sites and counts; names number each gene's sites anew.
        self.assertListEqual(_site_keys(before), _site_keys(after))
        np.testing.assert_array_equal(_dense(before.X), _dense(after.X))

    def test_grouped_thresholds_keep_the_sites_one_batch_uses(self):
        by_batch = ad.read_h5ad(self.tmp / "by_batch.h5ad")
        counts = _dense(self.peaks.X)
        utrs = self.peaks.var.utr_name.astype(str).values
        keep = np.zeros(self.peaks.n_vars, dtype=bool)
        for batch in self.peaks.obs.Batch.unique():
            sites = pd.Series(counts[(self.peaks.obs.Batch == batch).to_numpy()].sum(axis=0))
            utr_counts = sites.groupby(utrs).transform("sum")
            keep |= (sites >= np.maximum(utr_counts * 0.3, 3)).to_numpy()
        self.assertListEqual(list(by_batch.var_names), list(self.peaks.var_names[keep]))
        # Not a superset of the sites that all cells together keep: the minimum
        # count holds in each batch, whose counts are fewer.
        together = self._kept_by_threshold(0.3, 3)
        self.assertTrue((keep & ~together.to_numpy()).any())
        self.assertTrue((together.to_numpy() & ~keep).any())

    def test_apa_selection_keeps_exactly_the_utrs_with_two_or_more_sites(self):
        sites_per_utr = self.kept.var.groupby("utr_name", observed=True).size()
        expected = set(sites_per_utr[sites_per_utr >= 2].index)
        self.assertGreater(len(expected), 0)
        self.assertSetEqual(set(self.apa.var.utr_name.astype(str)), expected)
        np.testing.assert_array_equal(
            _dense(self.apa.X), _dense(self.kept[:, self.apa.var_names].X)
        )

    def test_generated_names_number_each_genes_sites_and_add_their_size(self):
        # <gene>_<k>-<size>, k = 1, 2, ... within each gene, in the order of the sites.
        var = self.apa.var
        number = var.groupby("gene_name", observed=True).cumcount() + 1
        expected = (
            var.gene_name.astype(str) + "_" + number.astype(str) + "-" + var.utr_size.astype(str)
        )
        self.assertListEqual(list(self.named.var_names), expected.tolist())
        self.assertListEqual(self.named.var.UTR_site.tolist(), list(self.apa.var_names))
        np.testing.assert_array_equal(_dense(self.named.X), _dense(self.apa.X))

    def test_generate_names_overwrites_its_output(self):
        # Like every other command. It used to write named_1.h5ad instead,
        # leaving the old named.h5ad for the next step to read.
        (self.tmp / "named.h5ad").write_text("stale")
        utrcalling("generate-names", "apa.h5ad", "named.h5ad", cwd=self.tmp)
        assert_same_anndata(ad.read_h5ad(self.tmp / "named.h5ad"), self.named)
        self.assertFalse((self.tmp / "named_1.h5ad").exists())

    def test_each_step_adds_its_command_to_the_record(self):
        steps = ["run", "call-peaks-different-bandwidths", "remove-peaks-below-threshold"]
        steps += ["select-alternative-transcription-events", "generate-names"]
        self.assertListEqual(commands_run(self.named), steps)
        self.assertEqual(provenance.recorded(self.named)["matrix"], "peaks")
        fractions = provenance.recorded(self.fractions)
        self.assertListEqual(commands_run(self.fractions), [*steps, "counts-to-percentages"])
        self.assertEqual(fractions["matrix"], "fractions")
        self.assertEqual(fractions["bandwidth"], 12)

    def test_dense_input_gives_dense_fractions(self):
        dense = self.named.copy()
        dense.X = _dense(dense.X)
        dense.write_h5ad(self.tmp / "named_dense.h5ad")
        utrcalling("counts-to-percentages", "named_dense.h5ad", "dense.h5ad", cwd=self.tmp)
        fractions = ad.read_h5ad(self.tmp / "dense.h5ad").X
        self.assertFalse(scipy.sparse.issparse(fractions))
        np.testing.assert_allclose(fractions, _dense(self.fractions.X))

    def test_fractions_sum_to_one_per_cell_and_utr(self):
        self.assertTrue(scipy.sparse.issparse(self.fractions.X))
        fractions = _dense(self.fractions.X)
        counts = _dense(self.named.X)
        for utr in self.named.var.utr_name.unique():
            columns = (self.named.var.utr_name == utr).to_numpy()
            has_counts = counts[:, columns].sum(axis=1) > 0
            sums = fractions[:, columns].sum(axis=1)
            np.testing.assert_allclose(sums[has_counts], 1.0)
            np.testing.assert_array_equal(sums[~has_counts], 0.0)


class TestComputeMeanUtrSizesCommand(_InTempDir):
    """`compute-mean-utr-sizes` on the raw sizes of `run` (the paper's blood workflow)."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        utrcalling(
            "compute-mean-utr-sizes",
            WORKDIR / "raw.h5ad",
            "means.h5ad",
            "-u",
            "utr_name",
            cwd=cls.tmp,
        )
        cls.raw = ad.read_h5ad(WORKDIR / "raw.h5ad")
        cls.means = ad.read_h5ad(cls.tmp / "means.h5ad")

    def test_records_its_mean_sizes(self):
        self.assertEqual(provenance.recorded(self.means)["matrix"], "mean_sizes")
        self.assertListEqual(commands_run(self.means), ["run", "compute-mean-utr-sizes"])

    def test_one_column_per_utr_and_the_same_cells(self):
        self.assertListEqual(sorted(self.means.var_names), sorted(self.raw.var.utr_name.unique()))
        pd.testing.assert_index_equal(self.means.obs_names, self.raw.obs_names)

    def test_unsorted_utrs_keep_their_annotations(self):
        # The means come out sorted by UTR: their annotations must follow.
        shuffled = self.raw[:, np.random.default_rng(0).permutation(self.raw.n_vars)]
        shuffled.write_h5ad(self.tmp / "shuffled.h5ad")
        utrcalling("compute-mean-utr-sizes", "shuffled.h5ad", "shuffled_means.h5ad", cwd=self.tmp)
        means = ad.read_h5ad(self.tmp / "shuffled_means.h5ad")
        assert_same_anndata(means[:, self.means.var_names], self.means)

    def test_values_are_the_count_weighted_mean_size(self):
        counts = _dense(self.raw.X)
        sizes = self.raw.var.utr_size.to_numpy()
        for utr in self.means.var_names:
            columns = (self.raw.var.utr_name == utr).to_numpy()
            total = counts[:, columns].sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                expected = (counts[:, columns] * sizes[columns]).sum(axis=1) / total
            expected[total == 0] = np.nan
            np.testing.assert_allclose(_dense(self.means[:, utr].X).ravel(), expected)


class TestConcatenateExperimentsCommand(_InTempDir):
    """`concatenate-experiments` puts back together two experiments of one data set."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        raw = ad.read_h5ad(WORKDIR / "raw.h5ad")
        old = raw[raw.obs.CellAge == "old"]
        young = raw[raw.obs.CellAge == "young"]
        # Each experiment only has the sites it saw: the rest are filled with zeros.
        for name, part in (("old", old), ("young", young)):
            seen = _dense(part.X).sum(axis=0) > 0
            part[:, seen].copy().write_h5ad(cls.tmp / f"{name}.h5ad")
        utrcalling(
            "concatenate-experiments",
            "-i",
            "old.h5ad",
            "young.h5ad",
            "-o",
            "all.h5ad",
            cwd=cls.tmp,
        )
        cls.raw = raw
        cls.combined = ad.read_h5ad(cls.tmp / "all.h5ad")
        cls.expected_cells = list(old.obs_names) + list(young.obs_names)

    def test_continues_the_record_of_the_first_experiment(self):
        info = provenance.recorded(self.combined)
        self.assertEqual(info["matrix"], "raw_sizes")
        self.assertListEqual(commands_run(self.combined), ["run", "concatenate-experiments"])
        # Both experiments come from the same run here: its BAM files twice.
        run_summary = read_summary(WORKDIR / "raw_mapping_summary.tsv")
        bams = list(run_summary.index[:-1])
        self.assertListEqual(list(info["mapping_summary"].index), [*bams, *bams, "total"])
        pd.testing.assert_series_equal(
            info["mapping_summary"].loc["total"], run_summary.loc["total"] * 2
        )

    def test_has_the_cells_of_both_experiments(self):
        self.assertListEqual(list(self.combined.obs_names), self.expected_cells)

    def test_sites_are_named_gene_number_size(self):
        self.assertListEqual(list(self.combined.var_names), expected_site_names(self.combined.var))

    def test_counts_per_site_match_the_original_data(self):
        self.assertCountEqual(_site_keys(self.combined), _site_keys(self.raw))
        order = {key: i for i, key in enumerate(_site_keys(self.raw))}
        columns = [order[key] for key in _site_keys(self.combined)]
        np.testing.assert_array_equal(
            _dense(self.combined.X), _dense(self.raw[self.expected_cells].X)[:, columns]
        )


class TestQcCommands(_InTempDir):
    """The qc commands on the outputs of `run`, checked against those outputs."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.raw = ad.read_h5ad(WORKDIR / "raw.h5ad")
        cls.kde = ad.read_h5ad(WORKDIR / "raw_kde.h5ad")
        raw, kde = WORKDIR / "raw.h5ad", WORKDIR / "raw_kde.h5ad"
        utrcalling("qc", "report", raw, kde, "-o", "report_mqc.tsv", "--multiqc", cwd=cls.tmp)
        utrcalling("qc", "cells", kde, "cells.tsv", cwd=cls.tmp)
        utrcalling("qc", "batches", kde, "batches", "-g", "Batch", "-m", "5", cwd=cls.tmp)
        utrcalling("qc", "peaks", raw, "peaks.tsv", "-b", "12", "--cores", "2", cwd=cls.tmp)
        # Known sites 5 nt downstream of every other site of run's peaks.
        var = cls.kde.var
        plus = var.strand == "+"
        cls.positions = np.where(plus, var.start + var.utr_size, var.end - var.utr_size)
        chosen = np.arange(len(var)) % 2 == 0
        known = pd.DataFrame(
            {
                "chrom": "chr" + var.chrom.astype(str),
                "start": cls.positions + np.where(plus, 5, -5) - 1,
                "name": "known",
                "score": 0,
                "strand": var.strand.astype(str),
            }
        )[chosen]
        known.insert(2, "end", known.start + 1)
        known.to_csv(cls.tmp / "known.bed", sep="\t", header=False, index=False)
        cls.known = known
        utrcalling("qc", "known-sites", kde, "known.bed", "known.tsv", cwd=cls.tmp)
        cls._run_priming()

    @classmethod
    def _run_priming(cls):
        """qc priming on run's peaks, with their UTRs moved onto a small random genome.

        Every third site gets 7 A's in the genome after it.
        """
        placed, genome_size = place_utrs(cls.kde)
        placed.write_h5ad(cls.tmp / "placed.h5ad")
        var = placed.var
        plus = (var.strand == "+").to_numpy()
        positions = np.where(plus, var.start + var.utr_size, var.end - var.utr_size)
        genome = list(random_genome(genome_size, seed=0))
        for position, on_plus in zip(positions[::3], plus[::3], strict=True):
            start = position if on_plus else position - 8  # 0-based, after the site
            genome[start : start + 7] = "AAAAAAA" if on_plus else "TTTTTTT"
        cls.genome = "".join(genome)
        cls.placed_positions = positions
        (cls.tmp / "genome.fa").write_text(f">1\n{cls.genome}\n")
        utrcalling("qc", "priming", "placed.h5ad", "genome.fa", "priming.tsv", cwd=cls.tmp)
        utrcalling(
            "tools",
            "site-sequences",
            "placed.h5ad",
            "genome.fa",
            "sequences.tsv",
            "--upstream",
            "8",
            "--downstream",
            "5",
            cwd=cls.tmp,
        )
        sequences = ["tools", "site-sequences", "placed.h5ad", "genome.fa"]
        utrcalling(*sequences, "utrs.tsv", "--upstream", "utr", "--downstream", "UTR", cwd=cls.tmp)
        utrcalling(*sequences, "isoforms.fa", "--upstream", "utr", "--downstream", "0", cwd=cls.tmp)

    def _table(self, name, index_col):
        return pd.read_csv(self.tmp / name, sep="\t", comment="#", index_col=index_col)

    def test_report(self):
        report = self._table("report_mqc.tsv", "file")
        self.assertListEqual(list(report.index), ["raw.h5ad", "raw_kde.h5ad"])
        self.assertListEqual(report.matrix.tolist(), ["raw_sizes", "peaks"])
        self.assertListEqual(report.sites.tolist(), [self.raw.n_vars, self.kde.n_vars])
        self.assertListEqual(report.molecules.tolist(), [self.raw.X.sum()] * 2)
        summary = read_summary(WORKDIR / "raw_mapping_summary.tsv").loc["total"]
        self.assertListEqual(report.mapping_alignments.tolist(), [summary.alignments] * 2)
        self.assertEqual(report.loc["raw_kde.h5ad", "bandwidth"], 12)
        self.assertEqual(report.loc["raw_kde.h5ad", "kernel"], "gaussian")
        self.assertTrue(pd.isna(report.loc["raw.h5ad", "bandwidth"]))
        self.assertTrue(pd.isna(report.loc["raw.h5ad", "kernel"]))

    def test_cells(self):
        cells = self._table("cells.tsv", "cell")
        self.assertListEqual(list(cells.index), list(self.kde.obs_names))
        np.testing.assert_array_equal(cells.molecules, np.asarray(self.kde.X.sum(axis=1)).ravel())
        sites = np.asarray((self.kde.X > 0).sum(axis=1)).ravel()
        np.testing.assert_array_equal(cells.sites, sites)

    def test_batches(self):
        groups = self._table("batches_groups.tsv", "group")
        batches = self.kde.obs.Batch.astype(str).value_counts()
        self.assertListEqual(groups.cells.tolist(), [batches["1"], batches["2"]])
        self.assertEqual(groups.molecules.sum(), self.kde.X.sum())
        pairs = self._table("batches_pairs.tsv", ["group_1", "group_2"])
        self.assertListEqual(list(pairs.index), [(1, 2)])
        self.assertGreater(pairs.utrs.iloc[0], 0)

    def test_peaks_are_those_of_run(self):
        peaks = self._table("peaks.tsv", "site")
        self.assertCountEqual(peaks.index, self.kde.var_names)
        molecules = pd.Series(np.asarray(self.kde.X.sum(axis=0)).ravel(), self.kde.var_names)
        pd.testing.assert_series_equal(
            peaks.molecules, molecules[peaks.index], check_names=False, check_index_type=False
        )
        shares = peaks.groupby("utr_name").utr_share.sum()
        np.testing.assert_allclose(shares, 1.0)

    def test_known_sites(self):
        known = self._table("known.tsv", "site")
        self.assertListEqual(list(known.index), list(self.kde.var_names))
        np.testing.assert_array_equal(known.position, self.positions)
        # Brute force: each site's distance to the known sites of its chromosome and strand.
        for site, row in known.iterrows():
            same = self.known[
                (self.known.chrom == f"chr{row.chrom}") & (self.known.strand == row.strand)
            ]
            with self.subTest(site=site):
                if same.empty:  # no known site on this chromosome and strand
                    self.assertTrue(pd.isna(row.known_site_distance))
                    self.assertFalse(row.near_known_site)
                    continue
                distance = np.abs(same.start.to_numpy() - (row.position - 1)).min()
                self.assertEqual(row.known_site_distance, distance)
                self.assertEqual(row.near_known_site, distance <= 20)
        self.assertGreater(known.near_known_site.sum(), len(known) / 2 - 1)

    def test_site_sequences(self):
        table = self._table("sequences.tsv", "site")
        np.testing.assert_array_equal(table.position, self.placed_positions)
        complement = str.maketrans("ACGT", "TGCA")
        for site, row in table.iterrows():
            p = row.position  # the last transcribed base: -1
            if row.strand == "+":
                upstream, downstream = self.genome[p - 8 : p], self.genome[p : p + 5]
            else:  # read leftwards, complemented
                upstream = self.genome[p - 1 : p + 7].translate(complement)[::-1]
                downstream = self.genome[p - 6 : p - 1].translate(complement)[::-1]
            with self.subTest(site=site):
                self.assertEqual((row.upstream, row.downstream), (upstream, downstream))

    def test_site_sequences_to_the_ends_of_the_utr(self):
        table = self._table("utrs.tsv", "site").fillna("")
        var = ad.read_h5ad(self.tmp / "placed.h5ad").var
        complement = str.maketrans("ACGT", "TGCA")
        for site, row in table.iterrows():
            start, end, p = var.start[site], var.end[site], row.position
            if row.strand == "+":  # the UTR's start up to the site (-1), then on to its end
                upstream, downstream = self.genome[start - 1 : p], self.genome[p:end]
            else:
                upstream = self.genome[p - 1 : end].translate(complement)[::-1]
                downstream = self.genome[start - 1 : p - 1].translate(complement)[::-1]
            with self.subTest(site=site):
                self.assertEqual((row.upstream, row.downstream), (upstream, downstream))
                # Together, the whole UTR, for sites within it; a site past its
                # annotated end makes a longer UTR, and reads up to the site.
                if start <= p <= end:
                    self.assertEqual(len(row.upstream + row.downstream), end - start + 1)
                elif (p > end) == (row.strand == "+"):  # past the UTR's 3' end
                    self.assertEqual(row.downstream, "")
        positions = table.position.to_numpy()
        inside = (var.start.to_numpy() <= positions) & (positions <= var.end.to_numpy())
        self.assertGreater(inside.mean(), 0.9)

    def test_the_utr_isoform_of_each_site_as_fasta(self):
        table = self._table("utrs.tsv", "site").fillna("")
        records = (self.tmp / "isoforms.fa").read_text().splitlines()
        headers, sequences = records[::2], records[1::2]
        self.assertListEqual([header.split()[0][1:] for header in headers], list(table.index))
        self.assertListEqual(sequences, table.upstream.tolist())
        for header, sequence in zip(headers, sequences, strict=True):
            self.assertTrue(header.endswith(f"upstream={len(sequence)} downstream=0"))

    def test_priming(self):
        priming = self._table("priming.tsv", "site")
        self.assertListEqual(list(priming.index), list(self.kde.var_names))
        np.testing.assert_array_equal(priming.position, self.placed_positions)
        complement = str.maketrans("ACGT", "TGCA")
        # Brute force: the 20 nt after each site, read on its strand.
        for site, row in priming.iterrows():
            if row.strand == "+":
                downstream = self.genome[row.position : row.position + 20]
            else:
                downstream = self.genome[row.position - 21 : row.position - 1]
                downstream = downstream.translate(complement)[::-1]
            longest = max(map(len, re.findall("A+", downstream)), default=0)
            most = max(downstream[i : i + 10].count("A") for i in range(11))
            with self.subTest(site=site):
                self.assertEqual(row.downstream, downstream)
                self.assertEqual(row.longest_a_run, longest)
                self.assertEqual(row.most_a_in_10_nt, most)
                self.assertEqual(row.internal_priming, longest >= 6 or most >= 7)
        self.assertTrue(priming.internal_priming.iloc[::3].all())  # the planted A's


if __name__ == "__main__":
    unittest.main()
