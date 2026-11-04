"""The command chain of the paper workflows, run end to end from the command line.

Each test runs `utrcalling <command> <flags>` in this process, with the flags the
Snakemake workflows of Macedo et al. 2026 use (here on the 10x v2 mouse test data,
at UTR level). These command lines must keep working when the CLI changes.
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
    previous = os.getcwd()
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


if __name__ == "__main__":
    unittest.main()
