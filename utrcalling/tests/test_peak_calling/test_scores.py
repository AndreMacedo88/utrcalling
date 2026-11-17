"""Tests for utrcalling.core.peak_calling.scores and peak_calling_anndata.score_bandwidths."""

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
from utrcalling.core import peak_calling_anndata as pca
from utrcalling.core import provenance
from utrcalling.core.peak_calling import scores
from utrcalling.core.peak_calling.scores import ScoreContext, SiteScore
from utrcalling.core.peak_calling.settings import CANDIDATE_BANDWIDTHS, SCORES
from utrcalling.tests.test_peak_calling_anndata import _make_adata


def _tmpdir(test):
    path = Path(tempfile.mkdtemp())
    test.addCleanup(shutil.rmtree, path)
    return path


def _bandwidths(table):
    return table.index.get_level_values("bandwidth").tolist()


class TestPasScore(unittest.TestCase):
    """pas: estimate_best_bandwidth's scores, with the sites each bandwidth makes."""

    def test_scores_and_sites_per_bandwidth(self):
        genome = _tmpdir(self) / "genome.fa"
        genome.write_text(">chr1\n" + "C" * 119 + "AATAAA" + "C" * 175 + "\n")
        # Two UTRs, each with sizes 40-42 and 90-92, on the same coordinates.
        adata = _make_adata(
            count_matrix=[[2, 5, 2, 3, 3, 3] * 2] * 4,
            utr_names=["UTR0"] * 6 + ["UTR1"] * 6,
            utr_sizes=[40, 41, 42, 90, 91, 92] * 2,
            n_obs=4,
        )
        bandwidths = [1, 60]
        with mock.patch.object(pca, "_find_pas", wraps=pca._find_pas) as find_pas:
            table = pca.score_bandwidths(adata, genome=str(genome), bandwidths=bandwidths, n_jobs=1)
        # The hexamers of each UTR are found once, whatever the bandwidths.
        self.assertEqual(find_pas.call_count, 2)

        self.assertListEqual(table.index.names, ["kernel", "bandwidth"])
        self.assertListEqual(table.columns.tolist(), ["pas", "sites", "sites_per_utr"])
        self.assertListEqual(_bandwidths(table), bandwidths)
        self.assertListEqual(table.index.get_level_values("kernel").tolist(), ["gaussian"] * 2)
        expected = pca.estimate_best_bandwidth(
            adata, genome=str(genome), hamming=1, bandwidths=bandwidths, n_jobs=1
        )
        np.testing.assert_allclose(table.pas.to_numpy(), expected)
        sites = [
            pca.peak_call_anndata_parallel(adata, "kde", bandwidth=bw, n_jobs=1).n_vars
            for bw in bandwidths
        ]
        self.assertListEqual(table.sites.tolist(), sites)
        self.assertListEqual(table.sites_per_utr.tolist(), [s / 2 for s in sites])
        # A narrow bandwidth keeps the two groups of sizes of each UTR apart.
        self.assertListEqual(sites, [4, 2])

    def test_chromosomes_match_with_or_without_chr(self):
        tmpdir = _tmpdir(self)
        sequence = "C" * 119 + "AATAAA" + "C" * 175
        (tmpdir / "chr.fa").write_text(f">chr1\n{sequence}\n")
        (tmpdir / "plain.fa").write_text(f">1\n{sequence}\n>3\nACGT\n")
        adata = _make_adata(
            count_matrix=[[2, 5, 2, 3, 3, 3, 4] * 1] * 4,
            utr_names=["UTR0"] * 6 + ["UTR1"],
            utr_sizes=[40, 41, 42, 90, 91, 92, 10],
            n_obs=4,
        )
        adata.var["chrom"] = ["chr1"] * 6 + ["chr2"]  # the genomes lack chromosome 2
        tables = {
            name: pca.score_bandwidths(adata, genome=str(tmpdir / name), bandwidths=[1], n_jobs=1)
            for name in ("chr.fa", "plain.fa")
        }
        self.assertEqual(tables["chr.fa"].pas.iloc[0], tables["plain.fa"].pas.iloc[0])
        # Only UTR0 counts: as if UTR1 were not in the matrix.
        expected = pca.estimate_best_bandwidth(
            adata[:, :6].copy(), genome=str(tmpdir / "chr.fa"), bandwidths=[1], n_jobs=1
        )
        self.assertEqual(tables["chr.fa"].pas.iloc[0], expected[0])
        (tmpdir / "other.fa").write_text(">NC_000067.7\nACGT\n")
        with self.assertRaisesRegex(ValueError, "same genome assembly"):
            pca.score_bandwidths(adata, genome=str(tmpdir / "other.fa"), bandwidths=[1])

    def test_utrs_without_a_hexamer_score_0_but_count(self):
        genome = _tmpdir(self) / "genome.fa"
        genome.write_text(">chr1\n" + "C" * 119 + "AATAAA" + "C" * 375 + "\n")
        # UTR0 on 100-200 has the hexamer; UTR1 on 300-400 has none.
        adata = _make_adata(
            count_matrix=[[2, 5, 2, 3, 3, 3] * 2] * 4,
            utr_names=["UTR0"] * 6 + ["UTR1"] * 6,
            utr_sizes=[40, 41, 42, 90, 91, 92] * 2,
            n_obs=4,
        )
        adata.var.loc[adata.var.utr_name == "UTR1", ["start", "end"]] = [300, 400]
        table = pca.score_bandwidths(adata, genome=str(genome), bandwidths=[1], n_jobs=1)
        expected = pca.estimate_best_bandwidth(adata, genome=str(genome), bandwidths=[1], n_jobs=1)
        self.assertEqual(table.pas.iloc[0], expected[0])
        self.assertGreater(expected[0], 0)
        utr_scores = scores.make_score(
            "pas", adata, ScoreContext(utr_end="three_prime", genome=str(genome))
        ).utr_scores(pca.peak_call_anndata_parallel(adata, "kde", bandwidth=1, n_jobs=1))
        self.assertEqual(utr_scores["UTR1"], 0)
        self.assertEqual(utr_scores.mean(), expected[0])

    def test_raw_sizes_score_with_bandwidth_0(self):
        genome = _tmpdir(self) / "genome.fa"
        genome.write_text(">chr1\n" + "C" * 119 + "AATAAA" + "C" * 175 + "\n")
        adata = _make_adata(
            count_matrix=[[2, 5, 2]] * 4, utr_names=["UTR0"] * 3, utr_sizes=[40, 41, 42], n_obs=4
        )
        table = pca.score_bandwidths(adata, genome=str(genome), bandwidths=[0], n_jobs=1)
        self.assertEqual(table.sites.iloc[0], 3)  # the sizes as they are
        self.assertEqual(table.pas.iloc[0], 1)  # all 10-30 nt after the hexamer


class TestBestSetting(unittest.TestCase):
    def setUp(self):
        index = pd.MultiIndex.from_tuples(
            [("gaussian", 4), ("linear", "scott:0.5"), ("gaussian", 12), ("linear", 20)],
            names=["kernel", "bandwidth"],
        )
        self.scores = pd.DataFrame(
            {"pas": [0.2, 0.5, 0.5, 0.1], "known-sites": [0.9, 0.1, 0.1, 0.1]}, index=index
        )

    def test_the_highest_score_and_the_first_of_equal_ones(self):
        self.assertEqual(pca.best_setting(self.scores), ("linear", "scott:0.5"))
        self.assertEqual(pca.best_setting(self.scores.iloc[2:]), ("gaussian", 12))
        self.assertEqual(pca.best_setting(self.scores, "known-sites"), ("gaussian", 4))

    def test_bandwidths_of_a_table_read_from_text_come_back_as_numbers(self):
        path = _tmpdir(self) / "scores.tsv"
        self.scores.to_csv(path, sep="\t")
        table = pd.read_csv(path, sep="\t", index_col=[0, 1], dtype={"bandwidth": str})
        self.assertEqual(pca.best_setting(table), ("linear", "scott:0.5"))
        kernel, bandwidth = pca.best_setting(table.iloc[2:])
        self.assertEqual(kernel, "gaussian")
        self.assertIs(bandwidth, 12)

    def test_no_setting_scored(self):
        self.scores["pas"] = np.nan
        with self.assertRaisesRegex(ValueError, "No setting has a pas score"):
            pca.best_setting(self.scores)


class InitiatorData(unittest.TestCase):
    """5' sites in a small genome, with start sites at CA dinucleotides."""

    def setUp(self):
        self.tmp = _tmpdir(self)
        sequence = list("G" * 300)
        sequence[99:101] = "CA"  # -1 and +1 of a + strand start site at 101
        sequence[249:251] = "TG"  # read leftwards, CA: a - strand start site at 250
        self.genome = self.tmp / "genome.fa"
        self.genome.write_text(">chr1\n" + "".join(sequence) + "\n")
        # UTR P (+ strand, 1-200): 30 molecules start at 101 (CA), 10 at 151 (GG),
        # 1 at 51. UTR M (- strand, 201-300): 20 at 250 (CA). UTR X: chromosome 2.
        var = pd.DataFrame(
            {
                "utr_name": ["P", "P", "P", "M", "X"],
                "utr_size": [150, 100, 50, 50, 10],
                "chrom": ["1", "1", "1", "1", "2"],
                "start": [1, 1, 1, 201, 1],
                "end": [200, 200, 200, 300, 100],
                "strand": ["+", "+", "+", "-", "+"],
            },
            index=[f"site{i}" for i in range(5)],
        )
        # Sizes 150, 100 and 50 of P start at 51, 101 and 151 (end - size + 1).
        counts = np.array([[1, 30, 10, 20, 9]], dtype=np.int32)
        self.adata = ad.AnnData(X=scipy.sparse.csr_matrix(counts), var=var)

    def _table(self, bandwidths, kernel="linear", **kwargs):
        return pca.score_bandwidths(
            self.adata,
            genome=self.genome,
            bandwidths=bandwidths,
            kernel=kernel,
            utr_end="five_prime",
            n_jobs=1,
            **kwargs,
        )


class TestInitiatorScore(InitiatorData):
    """For 5' data, settings score by the initiator (a pyrimidine-purine pair) at their peaks."""

    def test_the_share_of_molecules_at_initiators(self):
        # P: 30 of 41 molecules (the site at 51 has 1, not more than 1%); M: all.
        table = self._table([1])
        self.assertListEqual(table.columns.tolist(), ["initiator", "sites", "sites_per_utr"])
        self.assertAlmostEqual(table.initiator.iloc[0], (30 / 41 + 1) / 2)

    def test_smoothing_off_the_start_sites_scores_lower(self):
        # A wide Gaussian joins P's sites between 101 and 151: no initiator there.
        narrow, wide = self._table([1, 60], kernel="gaussian").initiator
        self.assertGreater(narrow, wide)
        self.assertAlmostEqual(wide, (0 + 1) / 2)

    def test_the_utr_end_comes_from_the_record(self):
        with mock.patch.object(sys, "argv", ["utrcalling", "run"]):
            provenance.record(self.adata, utr_end="five_prime", summarization_level="utr")
        table = pca.score_bandwidths(
            self.adata, genome=str(self.genome), bandwidths=["1"], n_jobs=1
        )
        self.assertListEqual(_bandwidths(table), [1])
        self.assertAlmostEqual(table.initiator.iloc[0], (30 / 41 + 1) / 2)

    def test_default_candidates_per_utr_end(self):
        table = pca.score_bandwidths(self.adata, genome=self.genome, utr_end="five_prime", n_jobs=1)
        self.assertListEqual(_bandwidths(table), list(CANDIDATE_BANDWIDTHS["five_prime"]))

    def test_refusals(self):
        with self.assertRaisesRegex(ValueError, "utr_end 'middle'"):
            pca.score_bandwidths(self.adata, genome=self.genome, bandwidths=[1], utr_end="middle")
        with self.assertRaisesRegex(ValueError, "The pas score scores 3' sites, not the 5' sites"):
            self._table([1], scores=["pas"])
        with mock.patch.object(sys, "argv", ["utrcalling", "run"]):
            provenance.record(self.adata, summarization_level="gene")
        with self.assertRaisesRegex(ValueError, "Gene-level sites"):
            self._table([1])


class TestSeveralKernels(InitiatorData):
    """Settings are kernels and bandwidths: each bandwidth is scored with each kernel."""

    def test_each_kernel_with_each_bandwidth(self):
        table = self._table([1, 60], kernel=["linear", "gaussian"])
        self.assertListEqual(
            table.index.tolist(),
            [("linear", 1), ("linear", 60), ("gaussian", 1), ("gaussian", 60)],
        )
        for kernel in ("linear", "gaussian"):
            with self.subTest(kernel=kernel):
                alone = self._table([1, 60], kernel=kernel)
                pd.testing.assert_frame_equal(table.loc[[kernel]], alone)

    def test_unknown_kernels_are_refused(self):
        with self.assertRaisesRegex(ValueError, "Unknown kernel 'box'"):
            self._table([1], kernel=["linear", "box"])


class TestKnownSitesScore(InitiatorData):
    """known-sites: the share of molecules at sites near the known sites of a database."""

    def setUp(self):
        super().setUp()
        # Known start sites at 101 (+) and 250 (-), as BED (0-based starts).
        self.known = self.tmp / "known.bed"
        self.known.write_text("chr1\t100\t101\ttss_p\t0\t+\nchr1\t249\t250\ttss_m\t0\t-\n")

    def _known_table(self, bandwidths, **kwargs):
        return pca.score_bandwidths(
            self.adata,
            bandwidths=bandwidths,
            kernel="linear",
            utr_end="five_prime",
            scores=["known-sites"],
            known_sites=self.known,
            n_jobs=1,
            **kwargs,
        )

    def test_the_share_of_molecules_near_known_sites(self):
        # No genome needed. P: 30 of 41 molecules at 101; M: all at 250. X, on
        # a chromosome without known sites, is left out.
        table = self._known_table([1])
        self.assertListEqual(table.columns.tolist(), ["known-sites", "sites", "sites_per_utr"])
        self.assertAlmostEqual(table["known-sites"].iloc[0], (30 / 41 + 1) / 2)

    def test_the_window(self):
        # P's site at 151 lies 50 nt from the known site at 101.
        self.assertAlmostEqual(
            self._known_table([1], window=50)["known-sites"].iloc[0], (40 / 41 + 1) / 2
        )

    def test_a_known_site_on_the_other_strand_does_not_count(self):
        self.known.write_text("chr1\t100\t101\ttss_p\t0\t-\nchr1\t249\t250\ttss_m\t0\t-\n")
        utr_scores = scores.make_score(
            "known-sites",
            self.adata,
            ScoreContext(utr_end="five_prime", known_sites=str(self.known)),
        ).utr_scores(self.adata)
        # P has no known site on its strand: left out, as X is.
        self.assertListEqual(utr_scores.dropna().index.tolist(), ["M"])

    def test_3_prime_sites(self):
        # The same UTRs read as 3' data: sizes count from the UTR's start, so
        # P's sizes 150, 100 and 50 end at 151, 101 and 51.
        table = pca.score_bandwidths(
            self.adata,
            bandwidths=[1],
            kernel="linear",
            utr_end="three_prime",
            scores=["known-sites"],
            known_sites=self.known,
            n_jobs=1,
        )
        # P: 30 of 41 at 101. M (- strand, 201-300): size 50 ends at 250.
        self.assertAlmostEqual(table["known-sites"].iloc[0], (30 / 41 + 1) / 2)

    def test_with_the_initiator(self):
        table = self._table(
            [1, 60],
            kernel="gaussian",
            scores=["initiator", "known-sites"],
            known_sites=self.known,
            window=2,
        )
        self.assertListEqual(
            table.columns.tolist(), ["initiator", "known-sites", "sites", "sites_per_utr"]
        )
        # A wide Gaussian moves P's peak off its start site, 10 nt from the known one.
        self.assertAlmostEqual(table["known-sites"].iloc[0], (30 / 41 + 1) / 2)
        self.assertAlmostEqual(table["known-sites"].iloc[1], (0 + 1) / 2)
        self.assertEqual(pca.best_setting(table), ("gaussian", 1))

    def test_another_assembly_is_refused(self):
        self.known.write_text("NC_000067.7\t100\t101\ttss\t0\t+\n")
        with self.assertRaisesRegex(ValueError, "same genome assembly"):
            self._known_table([1])


class TestScoreClass(unittest.TestCase):
    """score_class checks a score against what the command line gives, before any work."""

    def test_what_a_score_needs(self):
        tmp = _tmpdir(self)
        three_prime = ScoreContext(utr_end="three_prime")
        with self.assertRaisesRegex(
            ValueError,
            "The pas score needs the genome sequence \\(FASTA\\) of the annotation: give it "
            "with --genome",
        ):
            scores.score_class("pas", three_prime)
        with self.assertRaisesRegex(FileNotFoundError, "--known-sites missing.bed: no such file"):
            scores.score_class(
                "known-sites", ScoreContext(utr_end="five_prime", known_sites="missing.bed")
            )
        (tmp / "genome.fa").write_text(">1\nA\n")
        context = ScoreContext(utr_end="five_prime", genome=str(tmp / "genome.fa"))
        self.assertIs(scores.score_class("Initiator", context), SCORES.get("initiator"))

    def test_unknown_scores(self):
        with self.assertRaisesRegex(ValueError, "Unknown score 'best': expected one of pas"):
            scores.score_class("best", ScoreContext(utr_end="three_prime"))

    def test_a_score_must_be_a_site_score(self):
        SCORES.register("not-a-score", len)
        self.addCleanup(SCORES.unregister, "not-a-score")
        with self.assertRaisesRegex(TypeError, "'not-a-score' must be a subclass of SiteScore"):
            scores.score_class("not-a-score", ScoreContext(utr_end="three_prime"))


class TestShareAtFrequentSites(unittest.TestCase):
    def test_shares(self):
        # A: 151 molecules; sites with more than ceil(1.51) = 2 are frequent.
        shares = scores.share_at_frequent_sites(
            utrs=["A", "A", "A", "B", "C"],
            molecules=[100, 2, 49, 0, 5],
            hits=[True, True, False, True, True],
        )
        self.assertEqual(shares.index.name, "utr_name")
        self.assertListEqual(shares.index.tolist(), ["A", "B", "C"])
        self.assertEqual(shares["A"], 100 / 151)
        self.assertTrue(np.isnan(shares["B"]))  # no molecules: no score
        self.assertEqual(shares["C"], 1)


class FewerSitesScore(SiteScore):
    """A score for the tests: higher for UTRs with fewer sites."""

    instances = 0

    def __init__(self, raw_sizes, context):
        super().__init__(raw_sizes, context)
        type(self).instances += 1

    def utr_scores(self, sites):
        return 1 / sites.var.groupby("utr_name", observed=True).size()


class TestAnAddedScore(InitiatorData):
    """A score added by name works wherever settings are scored."""

    def setUp(self):
        super().setUp()
        SCORES.register("fewer-sites", FewerSitesScore)
        self.addCleanup(SCORES.unregister, "fewer-sites")
        FewerSitesScore.instances = 0

    def test_score_bandwidths(self):
        table = self._table([1, 60], scores=["Fewer-Sites"])
        self.assertEqual(FewerSitesScore.instances, 1)  # once, whatever the settings
        # Bandwidth 1 keeps P's three sites: P 1/3, M 1, X 1; 60 joins them.
        np.testing.assert_allclose(table["fewer-sites"], [(1 / 3 + 1 + 1) / 3, 1])
        self.assertEqual(pca.best_setting(table), ("linear", 60))

    def test_qc_score_bandwidths(self):
        self.adata.write_h5ad(self.tmp / "raw.h5ad")
        out = self.tmp / "scores.tsv"
        argv = ["utrcalling", "qc", "score-bandwidths", str(self.tmp / "raw.h5ad"), str(out)]
        argv += ["-b", "1", "60", "--kde-kernel", "linear", "--score", "fewer-sites"]
        argv += ["--utr-end", "five_prime"]
        with mock.patch.object(sys, "argv", argv):
            runtime.main()
        table = pd.read_csv(out, sep="\t")
        self.assertListEqual(
            table.columns.tolist(), ["kernel", "bandwidth", "fewer-sites", "sites", "sites_per_utr"]
        )
        np.testing.assert_allclose(table["fewer-sites"], [(1 / 3 + 1 + 1) / 3, 1])


if __name__ == "__main__":
    unittest.main()
