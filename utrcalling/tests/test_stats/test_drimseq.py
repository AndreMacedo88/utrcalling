"""Tests for utrcalling.core.stats.drimseq: the bridge to DRIMSeq in R."""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from utrcalling.core.stats import drimseq
from utrcalling.tests.test_stats import simulate


def _inputs():
    rng = np.random.default_rng(0)
    counts = [
        rng.poisson(20, size=(8, 2)),
        rng.poisson(20, size=(8, 3)),
        rng.poisson(20, size=(8, 2)),
    ]
    full = np.column_stack([np.ones(8), np.repeat([0.0, 1.0], 4)])
    return (
        ["UTR_a", "UTR_b", "UTR_c"],
        [["a1", "a2"], ["b1", "b2", "b3"], ["c1", "c2"]],
        counts,
        full,
        full[:, :1],
    )


class TestCheck(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)

    def test_returns_drimseqs_version(self):
        self.assertEqual(drimseq.check(simulate.fake_rscript(self.tmp)), "1.34.0")

    def test_missing_r_says_how_to_install_it(self):
        with self.assertRaisesRegex(RuntimeError, "conda install .* bioconductor-drimseq"):
            drimseq.check(str(self.tmp / "no_such_Rscript"))

    def test_r_without_drimseq(self):
        with self.assertRaisesRegex(RuntimeError, "could not load DRIMSeq"):
            drimseq.check(simulate.fake_rscript(self.tmp, mode="no_drimseq"))


class TestRun(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)

    def test_results_come_back_by_utr_and_site(self):
        utrs, sites, counts, full, reduced = _inputs()
        found = drimseq.run(
            utrs,
            sites,
            counts,
            full,
            reduced,
            moderation="common",
            cores=2,
            seed=7,
            rscript=simulate.fake_rscript(self.tmp),
        )
        self.assertListEqual(list(found.genes.index), utrs)
        np.testing.assert_allclose(found.genes["statistic"], [3.0, 7.0, 11.0])
        np.testing.assert_array_equal(found.genes["df"], [1, 2, 1])
        self.assertTrue(np.isnan(found.genes.loc["UTR_b", "p_value"]))
        self.assertTrue(np.isnan(found.genes.loc["UTR_c", "precision"]))
        self.assertListEqual(list(found.sites.index), ["a1", "a2", "b1", "b2", "b3", "c1", "c2"])
        np.testing.assert_allclose(found.sites["p_value"], 0.25)
        self.assertEqual(found.versions, {"R": "4.4.3", "DRIMSeq": "1.34.0"})

    def test_a_failure_shows_rs_message(self):
        utrs, sites, counts, full, reduced = _inputs()
        with self.assertRaisesRegex(RuntimeError, "(?s)DRIMSeq failed:.*something went wrong"):
            drimseq.run(
                utrs,
                sites,
                counts,
                full,
                reduced,
                rscript=simulate.fake_rscript(self.tmp, mode="fail"),
            )


@unittest.skipUnless(simulate.real_rscript(), "needs R with DRIMSeq (set UTRCALLING_TEST_RSCRIPT)")
class TestRealDrimseq(unittest.TestCase):
    def test_drimseq_finds_the_changing_utr(self):
        rng = np.random.default_rng(1)
        group = np.repeat([0.0, 1.0], 6)
        changing = np.array(
            [rng.multinomial(200, [0.8, 0.2] if g == 0 else [0.3, 0.7]) for g in group]
        )
        steady = np.array([rng.multinomial(200, [0.5, 0.3, 0.2]) for _ in group])
        other = np.array([rng.multinomial(150, [0.6, 0.4]) for _ in group])
        full = np.column_stack([np.ones(12), group])
        found = drimseq.run(
            ["changing", "steady", "other"],
            [["c1", "c2"], ["s1", "s2", "s3"], ["o1", "o2"]],
            [changing, steady, other],
            full,
            full[:, :1],
            rscript=simulate.real_rscript(),
        )
        self.assertLess(found.genes.loc["changing", "p_value"], 1e-6)
        self.assertGreater(found.genes.loc["steady", "p_value"], 1e-3)
        self.assertTrue((found.genes["precision"] > 0).all())
        self.assertEqual(found.genes.loc["steady", "df"], 2)
        self.assertIn("DRIMSeq", found.versions)
