"""Tests for utrcalling.core.stats.site_usage: the test of differential site usage."""

import itertools
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import scipy.sparse

from utrcalling.core import provenance
from utrcalling.core.stats import dirmult
from utrcalling.core.stats import site_usage as su
from utrcalling.tests.test_stats import simulate

PSEUDOTIME = "~ C(batch) + bs(pseudotime, df=3)"
BATCH = "~ C(batch)"


class TestTheTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cells = simulate.site_counts(n_units=500, n_utrs=10, changing=4, strength=2.5)
        cls.result = su.site_usage(cls.cells, PSEUDOTIME, BATCH, replicates="sample", n_jobs=1)

    def test_changing_utrs_stand_out(self):
        genes = self.result.genes
        changing = [f"UTR{i}" for i in range(4)]
        steady = [f"UTR{i}" for i in range(4, 10)]
        self.assertTrue((genes.loc[changing, "p_adjusted"] < 0.01).all())
        self.assertGreater(genes.loc[steady, "p_value"].min(), 0.001)

    def test_columns_and_statuses(self):
        genes = self.result.genes
        self.assertTrue((genes["status"] == "ok").all())
        self.assertListEqual(list(genes.index), [f"UTR{i}" for i in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)])
        for column in (
            "gene_name",
            "n_sites",
            "n_units",
            "total_counts",
            "statistic",
            "df",
            "p_value",
            "p_adjusted",
            "precision",
            "precision_target",
            "delta_length_index",
            "non_monotonicity",
            "rho_length_index",
            "max_variance_ratio",
        ):
            self.assertIn(column, genes)
        self.assertNotIn("permutation_p_value", genes)
        # 3 spline columns, times sites - 1.
        np.testing.assert_array_equal(genes["df"], 3 * (genes["n_sites"] - 1))
        self.assertListEqual(genes["gene_name"].tolist()[:2], ["G0", "G1"])

    def test_effects_follow_the_simulated_slopes(self):
        genes = self.result.genes
        # Slopes rise from the shortest site to the longest: longer UTRs later.
        self.assertTrue(
            (genes.loc[[f"UTR{i}" for i in range(4)], "delta_length_index"] > 0.2).all()
        )
        self.assertTrue((genes.loc[[f"UTR{i}" for i in range(4)], "rho_length_index"] > 0).all())
        self.assertLess(
            genes.loc[[f"UTR{i}" for i in range(4, 10)], "delta_length_index"].abs().max(), 0.2
        )

    def test_sites_table(self):
        sites = self.result.sites
        self.assertEqual(len(sites), self.cells.n_vars)
        first = sites[sites["utr_name"] == "UTR1"]
        np.testing.assert_allclose(first["rank"], [0.0, 0.5, 1.0])
        np.testing.assert_allclose(first["proportion_from"].sum(), 1.0)
        np.testing.assert_allclose(first["delta_proportion"].sum(), 0.0, atol=1e-12)
        self.assertGreater(first["delta_proportion"].iloc[-1], 0)

    def test_settings(self):
        settings = self.result.settings
        self.assertEqual(settings["engine"], "dirmult")
        self.assertEqual(settings["tested"], ["pseudotime"])
        self.assertEqual(settings["effect_of"], "pseudotime")
        self.assertEqual(settings["units"], 500)
        self.assertNotIn("permutation", settings)
        json.dumps(settings)


class TestStatuses(unittest.TestCase):
    def test_utrs_that_cannot_be_tested_say_why(self):
        cells = simulate.site_counts(n_units=80, n_utrs=6, changing=0, depth=1.0)
        dense = cells.X.toarray()
        dense[:, 0] = 0  # UTR0 keeps one site with molecules
        dense[20:, 2:5] = 0  # UTR1 (sites 2-4) keeps its first 20 units
        dense[:, 5:7] = 0
        dense[:3, 5:7] = 1  # UTR2: 6 molecules in 3 units
        cells.X = scipy.sparse.csr_matrix(dense)
        genes = su.site_usage(cells, "~ condition", min_units=4, min_counts=10, n_jobs=1).genes
        self.assertEqual(genes.loc["UTR0", "status"], "single_site")
        self.assertEqual(genes.loc["UTR2", "status"], "too_few_units")
        self.assertEqual(genes.loc["UTR1", "status"], "not_identifiable")  # one condition only
        self.assertEqual(genes.loc["UTR1", "df"], 0)
        genes = su.site_usage(cells, "~ condition", min_units=2, min_counts=10, n_jobs=1).genes
        self.assertEqual(genes.loc["UTR2", "status"], "too_few_counts")
        self.assertTrue(np.isnan(genes.loc["UTR2", "p_adjusted"]))

    def test_no_more_units_than_columns(self):
        cells = simulate.site_counts(n_units=40, n_utrs=2, changing=0)
        dense = cells.X.toarray()
        dense[:, :2] = 0
        dense[[0, 10], :2] = 5  # two units for intercept and slope: a perfect fit
        cells.X = scipy.sparse.csr_matrix(dense)
        genes = su.site_usage(cells, "~ pseudotime", min_units=2, n_jobs=1).genes
        self.assertEqual(genes.loc["UTR0", "status"], "too_few_units")
        self.assertEqual(genes.loc["UTR1", "status"], "ok")

    def test_no_utr_tested(self):
        cells = simulate.site_counts(n_units=40, n_utrs=2, changing=0)
        result = su.site_usage(cells, "~ condition", min_units=1000, n_jobs=1)
        self.assertTrue((result.genes["status"] == "too_few_units").all())
        self.assertTrue(result.genes["p_adjusted"].isna().all())
        self.assertNotIn("df", result.genes)

    def test_fits_that_do_not_converge_are_left_out_of_the_adjustment(self):
        cells = simulate.site_counts(n_units=100, n_utrs=3, changing=1)
        real = dirmult.fit

        def unconverged(*args, **kwargs):
            fitted = real(*args, **kwargs)
            return dirmult.Fit(fitted.beta, fitted.precision, fitted.log_likelihood, False)

        with mock.patch.object(su.dirmult, "fit", side_effect=unconverged):
            genes = su.site_usage(cells, "~ condition", permute=True, n_jobs=1).genes
        self.assertTrue((genes["status"] == "not_converged").all())
        self.assertTrue(genes["p_value"].notna().all())
        self.assertTrue(genes["p_adjusted"].isna().all())
        # Nor are they permuted: their statistic is unsure.
        self.assertNotIn("permutation_p_value", genes)

    def test_a_full_fit_worse_than_the_reduced_one_is_fitted_again(self):
        cells = simulate.site_counts(n_units=150, n_utrs=1, changing=1, effect="condition")
        real = dirmult.fit
        calls = []

        def stuck_once(design, counts, precision=None, **kwargs):
            calls.append(precision)
            fitted = real(design, counts, precision, **kwargs)
            if len(calls) == 1:  # the first full fit stops at a poor point
                return dirmult.Fit(fitted.beta * 0, fitted.precision, -1e9, True)
            return fitted

        with mock.patch.object(su.dirmult, "fit", side_effect=stuck_once):
            genes = su.site_usage(cells, "~ condition", n_jobs=1).genes
        self.assertEqual(len(calls), 3)
        self.assertEqual(genes.loc["UTR0", "status"], "ok")
        self.assertLess(genes.loc["UTR0", "p_value"], 1e-3)

    def test_a_refit_that_does_not_help_is_dropped_and_the_utr_flagged(self):
        cells = simulate.site_counts(n_units=150, n_utrs=1, changing=1)
        real = dirmult.fit

        def always_poor(design, counts, precision=None, **kwargs):
            fitted = real(design, counts, precision, **kwargs)
            poor = design.shape[1] > 1  # the full model never fits
            likelihood = -1e9 if poor else fitted.log_likelihood
            return dirmult.Fit(fitted.beta, fitted.precision, likelihood, True)

        with mock.patch.object(su.dirmult, "fit", side_effect=always_poor):
            genes = su.site_usage(cells, "~ condition", n_jobs=1).genes
        self.assertEqual(genes.loc["UTR0", "status"], "optimization_failed")
        self.assertTrue(np.isnan(genes.loc["UTR0", "statistic"]))


class TestPrecisionModeration(unittest.TestCase):
    def test_targets(self):
        cells = simulate.site_counts(n_units=200, n_utrs=12, changing=0, depth=5)
        trended = su.site_usage(cells, "~ condition", n_jobs=1).genes["precision_target"]
        self.assertGreater(trended.nunique(), 1)
        common = su.site_usage(cells, "~ condition", precision_moderation="common", n_jobs=1).genes
        self.assertEqual(common["precision_target"].nunique(), 1)
        free = su.site_usage(cells, "~ condition", precision_moderation="none", n_jobs=1).genes
        self.assertTrue(free["precision_target"].isna().all())

    def test_few_utrs_share_a_common_target(self):
        cells = simulate.site_counts(n_units=200, n_utrs=4, changing=0, depth=5)
        genes = su.site_usage(cells, "~ condition", n_jobs=1).genes
        self.assertEqual(genes["precision_target"].nunique(), 1)

    def test_single_molecules_give_no_target(self):
        cells = simulate.site_counts(n_units=200, n_utrs=3, changing=0)
        dense = cells.X.toarray()
        cells.X = scipy.sparse.csr_matrix(
            np.eye(dense.shape[1], dtype=int)[np.argmax(dense, axis=1) % dense.shape[1]]
            * (dense.sum(1, keepdims=True) > 0)
        )
        with self.assertLogs("utrcalling", level="INFO") as logs:
            genes = su.site_usage(cells, "~ condition", min_counts=1, n_jobs=1).genes
        self.assertTrue(any("no moderation" in line for line in logs.output))
        self.assertTrue(genes["precision_target"].isna().all())


class TestReplicates(unittest.TestCase):
    def setUp(self):
        self.cells = simulate.site_counts(n_units=120, n_utrs=3, changing=1, effect="condition")

    def test_a_sample_level_variable_needs_samples_or_permutations(self):
        with self.assertRaisesRegex(
            ValueError, "condition is the same for all units of each sample"
        ):
            su.site_usage(self.cells, "~ condition", replicates="sample")
        with self.assertRaisesRegex(ValueError, "use the dirmult engine with --permute"):
            su.site_usage(self.cells, "~ condition", replicates="sample", engine="drimseq")

    def test_with_permutations_it_runs_and_warns(self):
        with self.assertLogs("utrcalling", level="WARNING") as logs:
            result = su.site_usage(
                self.cells, "~ condition", replicates="sample", permute=True, n_jobs=1
            )
        self.assertIn("Read the permutation p-values", logs.output[0])
        genes = result.genes
        # 4 samples, 2 per condition: 6 distinct swaps, all of them.
        self.assertTrue((genes["permutations"] == 6).all())
        self.assertTrue(genes["permutation_exact"].all())
        self.assertGreaterEqual(genes["permutation_p_value"].min(), 1 / 6)
        self.assertEqual(result.settings["permutation"]["shuffles"], "between replicates")
        self.assertEqual(result.settings["contrast"], {"from": "control", "to": "treated"})
        self.assertIn("observed_delta_length_index", genes)
        self.assertIn("permutation_p_adjusted", genes)

    def test_units_that_are_their_own_replicates(self):
        samples = self.cells.copy()
        samples.obs["sample"] = samples.obs_names
        genes = su.site_usage(samples, "~ condition", replicates="sample", n_jobs=1).genes
        self.assertTrue((genes["status"] == "ok").all())

    def test_strata_and_column_errors(self):
        with self.assertRaisesRegex(ValueError, "add --permute"):
            su.site_usage(self.cells, "~ condition", strata="family")
        with self.assertRaisesRegex(ValueError, "No column donor"):
            su.site_usage(self.cells, "~ condition", replicates="donor")
        missing = self.cells.copy()
        missing.obs["sample"] = missing.obs["sample"].where(missing.obs["batch"] > 0)
        with self.assertRaisesRegex(ValueError, "units of the models have no sample"):
            su.site_usage(missing, "~ condition", replicates="sample")

    def test_permuting_within_samples(self):
        cells = simulate.site_counts(n_units=200, n_utrs=3, changing=1, strength=3)
        result = su.site_usage(
            cells,
            PSEUDOTIME,
            BATCH,
            replicates="sample",
            permute=True,
            permutations=(9, 19),
            n_jobs=1,
        )
        genes = result.genes
        self.assertAlmostEqual(genes.loc["UTR0", "permutation_p_value"], 1 / 20)
        self.assertEqual(genes.loc["UTR0", "permutations"], 19)
        self.assertFalse(genes["permutation_exact"].any())
        self.assertEqual(result.settings["permutation"]["stages"], [9, 19])

    def test_paired_samples_swap_within_families(self):
        samples = simulate.site_counts(
            n_units=8, n_samples=8, n_utrs=3, changing=1, depth=200, effect="condition"
        )
        samples.obs["family"] = [f"f{i % 4}" for i in range(8)]
        result = su.site_usage(
            samples,
            "~ family + condition",
            "~ family",
            permute=True,
            strata="family",
            min_units=2,
            n_jobs=1,
        )
        # Four families of two samples: 2^4 swaps.
        self.assertTrue((result.genes["permutations"] == 16).all())
        self.assertEqual(result.settings["permutation"]["strata"], "family")


class TestOptionsAndInputs(unittest.TestCase):
    def setUp(self):
        self.cells = simulate.site_counts(n_units=60, n_utrs=2, changing=0)

    def test_option_errors(self):
        for kwargs, message in (
            ({"engine": "deseq"}, "Unknown engine"),
            ({"precision_moderation": "heavy"}, "Unknown precision moderation"),
            ({"permutations": (99, 49)}, "increasing totals"),
            ({"permutations": ()}, "increasing totals"),
            ({"min_units": 1}, "2 units or more"),
            ({"permute": True, "engine": "drimseq"}, "use the dirmult engine"),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                su.site_usage(self.cells, "~ condition", **kwargs)

    def test_matrices_that_are_not_site_counts_are_refused(self):
        raw = self.cells.copy()
        provenance.record(raw, matrix="raw_sizes")
        with self.assertRaisesRegex(ValueError, "raw sizes"):
            su.site_usage(raw, "~ condition")
        fractions = self.cells.copy()
        provenance.record(fractions, matrix="fractions")
        with self.assertRaisesRegex(ValueError, "holds fractions"):
            su.site_usage(fractions, "~ condition")
        normalised = self.cells.copy()
        normalised.X = normalised.X * 0.5
        with self.assertRaisesRegex(ValueError, "not counts"):
            su.site_usage(normalised, "~ condition")
        single = self.cells[:, [0, 2]].copy()
        with self.assertRaisesRegex(ValueError, "No UTR has two or more sites"):
            su.site_usage(single, "~ condition")
        with self.assertRaisesRegex(ValueError, "has no graph"):
            su.site_usage(self.cells, "~ condition", smooth_graph="connectivities")

    def test_without_gene_names_and_without_effects(self):
        unnamed = self.cells.copy()
        unnamed.var = unnamed.var.drop(columns="gene_name")
        result = su.site_usage(
            unnamed, "~ C(batch) * pseudotime", "~ C(batch) + pseudotime", n_jobs=1
        )
        self.assertNotIn("gene_name", result.genes)
        self.assertNotIn("delta_length_index", result.genes)
        self.assertNotIn("proportion_from", result.sites)
        self.assertNotIn("effect_of", result.settings)

    def test_effects_of_a_chosen_variable_and_contrast(self):
        result = su.site_usage(
            self.cells,
            "~ condition + pseudotime",
            effect_of="condition",
            contrast=("control", "treated"),
            n_jobs=1,
        )
        self.assertEqual(result.settings["contrast"], {"from": "treated", "to": "control"})

    def test_smoothed_descriptive_measures(self):
        cells = simulate.site_counts(n_units=200, n_utrs=3, changing=1, strength=3)
        order = np.argsort(cells.obs["pseudotime"].to_numpy())
        graph = np.zeros((200, 200))
        for left, right in itertools.pairwise(order):
            graph[left, right] = graph[right, left] = 1
        cells.obsp["neighbours"] = scipy.sparse.csr_matrix(graph)
        raw = su.site_usage(cells, PSEUDOTIME, BATCH, n_jobs=1)
        smoothed = su.site_usage(
            cells, PSEUDOTIME, BATCH, smooth_graph="neighbours", smooth_rounds=3, n_jobs=1
        )
        self.assertGreater(
            smoothed.genes.loc["UTR0", "rho_length_index"],
            raw.genes.loc["UTR0", "rho_length_index"],
        )
        # The test itself does not change.
        pd.testing.assert_series_equal(smoothed.genes["p_value"], raw.genes["p_value"])
        self.assertEqual(smoothed.settings["smoothing"], "neighbours x 3")

    def test_parallel_runs_give_the_same_results(self):
        cells = simulate.site_counts(n_units=80, n_utrs=26, changing=2)
        serial = su.site_usage(
            cells,
            PSEUDOTIME,
            BATCH,
            replicates="sample",
            permute=True,
            permutations=(5, 9),
            n_jobs=1,
        )
        parallel = su.site_usage(
            cells,
            PSEUDOTIME,
            BATCH,
            replicates="sample",
            permute=True,
            permutations=(5, 9),
            n_jobs=3,
        )
        pd.testing.assert_frame_equal(serial.genes, parallel.genes)
        pd.testing.assert_frame_equal(serial.sites, parallel.sites)


class TestDrimseqEngine(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.samples = simulate.site_counts(
            n_units=12, n_samples=12, n_utrs=4, changing=1, depth=150, effect="condition"
        )

    def test_drimseqs_test_with_effects_from_the_same_model(self):
        result = su.site_usage(
            self.samples,
            "~ condition",
            engine="drimseq",
            rscript=simulate.fake_rscript(self.tmp),
            n_jobs=1,
        )
        genes = result.genes
        self.assertListEqual(genes["status"].tolist(), ["ok", "not_tested", "not_tested", "ok"])
        np.testing.assert_allclose(genes.loc[["UTR0", "UTR3"], "statistic"], [3.0, 15.0])
        self.assertTrue(np.isnan(genes.loc["UTR1", "statistic"]))
        self.assertNotIn("precision_target", genes)
        self.assertTrue(np.isfinite(genes.loc["UTR0", "delta_length_index"]))
        self.assertTrue(np.isnan(genes.loc["UTR2", "delta_length_index"]))
        np.testing.assert_allclose(result.sites["p_value"], 0.25)
        self.assertIn("p_adjusted", result.sites)
        self.assertEqual(result.settings["drimseq"], "1.34.0")
        self.assertEqual(result.settings["versions"]["R"], "4.4.3")

    @unittest.skipUnless(
        simulate.real_rscript(), "needs R with DRIMSeq (set UTRCALLING_TEST_RSCRIPT)"
    )
    def test_with_real_drimseq(self):
        genes = su.site_usage(
            self.samples, "~ condition", engine="drimseq", rscript=simulate.real_rscript(), n_jobs=1
        ).genes
        self.assertLess(genes.loc["UTR0", "p_value"], 1e-3)
        self.assertTrue((genes["status"] == "ok").all())
        self.assertTrue((genes["precision"] > 0).all())
