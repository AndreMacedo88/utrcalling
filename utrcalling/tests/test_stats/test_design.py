"""Tests for utrcalling.core.stats.design: formulas into design matrices."""

import unittest

import numpy as np
import pandas as pd

from utrcalling.core.stats import design


def _obs(n=60, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "batch": np.tile(["b1", "b2", "b3"], n // 3),
            "batch_number": np.tile([1, 2, 3], n // 3),
            "pseudotime": rng.random(n),
            "condition": pd.Categorical(
                np.repeat(["treated", "control"], n // 2), categories=["control", "treated"]
            ),
            "dose": np.repeat([0.0, 1.5], n // 2),
            "flag": np.tile([True, False], n // 2),
            "depth": rng.normal(1000, 100, n),
        },
        index=[f"cell{i}" for i in range(n)],
    )


class TestIndependentColumns(unittest.TestCase):
    def test_columns_that_combine_others_are_left_out(self):
        matrix = np.array([[1, 0, 1, 2], [1, 1, 0, 2], [1, 0, 1, 2], [1, 1, 0, 2.0]])
        kept = design.independent_columns(matrix)
        self.assertEqual(len(kept), 2)
        self.assertEqual(design.matrix_rank(matrix), 2)
        self.assertEqual(design.matrix_rank(matrix[:, kept]), 2)

    def test_empty_and_zero_matrices_have_no_columns(self):
        self.assertEqual(len(design.independent_columns(np.zeros((3, 0)))), 0)
        self.assertEqual(len(design.independent_columns(np.zeros((3, 2)))), 0)


class TestBuild(unittest.TestCase):
    def test_full_and_reduced_matrices(self):
        built = design.build(_obs(), "~ batch + bs(pseudotime, df=4)", "~ batch")
        self.assertEqual(built.full.shape, (60, 7))
        self.assertEqual(built.reduced.shape, (60, 3))
        self.assertEqual(built.full_columns[:3], ("Intercept", "batch[T.b2]", "batch[T.b3]"))
        self.assertEqual(built.reduced_columns, ("Intercept", "batch[T.b2]", "batch[T.b3]"))
        self.assertEqual(built.tested, ("pseudotime",))
        self.assertEqual(built.reduced_variables, ("batch",))
        self.assertEqual(built.categorical, frozenset({"batch"}))
        self.assertTrue(built.kept.all())
        self.assertListEqual(list(built.data.columns), ["batch", "pseudotime"])

    def test_the_reduced_model_defaults_to_an_intercept(self):
        built = design.build(_obs(), "~ condition")
        self.assertEqual(built.reduced_columns, ("Intercept",))
        self.assertEqual(built.tested, ("condition",))

    def test_categories_are_text_booleans_and_numbers_in_c(self):
        built = design.build(_obs(), "~ condition + flag + C(batch_number) + dose + depth")
        self.assertEqual(built.categorical, frozenset({"condition", "flag", "batch_number"}))

    def test_whole_numbers_outside_c_are_flagged(self):
        with self.assertLogs("utrcalling", level="WARNING") as logs:
            design.build(_obs(), "~ batch_number + pseudotime", "~ batch_number")
        self.assertIn("batch_number holds 3 whole numbers", logs.output[0])
        self.assertIn("C(batch_number)", logs.output[0])

    def test_units_with_missing_values_are_left_out(self):
        obs = _obs()
        obs.loc[obs.index[:5], "pseudotime"] = np.nan
        with self.assertLogs("utrcalling", level="INFO") as logs:
            built = design.build(obs, "~ batch + pseudotime", "~ batch")
        self.assertIn("Left out 5 of 60 units", logs.output[0])
        self.assertEqual(built.full.shape[0], 55)
        self.assertFalse(built.kept[:5].any())
        self.assertTrue(built.kept[5:].all())

    def test_combined_columns_are_reported(self):
        obs = _obs()
        obs["twice"] = obs["depth"] * 2
        with self.assertLogs("utrcalling", level="WARNING") as logs:
            design.build(obs, "~ depth + twice + pseudotime", "~ depth + twice")
        self.assertIn("1 columns of the full model are combinations", logs.output[0])

    def test_errors(self):
        obs = _obs()
        with self.assertRaisesRegex(ValueError, "without a response"):
            design.build(obs, "y ~ pseudotime")
        with self.assertRaisesRegex(ValueError, "without a response"):
            design.build(obs, "pseudotime")
        with self.assertRaisesRegex(ValueError, "Could not read the full model"):
            design.build(obs, "~ batch +")
        with self.assertRaisesRegex(ValueError, "The models use donor"):
            design.build(obs, "~ donor")
        with self.assertRaisesRegex(ValueError, "Could not evaluate"):
            design.build(obs, "~ bs(batch, df=3)")
        with self.assertRaisesRegex(ValueError, "not nested"):
            design.build(obs, "~ pseudotime", "~ batch")
        with self.assertRaisesRegex(ValueError, "adds nothing"):
            design.build(obs, "~ batch + condition", "~ batch + dose")
        empty = obs.assign(pseudotime=np.nan)
        with self.assertRaisesRegex(ValueError, "No unit has a value"):
            design.build(empty, "~ pseudotime")

    def test_an_empty_formula_is_an_intercept(self):
        built = design.build(_obs(), "~ condition", "~")
        self.assertEqual(built.reduced_columns, ("Intercept",))


class TestEvaluation(unittest.TestCase):
    def test_splines_keep_their_knots_on_other_values(self):
        built = design.build(_obs(), "~ batch + bs(pseudotime, df=4)", "~ batch")
        np.testing.assert_allclose(built.evaluate_full(built.data.iloc[:7]), built.full[:7])
        shuffled = built.data.iloc[::-1]
        np.testing.assert_allclose(built.evaluate_full(shuffled), built.full[::-1])

    def test_points_set_the_variable_for_every_unit(self):
        built = design.build(_obs(), "~ batch + bs(pseudotime, df=4)", "~ batch")
        contrast = built.contrast()
        rows = np.array([0, 5, 9])
        stacked = built.at_points(rows, contrast)
        self.assertEqual(stacked.shape, (design.GRID_POINTS, 3, 7))
        # Batch columns stay each unit's own; spline columns are the point's.
        np.testing.assert_allclose(
            stacked[:, :, :3], np.broadcast_to(built.full[rows, :3], (50, 3, 3))
        )
        np.testing.assert_allclose(stacked[0, 0, 3:], stacked[0, 2, 3:])

    def test_categorical_points_keep_the_categories(self):
        built = design.build(_obs(), "~ condition + pseudotime", "~ pseudotime")
        contrast = built.contrast()
        stacked = built.at_points(np.arange(4), contrast)
        np.testing.assert_allclose(stacked[0, :, 1], 0.0)
        np.testing.assert_allclose(stacked[1, :, 1], 1.0)


class TestContrast(unittest.TestCase):
    def setUp(self):
        self.obs = _obs()

    def test_a_numeric_variable_is_read_between_two_percentiles(self):
        built = design.build(self.obs, "~ batch + pseudotime", "~ batch")
        contrast = built.contrast(effect_range=(20, 80))
        low, high = np.percentile(self.obs["pseudotime"], [20, 80])
        self.assertEqual(contrast.variable, "pseudotime")
        self.assertEqual(contrast.kind, "numeric")
        self.assertAlmostEqual(contrast.start, low)
        self.assertAlmostEqual(contrast.end, high)
        self.assertEqual(len(contrast.points), design.GRID_POINTS)
        self.assertAlmostEqual(contrast.points[-1], high)

    def test_categories_compare_the_last_level_with_the_first(self):
        built = design.build(self.obs, "~ condition")
        contrast = built.contrast()
        # The categorical's own order, not the alphabet.
        self.assertEqual((contrast.start, contrast.end), ("control", "treated"))
        self.assertEqual(contrast.points, ("control", "treated"))
        text = design.build(self.obs, "~ batch").contrast()
        self.assertEqual((text.start, text.end), ("b1", "b3"))

    def test_chosen_levels_are_matched_as_text(self):
        built = design.build(self.obs, "~ C(batch_number)")
        contrast = built.contrast(levels=("2", "3"))
        self.assertEqual((contrast.start, contrast.end), (3, 2))

    def test_another_variable_of_the_full_model(self):
        built = design.build(self.obs, "~ batch + pseudotime", "~ batch")
        self.assertEqual(built.contrast("batch").kind, "categorical")

    def test_no_contrast_when_only_an_interaction_is_tested(self):
        built = design.build(self.obs, "~ batch * pseudotime", "~ batch + pseudotime")
        self.assertEqual(built.tested, ())
        self.assertIsNone(built.contrast())

    def test_errors(self):
        built = design.build(self.obs, "~ condition + pseudotime")
        with self.assertRaisesRegex(ValueError, "adds several variables"):
            built.contrast()
        with self.assertRaisesRegex(ValueError, "depth is not a variable"):
            built.contrast("depth")
        with self.assertRaisesRegex(ValueError, "has no level sick"):
            built.contrast("condition", levels=("sick", "control"))
        with self.assertRaisesRegex(ValueError, "two different levels"):
            built.contrast("condition", levels=("control", "control"))
        with self.assertRaisesRegex(ValueError, "two percentiles"):
            built.contrast("pseudotime", effect_range=(90, 10))
