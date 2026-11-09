"""Tests for utrcalling.core.qc: the quality control tables."""

import unittest

import pandas as pd

from utrcalling.core import qc
from utrcalling.core.utr_mapping import MAPPING_STEPS


def _steps(**counts):
    return dict.fromkeys(MAPPING_STEPS, 0) | counts


class TestMappingSummary(unittest.TestCase):
    def test_a_row_per_input_and_a_total(self):
        summary = qc.mapping_summary(
            ["a.bam", "b.bam"],
            [
                _steps(alignments=10, wrong_strand=4, reads=6, molecules=5),
                _steps(alignments=8, outside_utr=1, reads=7, molecules=7),
            ],
            {
                "molecules_after_merge": 12,
                "molecules_after_size_filter": 11,
                "molecules_in_cell_list": 9,
                "molecules_counted": 9,
            },
        )
        self.assertListEqual(list(summary.index), ["a.bam", "b.bam", "total"])
        self.assertEqual(summary.index.name, "input")
        self.assertListEqual(list(summary.columns), [*MAPPING_STEPS, *qc.COUNTING_STEPS])
        self.assertTrue((summary.dtypes == "Int64").all())
        total = summary.loc["total"]
        self.assertEqual(total["alignments"], 18)
        self.assertEqual(total["wrong_strand"], 4)
        self.assertEqual(total["molecules"], 12)
        self.assertEqual(total["molecules_counted"], 9)
        # The counting steps are done on all inputs together.
        self.assertTrue(summary.loc["a.bam", list(qc.COUNTING_STEPS)].isna().all())

    def test_stored_mappings_without_their_mapping_steps(self):
        summary = qc.mapping_summary(
            ["a.parquet", "b.parquet"],
            [{"molecules": 5}, {"molecules": 7}],
            {"molecules_after_merge": 12, "molecules_counted": 10},
        )
        self.assertListEqual(
            list(summary.columns), ["molecules", "molecules_after_merge", "molecules_counted"]
        )
        self.assertListEqual(summary.molecules.tolist(), [5, 7, 12])

    def test_inputs_with_different_steps_share_only_the_common_ones(self):
        summary = qc.mapping_summary(
            ["a.parquet", "b.parquet"], [_steps(alignments=10, molecules=5), {"molecules": 7}], {}
        )
        self.assertEqual(summary.loc["a.parquet", "alignments"], 10)
        self.assertTrue(pd.isna(summary.loc["b.parquet", "alignments"]))
        self.assertTrue(pd.isna(summary.loc["total", "alignments"]))
        self.assertEqual(summary.loc["total", "molecules"], 12)


class TestJoinMappingSummaries(unittest.TestCase):
    """Concatenated experiments: the inputs of each, and their totals summed."""

    def setUp(self):
        self.first = qc.mapping_summary(
            ["a.bam"], [_steps(alignments=10, reads=6, molecules=5)], {"molecules_counted": 4}
        )
        self.second = qc.mapping_summary(
            ["b.bam", "c.bam"],
            [
                _steps(alignments=8, reads=7, molecules=7),
                _steps(alignments=2, reads=1, molecules=1),
            ],
            {"molecules_counted": 7},
        )

    def test_inputs_of_every_experiment_and_summed_totals(self):
        joined = qc.join_mapping_summaries([self.first, self.second])
        self.assertListEqual(list(joined.index), ["a.bam", "b.bam", "c.bam", "total"])
        self.assertListEqual(list(joined.columns), list(self.first.columns))
        self.assertTrue((joined.dtypes == "Int64").all())
        total = joined.loc["total"]
        self.assertEqual(total.alignments, 20)
        self.assertEqual(total.molecules, 13)
        self.assertEqual(total.molecules_counted, 11)

    def test_a_step_that_one_experiment_lacks_has_no_total(self):
        stored = qc.mapping_summary(["m.parquet"], [{"molecules": 3}], {"molecules_counted": 3})
        total = qc.join_mapping_summaries([self.first, stored]).loc["total"]
        self.assertTrue(pd.isna(total.alignments))
        self.assertEqual(total.molecules, 8)

    def test_an_experiment_without_a_summary(self):
        self.assertIsNone(qc.join_mapping_summaries([self.first, None]))


if __name__ == "__main__":
    unittest.main()
