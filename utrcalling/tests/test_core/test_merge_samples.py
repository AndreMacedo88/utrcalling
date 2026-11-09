"""core.merge_samples_utr_sizes: the samples' mapping summaries add up."""

import unittest

import numpy as np
import pandas as pd

from utrcalling.core import core, protocol


def _sample(observation, summary=None):
    frame = pd.DataFrame(
        {
            "utr_name": pd.Categorical(["UTR1", "UTR2"]),
            "observation_id": pd.Categorical([observation] * 2),
            "utr_size": np.array([10, 20], dtype=np.int32),
        }
    )
    if summary is not None:
        frame.attrs["mapping_summary"] = summary
    return frame


class TestMergeSamplesMappingSummary(unittest.TestCase):
    def setUp(self):
        self.protocol = protocol.create(
            "10x", 2, annotations=pd.DataFrame({"CellBarcode": ["A-1"]}), merge_method="union"
        )

    def test_the_merged_summary_sums_the_samples(self):
        samples = [
            _sample("A-1", {"alignments": 10, "reads": 4, "molecules": 2}),
            _sample("B-2", {"alignments": 7, "reads": 3, "molecules": 2}),
        ]
        merged = core.merge_samples_utr_sizes(samples, self.protocol)
        self.assertDictEqual(
            merged.attrs, {"mapping_summary": {"alignments": 17, "reads": 7, "molecules": 4}}
        )
        self.assertEqual(len(merged), 4)

    def test_without_every_samples_summary_the_merge_has_none(self):
        samples = [_sample("A-1", {"molecules": 2}), _sample("B-2")]
        merged = core.merge_samples_utr_sizes(samples, self.protocol)
        self.assertDictEqual(merged.attrs, {})


if __name__ == "__main__":
    unittest.main()
