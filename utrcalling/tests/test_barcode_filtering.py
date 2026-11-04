import os
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from utrcalling.core import barcode_filtering

os.chdir(str(Path(__file__).resolve().parents[0]))


class TestFilterBarcodesMemoryAware(unittest.TestCase):
    """Test the memory-aware barcode filtering wrapper."""

    def setUp(self):
        self.whitelist = np.array(["AAAA-1", "BBBB-1", "CCCC-1"])
        self.utr_sizes = pd.DataFrame(
            {
                "utr_name": pd.Categorical(["UTR1", "UTR1", "UTR2", "UTR2"]),
                "observation_id": pd.Categorical(["AAAA-1", "BBBB-1", "CCCC-1", "DDDD-1"]),
                "utr_size": np.array([100, 200, 300, 400], dtype=np.int32),
            }
        )

    def test_small_data_single_pass(self):
        """Small datasets should be filtered in a single pass (no chunking)."""
        result = barcode_filtering.filter_barcodes_memory_aware(self.utr_sizes, self.whitelist)
        # DDDD-1 is not in whitelist and has Hamming distance >1, so it's dropped
        self.assertEqual(len(result), 3)
        self.assertNotIn("DDDD-1", result["observation_id"].values)

    def test_preserves_categorical_dtypes(self):
        """Categorical dtypes should be preserved after filtering."""
        result = barcode_filtering.filter_barcodes_memory_aware(self.utr_sizes, self.whitelist)
        self.assertTrue(hasattr(result["utr_name"].dtype, "categories"))
        self.assertTrue(hasattr(result["observation_id"].dtype, "categories"))

    def test_exact_match_from_new_module(self):
        """filter_barcodes should work when imported from the new module."""
        result = barcode_filtering.filter_barcodes(self.utr_sizes, self.whitelist)
        self.assertEqual(len(result), 3)
        self.assertSetEqual(
            set(result["observation_id"].unique()),
            {"AAAA-1", "BBBB-1", "CCCC-1"},
        )


def _utr_sizes(observation_ids, categorical=True):
    """One mapped molecule per observation, with sizes 100, 101, ..."""
    n = len(observation_ids)
    return pd.DataFrame(
        {
            "utr_name": pd.Categorical([f"UTR{i % 2}" for i in range(n)]),
            "observation_id": (
                pd.Categorical(observation_ids)
                if categorical
                else np.array(observation_ids, dtype=object)
            ),
            "utr_size": np.arange(100, 100 + n, dtype=np.int32),
        }
    )


class TestFilterBarcodesCorrection(unittest.TestCase):
    """Barcodes one mismatch away from a whitelisted barcode are corrected to it."""

    def test_one_mismatch_is_corrected_within_its_batch(self):
        sizes = _utr_sizes(["AAAT-1", "AAAA-1", "CCCC-1"])
        result = barcode_filtering.filter_barcodes(sizes, np.array(["AAAA-1", "CCCC-1"]))
        self.assertListEqual(result.observation_id.tolist(), ["AAAA-1", "AAAA-1", "CCCC-1"])
        self.assertListEqual(result.utr_size.tolist(), [100, 101, 102])

    def test_a_barcode_of_another_batch_is_not_a_correction(self):
        # AAAA-2 is one character away from AAAA-1, but that character is the batch.
        sizes = _utr_sizes(["AAAA-1", "CCCC-2"])
        result = barcode_filtering.filter_barcodes(sizes, np.array(["AAAA-2", "CCCC-2"]))
        self.assertListEqual(result.observation_id.tolist(), ["CCCC-2"])

    def test_barcodes_without_batches_are_corrected(self):
        sizes = _utr_sizes(["AAAT", "CCCC", "GGGG"])
        result = barcode_filtering.filter_barcodes(sizes, np.array(["AAAA", "CCCC"]))
        self.assertListEqual(result.observation_id.tolist(), ["AAAA", "CCCC"])

    def test_plain_string_barcodes_are_corrected_and_made_categorical(self):
        sizes = _utr_sizes(["AAAT-1", "CCCC-1"], categorical=False)
        result = barcode_filtering.filter_barcodes(sizes, np.array(["AAAA-1", "CCCC-1"]))
        self.assertListEqual(result.observation_id.tolist(), ["AAAA-1", "CCCC-1"])
        self.assertIsInstance(result.observation_id.dtype, pd.CategoricalDtype)


class TestFilterBarcodesInChunks(unittest.TestCase):
    """Data that would not fit in memory is filtered in chunks, with the same result."""

    def test_chunks_give_the_result_of_one_pass(self):
        sizes = _utr_sizes(["AAAT-1", "AAAA-1", "CCCC-1", "GGGG-1", "CCCG-1"] * 3)
        whitelist = np.array(["AAAA-1", "CCCC-1"])
        expected = barcode_filtering.filter_barcodes(sizes, whitelist)
        # Filtering needs about 3x the data size: this much memory makes 3 chunks.
        memory = mock.Mock(available=sizes.memory_usage(deep=True).sum())
        with (
            mock.patch.object(barcode_filtering.psutil, "virtual_memory", return_value=memory),
            mock.patch.object(
                barcode_filtering, "filter_barcodes", wraps=barcode_filtering.filter_barcodes
            ) as filter_chunk,
        ):
            result = barcode_filtering.filter_barcodes_memory_aware(sizes, whitelist)
        self.assertEqual(filter_chunk.call_count, 3)
        pd.testing.assert_frame_equal(
            result.reset_index(drop=True), expected.reset_index(drop=True)
        )


class TestFilterBarcodesImport(unittest.TestCase):
    """Verify filter_barcodes is importable from the canonical module."""

    def test_import_from_barcode_filtering(self):
        from utrcalling.core.barcode_filtering import filter_barcodes

        self.assertIs(filter_barcodes, barcode_filtering.filter_barcodes)


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2)
