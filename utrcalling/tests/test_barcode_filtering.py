import os
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from utrcalling.core import barcode_filtering

os.chdir(str(Path(__file__).resolve().parents[0]))


class TestFilterBarcodesMemoryAware(unittest.TestCase):
    """Test the memory-aware barcode filtering wrapper."""

    def setUp(self):
        self.whitelist = np.array(["AAAA-1", "BBBB-1", "CCCC-1"])
        self.utr_sizes = pd.DataFrame({
            "utr_name": pd.Categorical(["UTR1", "UTR1", "UTR2", "UTR2"]),
            "observation_id": pd.Categorical(["AAAA-1", "BBBB-1", "CCCC-1", "DDDD-1"]),
            "utr_size": np.array([100, 200, 300, 400], dtype=np.int32),
        })

    def test_small_data_single_pass(self):
        """Small datasets should be filtered in a single pass (no chunking)."""
        result = barcode_filtering.filter_barcodes_memory_aware(
            self.utr_sizes, self.whitelist
        )
        # DDDD-1 is not in whitelist and has Hamming distance >1, so it's dropped
        self.assertEqual(len(result), 3)
        self.assertNotIn("DDDD-1", result["observation_id"].values)

    def test_preserves_categorical_dtypes(self):
        """Categorical dtypes should be preserved after filtering."""
        result = barcode_filtering.filter_barcodes_memory_aware(
            self.utr_sizes, self.whitelist
        )
        self.assertTrue(hasattr(result["utr_name"].dtype, "categories"))
        self.assertTrue(hasattr(result["observation_id"].dtype, "categories"))

    def test_exact_match_from_new_module(self):
        """filter_barcodes should work when imported from the new module."""
        result = barcode_filtering.filter_barcodes(
            self.utr_sizes, self.whitelist
        )
        self.assertEqual(len(result), 3)
        self.assertSetEqual(
            set(result["observation_id"].unique()),
            {"AAAA-1", "BBBB-1", "CCCC-1"},
        )


class TestFilterBarcodesReExport(unittest.TestCase):
    """Verify the re-export from process_mapped_reads still works."""

    def test_re_export_accessible(self):
        from utrcalling.core.process_mapped_reads import filter_barcodes
        self.assertIs(filter_barcodes, barcode_filtering.filter_barcodes)


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2)
